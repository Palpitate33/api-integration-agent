"""DeepSeek Planner 的 Prompt Builder：把规划输入渲染成体积受控的 prompt。

设计约束：
    - Planner 只能使用 PlannerState 里已有的 APIInfo / ProjectStructure / Evidence，
      本模块**不读取文件系统**、不扫描仓库、不发网络请求。
    - 上下文总体积有硬上限；Evidence 按"HTTP 客户端 → 认证 → 业务 → 模型 →
      测试 → 配置"的优先级裁剪，绝不把整个仓库塞给模型。
    - System Prompt 与 User Prompt 模板集中在这里，不散落在 planner 实现中。

关于 endpoints 的特殊约定：
    APIEndpoint 是嵌套很深的结构化对象（参数 / 请求体 / 响应 schema），
    让 LLM 逐字复现既浪费上下文又必然出错。因此 prompt 要求模型只输出
    **端点标识**（method + path + purpose），由 DeepSeekPlanner 用真实的
    APIEndpoint 对象回填——这就是"LLM 负责判断，程序负责约束"。
"""

import json

from integration_agent.agent.state import PlannerState
from integration_agent.api import APIInfo, APIResponse
from integration_agent.repository import ProjectStructure

# ---------------------------------------------------------------- 上下文预算

DEFAULT_MAX_PROMPT_CHARS = 20_000  # 整个 prompt 的硬上限
DEFAULT_MAX_REQUEST_CHARS = 4_000  # 用户需求段上限
DEFAULT_MAX_API_CHARS = 6_000  # API 信息段上限
DEFAULT_MAX_STRUCTURE_CHARS = 3_000  # 仓库结构段上限
DEFAULT_MAX_EVIDENCE_CHARS = 9_000  # 证据段上限
DEFAULT_MAX_SNIPPET_CHARS = 800  # 单条证据片段上限
DEFAULT_MAX_EVIDENCE_ITEMS = 40  # 证据条目数上限

# 证据裁剪优先级（序号越小越先保留）。key 用于在 path 中匹配。
EVIDENCE_PRIORITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("http_client", ("client", "http", "session", "transport", "request", "connector")),
    ("authentication", ("auth", "token", "credential", "login", "oauth", "secret")),
    ("service", ("service", "api", "endpoint", "handler", "resource", "view", "router")),
    ("model", ("model", "schema", "dto", "entity", "type", "dataclass")),
    ("test", ("test", "spec", "fixture", "conftest")),
    ("config", ("config", "setting", "settings", "env", "const")),
)
UNCLASSIFIED_PRIORITY = len(EVIDENCE_PRIORITY)  # 未命中任何关键词 → 最低优先级

# ---------------------------------------------------------------- System Prompt

SYSTEM_PROMPT = """\
You are APIForge Planner, an expert software integration planning agent.

Your task is to analyze a third-party API specification together with an existing \
software repository and produce a safe, evidence-based integration plan.

You are NOT a code generator.

You must NOT invent repository files, functions, classes, dependencies, frameworks, \
or conventions that are not supported by the provided evidence.

Your responsibility is to decide HOW the API should be integrated into the existing project.

You must reason about:

1. Existing project architecture
2. Existing HTTP/API clients
3. Existing authentication mechanisms
4. Existing service/business modules
5. Existing models/types
6. Existing configuration mechanisms
7. Existing error-handling conventions
8. Existing testing conventions
9. Required dependencies
10. Files that should be modified
11. Files that should be created
12. API endpoint integration strategy

Evidence has priority over assumptions.

When evidence is insufficient, explicitly state uncertainty instead of inventing information.

Prefer reusing existing abstractions over introducing new abstractions.

Prefer minimal changes over unnecessary architectural changes.

Do not redesign the existing project.

Do not modify unrelated files.

Do not expose, invent, or request secrets.

Authentication must reference environment variables or existing configuration mechanisms \
rather than hardcoded credentials.

Every important file decision should be supported by repository evidence when possible.

Your output MUST be valid JSON.

The JSON must represent an IntegrationPlan compatible with the APIForge IntegrationPlan \
Pydantic model.

Do not output Markdown.

Do not output explanations outside the JSON object.\
"""

# ---------------------------------------------------------------- User Prompt 模板

USER_PROMPT_TEMPLATE = """\
You are planning an API integration for the following project.

========================
INTEGRATION REQUEST
========================

{integration_request}

========================
API INFORMATION
========================

{api_info_json}

========================
PROJECT STRUCTURE
========================

{project_structure_json}

========================
PROJECT CONSTRAINTS
========================

{project_constraints}

========================
REPOSITORY EVIDENCE
========================

{evidence_json}

========================
PLANNING RULES
========================

1. Only use files and symbols supported by the evidence.
2. Reuse existing HTTP clients when available.
3. Reuse existing authentication mechanisms when available.
4. Reuse existing models when appropriate.
5. Follow the project's existing testing conventions.
6. Minimize the number of changed files.
7. Do not modify unrelated code.
8. Do not invent dependencies unless the API genuinely requires them.
9. Never hardcode credentials or API keys.
10. If information is missing, record the uncertainty in warnings.
11. Every selected file must have a concrete reason.
12. Every important decision should reference evidence.
13. Output JSON only.

========================
OUTPUT FORMAT
========================

Return a single JSON object shaped exactly like this skeleton (respond in json only — \
no prose, no code fences). Keys are required; replace every placeholder with your decision:

{output_skeleton}

Additional rules for this output shape:

- "endpoints" contains ONLY identifiers. Do NOT reproduce API parameters, request \
bodies or response schemas; the program fills them in from the API specification.
  Use the exact "method" and "path" strings listed in API INFORMATION.
  An endpoint that is not listed there will be discarded.
- "files_to_modify[].path" MUST be a file that appears in PROJECT STRUCTURE \
(python_files / manifest_files) or in REPOSITORY EVIDENCE. Otherwise it is discarded.
- "files_to_create[].path" MUST NOT already exist in the repository.
- Test files belong ONLY in "testing_strategy". Do NOT also list them in \
"files_to_create" — the same path in both places produces two conflicting versions \
of that file, which breaks test collection.
- If you declare "error_handling", include a "files_to_create" entry of kind \
"exceptions" that defines the exception classes. The generated client and tests \
import them from that module; do not assume they live inside the client file.
- If any endpoint above lists a "request_body_schema" or "response_schemas", the \
generated client validates payloads into those model classes, so include a \
"files_to_create" entry of kind "models" that defines them. The schemas are named \
in API INFORMATION; their internal structure is not provided.
- "authentication" may be null only if the API declares no security scheme.
- "warnings" lists everything you are uncertain about. An empty list is acceptable.
- All paths are POSIX-style and relative to the project root. Never absolute, \
never containing "..".

Generate the IntegrationPlan now.\
"""

# 输出骨架：与 IntegrationPlan 的字段一一对应（有测试守护，防止与实际模型漂移）
OUTPUT_SKELETON: dict = {
    "integration_goal": "<one sentence: what is being integrated into what, and why>",
    "target_api": {
        "name": "<API name>",
        "version": "<API version>",
        "base_url": "<base url or null>",
        "total_endpoints": 0,
    },
    "repository": {
        "root": "<project root>",
        "name": "<project name>",
        "source_dirs": ["<source dir>"],
        "test_dirs": ["<test dir>"],
        "http_client": "<existing http client library, or null>",
        "http_client_module": "<file that already uses it, or null>",
        "existing_modules": ["<existing module to reuse>"],
        "inspected_files": ["<file you actually based a decision on>"],
    },
    "integration_strategy": {
        "approach": "new_client_module | extend_existing_client",
        "summary": "<how the integration is done>",
        "client_module": "<path of the client module>",
        "http_client": "<http client library to use>",
    },
    "files_to_create": [
        {
            "path": "<relative path>",
            "purpose": "<why this file must exist>",
            "kind": "client | models | exceptions | config | package_init | other",
        }
    ],
    "files_to_modify": [
        {
            "path": "<existing relative path>",
            "reason": "<why it must change; cite the evidence file>",
            "changes": ["<concrete change>"],
        }
    ],
    "dependencies": [
        {
            "name": "<package name>",
            "version": "<constraint or null>",
            "reason": "<why it is needed>",
            "already_installed": False,
        }
    ],
    "endpoints": [
        {"method": "<GET|POST|...>", "path": "<exact path from the API spec>", "purpose": "<...>"}
    ],
    "authentication": {
        "scheme": "<apiKey | http-bearer | oauth2 | ...>",
        "strategy": "<how credentials are attached to requests>",
        "config_location": "<where credentials are read from>",
        "required_env_vars": ["<ENV_VAR_NAME>"],
        "header_name": "<header name or null>",
    },
    "error_handling": {
        "exception_type": "<custom exception base class name>",
        "rules": [
            {
                "status_code": "<400|401|403|404|429|5xx>",
                "category": (
                    "validation | authentication | authorization | not_found | rate_limit "
                    "| server_error"
                ),
                "strategy": "<how to handle it>",
                "retryable": False,
                "documented_in_spec": False,
            }
        ],
        "notes": ["<note>"],
    },
    "testing_strategy": {
        "unit_tests": [{"path": "<path>", "purpose": "<...>", "covers": ["<...>"]}],
        "integration_tests": [{"path": "<path>", "purpose": "<...>", "covers": ["<...>"]}],
        "contract_tests": [{"path": "<path>", "purpose": "<...>", "covers": ["<...>"]}],
    },
    "assumptions": ["<assumption you had to make>"],
    "risks": [{"description": "<risk>", "severity": "low | medium | high", "mitigation": "<...>"}],
    "warnings": ["<uncertainty or missing information>"],
}


# ---------------------------------------------------------------- 渲染


def _dump(data: object) -> str:
    """紧凑 JSON（保留非 ASCII，便于中文项目名/描述可读）。"""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _truncate(text: str, limit: int, note: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: {note}]"


def _ref_name(ref: str | None) -> str | None:
    """``#/components/schemas/Pet`` → ``Pet``。"""
    if not ref:
        return None
    return ref.rsplit("/", 1)[-1] or None


def _response_schema_name(response: APIResponse) -> str | None:
    """响应会被校验成的模型类名（直接 schema_ref，或数组 items.$ref）。"""
    name = _ref_name(response.schema_ref)
    if name:
        return name
    items = (response.json_schema or {}).get("items")
    if isinstance(items, dict):
        return _ref_name(items.get("$ref"))
    return None


def render_api_info(api: APIInfo, *, max_chars: int = DEFAULT_MAX_API_CHARS) -> str:
    """渲染 API 信息：端点只保留标识与用途，不展开参数。

    **但 schema 名必须给**：Code Generator 会把响应体校验成模型类
    （``Pet.model_validate(response.json())``），并要求模型类有独立的 kind="models"
    文件。若 prompt 隐去 schema，模型会判断"没有 schema，不需要模型模块"，而生成器
    照旧生成模型引用——两者的决定互相矛盾。这里只给名字，不给结构，成本极低。
    """
    payload = {
        "name": api.name,
        "version": api.version,
        "base_url": api.base_url,
        "description": api.description,
        "authentication": api.auth.model_dump() if api.auth else None,
        "total_endpoints": len(api.endpoints),
        "endpoints": [
            {
                "method": endpoint.method,
                "path": endpoint.path,
                "operation_id": endpoint.operation_id,
                "summary": endpoint.summary,
                "tags": endpoint.tags,
                "request_body_schema": _ref_name(
                    endpoint.request_body.schema_ref if endpoint.request_body else None
                ),
                "response_schemas": {
                    item.status_code: name
                    for item in endpoint.responses
                    if (name := _response_schema_name(item))
                },
                "response_status_codes": [item.status_code for item in endpoint.responses],
            }
            for endpoint in api.endpoints
        ],
    }
    return _truncate(_dump(payload), max_chars, "API 端点过多，已截断")


def render_project_structure(
    project: ProjectStructure, *, max_chars: int = DEFAULT_MAX_STRUCTURE_CHARS
) -> str:
    """渲染仓库结构：只给事实清单，不给文件内容。"""
    payload = {
        "root": project.root,
        "name": project.name,
        "is_python_project": project.is_python_project,
        "manifest_files": project.manifest_files,
        "source_dirs": project.source_dirs,
        "test_dirs": project.test_dirs,
        "dependencies": project.dependencies,
        "python_files": project.python_files,
        "python_file_count": len(project.python_files),
    }
    return _truncate(_dump(payload), max_chars, "python_files 过多，已截断")


def render_project_constraints(project: ProjectStructure, state: PlannerState) -> str:
    """把"不可违反"的硬事实写成自然语言约束，避免模型自行发挥。"""
    lines = [
        f"- The repository root is {project.root!r}; all paths in the plan must be "
        f"relative to it, POSIX-style.",
        f"- The project declares {len(project.python_files)} Python file(s) and "
        f"{len(project.dependencies)} dependency/ies.",
    ]
    if project.manifest_files:
        lines.append(
            f"- Dependency changes may only be written to one of: "
            f"{', '.join(project.manifest_files)}."
        )
    else:
        lines.append(
            "- No dependency manifest (pyproject.toml / requirements.txt) was found; "
            "do not assume one exists."
        )
    if project.test_dirs:
        lines.append(f"- Tests belong under: {', '.join(project.test_dirs)}.")
    else:
        lines.append("- No test directory was detected; you must create one if tests are needed.")
    lines.append(
        "- You may only reference files that appear in PROJECT STRUCTURE or "
        "REPOSITORY EVIDENCE. Anything else will be discarded."
    )
    lines.append("- Credentials are read from environment variables only; never hardcode them.")
    if not state.request.strip():
        lines.append(
            "- No integration request was provided: plan for the API as a whole, "
            "prioritising its most representative endpoints."
        )
    return "\n".join(lines)


def _priority_of(path: str) -> int:
    """按路径关键词把证据文件分档；序号越小越优先保留。"""
    lowered = path.lower()
    for index, (_, keywords) in enumerate(EVIDENCE_PRIORITY):
        if any(keyword in lowered for keyword in keywords):
            return index
    return UNCLASSIFIED_PRIORITY


def build_evidence_payload(
    state: PlannerState,
    *,
    max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
) -> tuple[list[dict], list[str]]:
    """把证据整理成"按优先级裁剪后"的条目列表，返回 (条目, 裁剪说明)。

    Planner 不能自己扫仓库，只能消费 Repository Understanding 产出的证据；
    因此这里只做**筛选与截断**，不产生新证据。
    """
    candidates: list[tuple[int, int, dict]] = []
    order = 0
    for evidence in state.evidence:
        for match in evidence.matches:
            snippet = match.snippet
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars] + "\n... [snippet truncated]"
            candidates.append(
                (
                    _priority_of(match.path),
                    order,
                    {
                        "query": evidence.query,
                        "file": match.path,
                        "line": match.line_number,
                        "snippet": snippet,
                    },
                )
            )
            order += 1
    # 稳定排序：先按优先级，再按原始出现顺序
    candidates.sort(key=lambda item: (item[0], item[1]))

    kept: list[dict] = []
    notes: list[str] = []
    used = 0
    dropped_priority = False
    dropped_size = False
    for _, _, item in candidates:
        if len(kept) >= max_items:
            dropped_size = True
            break
        cost = len(item["snippet"]) + len(item["file"]) + 32
        if used + cost > max_chars:
            dropped_priority = True
            continue
        kept.append(item)
        used += cost

    total = len(candidates)
    if dropped_size:
        notes.append(
            f"证据条目数超过上限，仅保留优先级最高的 {len(kept)}/{total} 条"
            "（已丢弃低优先级的检索片段）"
        )
    if dropped_priority:
        notes.append(
            f"证据体积超过 {max_chars} 字符上限，已丢弃部分低优先级片段"
            f"（保留 {len(kept)}/{total} 条）"
        )
    if not kept and total:
        notes.append("证据全部超出体积上限，本次规划没有可用的代码证据")
    return kept, notes


def build_user_prompt(
    state: PlannerState,
    *,
    max_request_chars: int = DEFAULT_MAX_REQUEST_CHARS,
    max_api_chars: int = DEFAULT_MAX_API_CHARS,
    max_structure_chars: int = DEFAULT_MAX_STRUCTURE_CHARS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> tuple[str, list[str]]:
    """渲染 User Prompt，返回 (prompt, 裁剪说明)。"""
    evidence, notes = build_evidence_payload(state, max_chars=max_evidence_chars)
    request = state.request.strip() or "(none provided)"
    if len(request) > max_request_chars:
        request = _truncate(request, max_request_chars, "集成需求过长，已截断")
        notes.append(f"集成需求超过 {max_request_chars} 字符，已截断")
    prompt = USER_PROMPT_TEMPLATE.format(
        integration_request=request,
        api_info_json=render_api_info(state.api, max_chars=max_api_chars),
        project_structure_json=render_project_structure(
            state.project, max_chars=max_structure_chars
        ),
        project_constraints=render_project_constraints(state.project, state),
        evidence_json=_dump(evidence) if evidence else "[] (no repository evidence available)",
        output_skeleton=_dump(OUTPUT_SKELETON),
    )
    return prompt, notes


# 逐级降级预算：(request, evidence, api, structure)
# 先压体积大头（证据 / API），再压需求原文——OUTPUT FORMAT 与 PLANNING RULES
# 永远放在最后才动，因为它们是模型产出合法 JSON 的前提。
PROMPT_BUDGET_LADDER: tuple[tuple[int, int, int, int], ...] = (
    (4_000, 6_000, 4_000, 2_000),
    (2_000, 3_000, 3_000, 1_500),
    (1_000, 1_500, 2_000, 1_000),
    (600, 800, 1_200, 600),
    (400, 400, 800, 400),
)


def build_planner_prompt(state: PlannerState, *, max_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> str:
    """组装完整 prompt（System + User）。

    若总体积超上限，按预算阶梯逐级压缩（证据 → API → 结构 → 需求原文），
    保证最终一定落在预算内——永远不可能把无限大的上下文发给模型。
    阶梯仍不够时硬截断，截断标记本身也计入配额。
    """
    for request_budget, evidence_budget, api_budget, structure_budget in (
        (
            DEFAULT_MAX_REQUEST_CHARS,
            DEFAULT_MAX_EVIDENCE_CHARS,
            DEFAULT_MAX_API_CHARS,
            DEFAULT_MAX_STRUCTURE_CHARS,
        ),
        *PROMPT_BUDGET_LADDER,
    ):
        user_prompt, _ = build_user_prompt(
            state,
            max_request_chars=request_budget,
            max_api_chars=api_budget,
            max_structure_chars=structure_budget,
            max_evidence_chars=evidence_budget,
        )
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        if len(prompt) <= max_chars:
            return prompt

    note = "\n... [prompt truncated to fit the context budget]"
    return prompt[: max_chars - len(note)] + note
