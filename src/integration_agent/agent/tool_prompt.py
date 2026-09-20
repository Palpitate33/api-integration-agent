"""ToolUsingPlanner 的 Prompt Builder：把 PlannerState 渲染成 Agent 的初始上下文。

与 prompt.py（DeepSeekPlanner 的单轮 prompt）的关系：
    - 复用其通用部分：render_api_info / render_project_structure /
      render_project_constraints / build_evidence_payload / OUTPUT_SKELETON /
      PROMPT_BUDGET_LADDER 全部直接 import，不复制一份模板出来。
    - 差异只有一处，也是本模块存在的理由：这里描述的是**一个可以自己决定还要
      看什么的 Agent**，而不是"把材料一次性摆好、你只回答"。因此 prompt 里要
      给出工具清单与调用纪律，并明确"证据不足时继续调用工具，而不是猜"。

设计约束：
    - 本模块**不读取文件系统**、不扫描仓库、不调用工具、不发网络请求。
      它只把 PlannerState 与 registry.describe() 已经准备好的文本拼起来。
    - 初始上下文只放 user request + API 基础信息 + 项目结构摘要 + 已有 evidence；
      需要更多信息时由模型自己决定调哪个工具——这是 Agent Planner 与确定性
      Planner 的关键差别，提前把所有工具跑一遍等于把它退化回确定性 Planner。
    - 上下文总体积有硬上限，超限时按预算阶梯逐级压缩，绝不把无限大的上下文发出去。

输出包络：
    模型只被要求输出一种形状——``{"final": {...IntegrationPlan 字段...}}``。
    包络让"计划正文"与"任何解释性文字"在结构上分得开，解析侧只需认一个键。
"""

import json

from integration_agent.agent.prompt import (
    DEFAULT_MAX_API_CHARS,
    DEFAULT_MAX_EVIDENCE_CHARS,
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_REQUEST_CHARS,
    DEFAULT_MAX_STRUCTURE_CHARS,
    OUTPUT_SKELETON,
    PROMPT_BUDGET_LADDER,
    build_evidence_payload,
    render_api_info,
    render_project_constraints,
    render_project_structure,
)
from integration_agent.agent.state import PlannerState

# ---------------------------------------------------------------- System Prompt

TOOL_SYSTEM_PROMPT = """\
You are APIForge Planner, an expert software integration planning agent.

Your task is to analyze a third-party API specification together with an existing \
software repository and produce a safe, evidence-based IntegrationPlan.

You are NOT a code generator, and you are NOT an executor.

YOU HAVE TOOLS
You can request read-only tools to inspect the target repository and the already-parsed \
API specification. Use them whenever the information you were given is not enough to \
make a decision.

Every tool is READ ONLY. You cannot write or delete files, run commands or shell, \
install dependencies, make network requests, read environment variables, or read \
credentials. Requesting such a thing is impossible — do not pretend otherwise.

HOW TO WORK

1. Read the integration request, the API information and the project structure you \
were given.
2. Decide whether that is enough to plan confidently. If it is not, call tools to \
gather the missing evidence BEFORE answering.
3. Evidence has priority over assumptions. Never assume that a file, a module, a \
function, a dependency or an API endpoint exists — verify it with a tool, or leave \
it out of the plan.
4. Do not call a tool to confirm something you were already told. Do not repeat a \
call you have already made: an identical call is answered identically and will be \
refused.
5. When you have enough evidence, STOP calling tools and answer with the final JSON.

RULES

- Files you may modify must already exist; files you create must not already exist.
- Endpoints must come from the API information you were given. An endpoint that is \
not in the specification will be discarded.
- Credentials are read from environment variables only. Never invent, request, or \
hardcode a secret.
- Prefer reusing existing abstractions over introducing new ones.
- Prefer minimal changes over architectural changes.
- Do not redesign the existing project and do not touch unrelated files.
- If something remains uncertain after gathering evidence, say so in "warnings" \
instead of inventing an answer.

Your final message MUST be a single JSON object and nothing else — no Markdown, \
no code fences, no prose outside the JSON."""

# ---------------------------------------------------------------- User Prompt

TOOL_USER_PROMPT_TEMPLATE = """\
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

1. Only use files and symbols supported by the API information, the project \
structure, the repository evidence, or a tool result you obtained in this session.
2. Reuse existing HTTP clients when available.
3. Reuse existing authentication mechanisms when available.
4. Follow the project's existing testing conventions.
5. Minimize the number of changed files. Do not modify unrelated code.
6. Do not invent dependencies unless the API genuinely requires them.
7. Never hardcode credentials or API keys.
8. If information is missing, record the uncertainty in "warnings".
9. Every selected file must have a concrete reason.
10. Call tools first if you need more evidence; answer only when you are ready.

========================
OUTPUT FORMAT
========================

When you are done gathering evidence, reply with a single JSON object shaped exactly \
like this skeleton, wrapped in a "final" key (respond in json only — no prose, no code \
fences). Keys are required; replace every placeholder with your decision:

{output_skeleton}

Additional rules for this output shape:

- Everything must be inside the "final" object. Nothing else at the top level.
- "endpoints" contains ONLY identifiers. Do NOT reproduce API parameters, request \
bodies or response schemas; the program fills them in from the API specification.
  Use the exact "method" and "path" strings listed in API INFORMATION.
  An endpoint that is not listed there will be discarded.
- "files_to_modify[].path" MUST be a file that appears in PROJECT STRUCTURE \
(python_files / manifest_files), in REPOSITORY EVIDENCE, or in a tool result. \
Otherwise it is discarded.
- "files_to_create[].path" MUST NOT already exist in the repository.
- Test files belong ONLY in "testing_strategy". Do NOT also list them in \
"files_to_create" — the same path in both places produces two conflicting versions \
of that file, which breaks test collection.
- If you declare "error_handling", include a "files_to_create" entry of kind \
"exceptions" that defines the exception classes.
- If any endpoint above lists a "request_body_schema" or "response_schemas", include \
a "files_to_create" entry of kind "models" that defines them.
- "authentication" may be null only if the API declares no security scheme.
- All paths are POSIX-style and relative to the project root. Never absolute, \
never containing "..".

Generate the IntegrationPlan now."""


def build_tool_system_prompt(tools_description: str) -> str:
    """把通用指令与当前注册表的工具清单拼成 System Prompt。

    工具清单来自 ToolRegistry.describe()（按名称排序、schema 键序固定），
    同一个注册表渲染两次逐字节一致——"同样输入产生同样 prompt"是回放与
    缓存的前提。
    """
    tools = tools_description.strip() or "(no tools are registered)"
    return (
        f"{TOOL_SYSTEM_PROMPT}\n\n"
        "========================\n"
        "AVAILABLE TOOLS\n"
        "========================\n\n"
        f"{tools}\n"
    )


def _dump(data: object) -> str:
    """紧凑 JSON（保留非 ASCII，便于中文项目名/描述可读）。"""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _truncate(text: str, limit: int, note: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: {note}]"


def build_tool_user_prompt(
    state: PlannerState,
    *,
    max_request_chars: int = DEFAULT_MAX_REQUEST_CHARS,
    max_api_chars: int = DEFAULT_MAX_API_CHARS,
    max_structure_chars: int = DEFAULT_MAX_STRUCTURE_CHARS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> str:
    """渲染 User Prompt：初始上下文 = 需求 + API + 项目结构 + 已有证据。

    这里**不执行任何工具**：要补什么证据由模型在回路上自己决定。
    """
    evidence, _ = build_evidence_payload(state, max_chars=max_evidence_chars)
    request = state.request.strip() or "(none provided)"
    if len(request) > max_request_chars:
        request = _truncate(request, max_request_chars, "集成需求过长，已截断")
    return TOOL_USER_PROMPT_TEMPLATE.format(
        integration_request=request,
        api_info_json=render_api_info(state.api, max_chars=max_api_chars),
        project_structure_json=render_project_structure(
            state.project, max_chars=max_structure_chars
        ),
        project_constraints=render_project_constraints(state.project, state),
        evidence_json=_dump(evidence) if evidence else "[] (no repository evidence available)",
        output_skeleton=_dump({"final": OUTPUT_SKELETON}),
    )


def build_tool_planner_prompts(
    state: PlannerState,
    *,
    tools_description: str,
    max_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> tuple[str, str]:
    """组装 (system_prompt, user_prompt)。

    总体积超上限时按与 DeepSeekPlanner 相同的预算阶梯逐级压缩（证据 → API →
    结构 → 需求原文），仍不够则硬截断 user prompt——system prompt 里的工具清单与
    输出格式是模型能正常工作的前提，永远不动它。
    """
    system_prompt = build_tool_system_prompt(tools_description)
    budgets = (
        (
            DEFAULT_MAX_REQUEST_CHARS,
            DEFAULT_MAX_EVIDENCE_CHARS,
            DEFAULT_MAX_API_CHARS,
            DEFAULT_MAX_STRUCTURE_CHARS,
        ),
        *PROMPT_BUDGET_LADDER,
    )
    for request_budget, evidence_budget, api_budget, structure_budget in budgets:
        user_prompt = build_tool_user_prompt(
            state,
            max_request_chars=request_budget,
            max_api_chars=api_budget,
            max_structure_chars=structure_budget,
            max_evidence_chars=evidence_budget,
        )
        if len(system_prompt) + len(user_prompt) <= max_chars:
            return system_prompt, user_prompt

    note = "\n... [prompt truncated to fit the context budget]"
    return system_prompt, user_prompt[: max_chars - len(system_prompt) - len(note)] + note


__all__ = [
    "TOOL_SYSTEM_PROMPT",
    "TOOL_USER_PROMPT_TEMPLATE",
    "build_tool_planner_prompts",
    "build_tool_system_prompt",
    "build_tool_user_prompt",
]
