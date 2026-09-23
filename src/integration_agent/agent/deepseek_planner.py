"""DeepSeek Planner：让 LLM 决定"怎么集成"，由程序保证"事实与约束"。

分工原则
--------
LLM 负责**判断**：集成目标、策略取舍、要新建/修改哪些文件、依赖、认证方式、
错误处理、测试策略、风险与不确定性。

程序负责**约束**：
    - 事实字段（API 名称/版本/base_url/端点总数、仓库 root/name/source_dirs/test_dirs）
      由 PlannerState 回填，LLM 无权改写——杜绝"发明项目结构"。
    - 端点由 method+path 在真实 APIInfo 中解析，未知端点直接丢弃。
    - 路径必须落在真实仓库文件集合内（新建文件必须不存在），越界/绝对路径直接丢弃。
    - 凭据只允许来自环境变量，疑似硬编码密钥会被丢弃并记录 warning。
    所有丢弃都写进 IntegrationPlan.warnings —— **不静默吞掉任何东西**。

安全边界（本模块不做的事）
--------------------------
    - 不写文件、不执行 shell / git / 安装依赖、不改环境变量、不发网络请求。
    - 不生成最终代码（那是 generation 阶段的职责）。
    - 不读取 API Key：Key 的读取完全封装在注入的 LLMClient 内部，本模块只调用
      generate(prompt)，也**不检查环境变量**。
    - 不自行全仓库扫描：只在调用方未提供证据时，通过 repository.code_search
      发起少量**受限**定向检索（有 max_results / max_total_chars 上限）。

LLMClient 从哪来
----------------
``LLMClient`` 是 provider-neutral 的最小契约（``generate(prompt) -> str``），
归属 ``integration_agent.llm`` —— 既不属 repair，也不属 agent，因此这里是
一个普通的运行时 import，不再需要绕开 import cycle 的 TYPE_CHECKING 变通：
``agent → llm`` 是单向的，``llm`` 不依赖任何一个 domain 包。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from integration_agent.agent.models import (
    AuthenticationPlan,
    FileModification,
    IntegrationPlan,
    PlannedEndpoint,
    PlannedFile,
    RepositoryContext,
    TargetAPI,
)
from integration_agent.agent.planner import DEFAULT_MAX_ENDPOINTS, KNOWN_HTTP_CLIENTS, STOPWORDS
from integration_agent.agent.prompt import DEFAULT_MAX_PROMPT_CHARS, build_planner_prompt
from integration_agent.agent.state import PlannerState
from integration_agent.api import APIEndpoint, APIInfo
from integration_agent.llm import LLMClient
from integration_agent.repository import search_code
from integration_agent.trace import emit

# ------------------------------------------------------------------ 常量

# 补齐证据时的检索上限：宁可少检索，也不把仓库倒给模型
MAX_EVIDENCE_QUERIES = 6
MAX_EVIDENCE_MATCHES = 3
MAX_EVIDENCE_SNIPPET_CHARS = 1200
# 定向检索全部落空时的兜底查询：任何 Python 模块都至少有个 def，用它换取
# "看到真实代码风格"的最低限度上下文
FALLBACK_EVIDENCE_QUERY = "def "

# 疑似凭据：命中即视为"模型试图把密钥写进计划"，一律丢弃并告警
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"\b(?:ghp|gho|ghs|ghr)_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.]{16,}"),
)

# 合法的环境变量名：必须是大写字母开头的 UPPER_SNAKE_CASE
ENV_VAR_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

# 绝对路径 / 盘符相对路径（C:\ 与 C:foo 都拒绝）
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:")

# ```json ... ``` 代码围栏（只在最外层做有限清理）
FENCE_PATTERN = re.compile(r"^\s*```(?:json|JSON)?\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)

# 错误信息脱敏：抹掉疑似凭据，并限制长度（绝不回显完整 prompt / 请求体）
CREDENTIAL_IN_TEXT_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{4,}|Bearer\s+\S+)")
MAX_ERROR_MESSAGE_CHARS = 300


# ------------------------------------------------------------------ 异常


class DeepSeekPlannerError(Exception):
    """DeepSeek Planner 错误基类。"""


class DeepSeekPlannerParseError(DeepSeekPlannerError):
    """LLM 响应无法解析为 JSON 对象。"""


class DeepSeekPlannerValidationError(DeepSeekPlannerError):
    """解析成功但不满足 IntegrationPlan 的模型约束。"""


# ------------------------------------------------------------------ Planner


class DeepSeekPlanner:
    """用 LLM 生成 IntegrationPlan，实现 IntegrationPlanner Protocol。

    失败时**直接抛异常**，不做任何静默 fallback：调用方（Pipeline）会把异常转成
    ``status="error" / failed_stage="plan"`` 并在结果中如实呈现。是否退回到
    DeterministicPlanner 由调用方决定，本类不替它做主。
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_endpoints: int = DEFAULT_MAX_ENDPOINTS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        if not callable(getattr(llm, "generate", None)):
            raise DeepSeekPlannerError(
                "llm 必须实现 generate(prompt: str) -> str（LLMClient 协议）"
            )
        if max_endpoints < 1:
            raise DeepSeekPlannerError("max_endpoints 必须 >= 1")
        if max_prompt_chars < 1000:
            raise DeepSeekPlannerError("max_prompt_chars 过小，至少 1000")
        self.llm = llm
        self.max_endpoints = max_endpoints
        self.max_prompt_chars = max_prompt_chars

    def plan(self, state: PlannerState) -> IntegrationPlan:
        """生成集成方案。整个过程只读、可重放（同一输入 + 同一 LLM 输出 = 同一计划）。"""
        started = time.perf_counter()
        emit(
            "planner",
            "planning_started",
            "开始规划集成方案",
            metadata={"planner": "llm"},
        )
        warnings: list[str] = []

        # 1. 证据准备（调用方已提供则原样使用，不重复检索）
        _collect_evidence(state)

        # 2. 渲染 prompt 并调用 LLM（唯一一次外部调用）
        prompt = build_planner_prompt(state, max_chars=self.max_prompt_chars)
        # 只记 prompt 规模：它装着仓库代码片段与 OpenAPI 节选，不放进 trace。
        emit("planner", "llm_called", "调用 LLM 生成计划", metadata={"prompt_chars": len(prompt)})
        llm_started = time.perf_counter()
        raw = self._call_llm(prompt)
        emit(
            "planner",
            "llm_completed",
            "LLM 返回计划文本",
            metadata={"response_chars": len(raw)},
            duration=time.perf_counter() - llm_started,
        )

        # 3. 解析 → 约束 → 校验
        data = parse_plan_json(raw)
        endpoints = resolve_endpoints(data, state, warnings, max_endpoints=self.max_endpoints)
        target_api, repository, grounding_warnings = ground_facts(data, state)
        warnings.extend(grounding_warnings)

        data["target_api"] = target_api.model_dump()
        data["repository"] = repository.model_dump()
        data["endpoints"] = endpoints

        plan = validate_plan(data)

        # 4. 约束层：把 LLM 的"判断"逐条对照证据，不合规的丢弃并记录
        plan, constraint_warnings = enforce_constraints(plan, state)
        warnings.extend(constraint_warnings)

        # LLM 自己声明的不确定性排在前面，程序发现的约束问题排在后面
        plan = plan.model_copy(update={"warnings": [*plan.warnings, *warnings]})
        emit(
            "planner",
            "planning_completed",
            "集成方案已生成",
            metadata={
                "planner": "llm",
                "endpoints": len(plan.endpoints),
                "files_to_create": len(plan.files_to_create),
                "files_to_modify": len(plan.files_to_modify),
                "dependencies": len(plan.dependencies),
                "warnings": len(plan.warnings),
            },
            duration=time.perf_counter() - started,
            status="completed",
        )
        return plan

    # -------------------------------------------------------------- 内部

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM；任何异常都转成 DeepSeekPlannerError，且信息已脱敏。

        这里刻意不把 prompt 内容放进异常——完整 prompt 可能包含仓库代码片段，
        不是错误信息该携带的东西。

        用 ``from None`` 而不是 ``from exc``：Pipeline 会把
        ``traceback.format_exc()`` 原样写进 PipelineResult 并返回给前端，一旦保留
        异常链，底层那条**未脱敏**的原始异常会随 traceback 一起泄漏出去，
        让这里的 sanitize 形同虚设。脱敏后的消息已包含异常类型与文本，
        排障信息并没有丢。
        """
        try:
            return self.llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - 统一包装为 Planner 错误
            raise DeepSeekPlannerError(
                f"调用 LLM 失败（{type(exc).__name__}）：{sanitize(str(exc))}"
            ) from None


# ------------------------------------------------------------------ 解析


def sanitize(text: str, *, limit: int = MAX_ERROR_MESSAGE_CHARS) -> str:
    """错误信息脱敏：抹掉疑似凭据 + 压平换行 + 截断长度。"""
    flattened = " ".join(text.split())
    return CREDENTIAL_IN_TEXT_PATTERN.sub("***", flattened)[:limit]


def parse_plan_json(raw: str) -> dict[str, Any]:
    """把 LLM 原始响应解析为 JSON 对象。

    只做**有限**清理：允许 ```json 围栏，允许在 JSON 前后带少量解释文字。
    绝不跳过 json.loads——任何情况下都必须先解析成 dict 才可能进入 Pydantic。
    """
    if not raw or not raw.strip():
        raise DeepSeekPlannerParseError("DeepSeek 返回空响应，无法解析 IntegrationPlan")

    text = raw.strip()
    fenced = FENCE_PATTERN.match(text)
    if fenced:
        text = fenced.group("body").strip()

    if not text.startswith("{"):
        # 模型偶尔会加一句前言/结语：取最外层花括号之间的内容再试一次
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise DeepSeekPlannerParseError("DeepSeek 返回的内容中找不到 JSON 对象")
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeepSeekPlannerParseError(f"DeepSeek 返回的内容不是合法 JSON：{exc}") from None

    if not isinstance(data, dict):
        raise DeepSeekPlannerParseError(
            f"DeepSeek 返回的 JSON 顶层必须是对象，实际是 {type(data).__name__}"
        )
    return data


def validate_plan(data: dict[str, Any]) -> IntegrationPlan:
    """经过 Pydantic 校验，把 dict 变成 IntegrationPlan。

    错误信息只保留字段路径与原因，不回显原始输入值（可能含模型编造的敏感内容）。

    同样使用 ``from None``：Pydantic 的 ValidationError 在 traceback 里会完整打印
    ``input_value=...``，一旦保留异常链，模型给的原始输入会绕过这里的摘要直接泄漏。
    """
    try:
        return IntegrationPlan.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError
        details = _validation_summary(exc)
        raise DeepSeekPlannerValidationError(
            f"DeepSeek 返回的 JSON 不满足 IntegrationPlan 模型约束：{details}"
        ) from None


def _validation_summary(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return sanitize(str(exc))
    try:
        items = errors()
    except Exception:  # noqa: BLE001 - 不因诊断信息本身再次失败
        return sanitize(str(exc))
    parts: list[str] = []
    for item in items[:5]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        parts.append(f"{location}: {item.get('msg', 'invalid')}")
    remaining = len(items) - len(parts)
    summary = "; ".join(parts)
    if remaining > 0:
        summary += f"; 另有 {remaining} 处错误"
    return summary


# ------------------------------------------------------------------ 路径安全


def validate_relative_path(path: str) -> str | None:
    """校验路径是否为安全的仓库内相对路径；返回错误说明，None 表示安全。

    与 repair.llm_repair_applier.validate_path 同样的语义，这里独立实现是为了
    保持 ``agent`` 不依赖 ``repair`` 的依赖方向（见模块 docstring）。
    """
    if not path or not path.strip():
        return "path 为空"
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        return "path 不能是绝对路径"
    if WINDOWS_ABSOLUTE_PATTERN.match(normalized):
        return "path 不能是 Windows 绝对路径或盘符相对路径"
    if any(part == ".." for part in normalized.split("/") if part not in ("", ".")):
        return "path 包含越界片段 '..'"
    return None


def _normalize_path(path: str) -> str:
    """统一成 POSIX 相对路径，便于与仓库文件集合比较。"""
    normalized = str(path).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def known_repository_files(state: PlannerState) -> set[str]:
    """本次规划中"确实存在"的仓库文件集合（扫描结果 + 检索证据）。"""
    known = {_normalize_path(path) for path in state.project.python_files}
    known |= {_normalize_path(path) for path in state.project.manifest_files}
    for evidence in state.evidence:
        known |= {_normalize_path(match.path) for match in evidence.matches}
    known.discard("")
    return known


# ------------------------------------------------------------------ 端点解析


def resolve_endpoints(
    data: dict[str, Any],
    state: PlannerState,
    warnings: list[str],
    *,
    max_endpoints: int = DEFAULT_MAX_ENDPOINTS,
) -> list[PlannedEndpoint]:
    """把 LLM 给出的端点标识解析为真实的 APIEndpoint 对象。

    LLM 只需回答"用哪些端点"，参数 / 请求体 / 响应 schema 一律从 APIInfo 取，
    从结构上消除"模型凭记忆编造 schema"的可能。
    """
    raw_endpoints = data.get("endpoints")
    if raw_endpoints is None:
        warnings.append("DeepSeek 未给出 endpoints，本次计划不包含任何端点")
        return []
    if not isinstance(raw_endpoints, list):
        warnings.append(f"endpoints 字段应为数组，实际是 {type(raw_endpoints).__name__}，已忽略")
        return []

    index: dict[tuple[str, str], APIEndpoint] = {
        (endpoint.method.upper(), _normalize_path(endpoint.path)): endpoint
        for endpoint in state.api.endpoints
    }

    resolved: list[PlannedEndpoint] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_endpoints:
        if not isinstance(item, dict):
            warnings.append(f"endpoints 中存在非对象条目，已丢弃：{type(item).__name__}")
            continue
        method = str(item.get("method", "")).strip().upper()
        path = _normalize_path(str(item.get("path", "")))
        endpoint = index.get((method, path))
        if endpoint is None:
            warnings.append(
                f"DeepSeek 指定的端点不在 API 文档中，已丢弃：{method} {path or '<空>'}"
            )
            continue
        key = (method, _normalize_path(endpoint.path))
        if key in seen:
            continue  # 重复选择：去重，不算问题
        seen.add(key)
        purpose = str(item.get("purpose") or "").strip() or _default_purpose(endpoint)
        resolved.append(PlannedEndpoint(endpoint=endpoint, purpose=purpose))

    if len(resolved) > max_endpoints:
        warnings.append(
            f"DeepSeek 选择了 {len(resolved)} 个端点，超过上限 {max_endpoints}，"
            f"已按下限截断（保留前 {max_endpoints} 个）"
        )
        resolved = resolved[:max_endpoints]
    return resolved


def _default_purpose(endpoint: APIEndpoint) -> str:
    return endpoint.summary or endpoint.operation_id or f"{endpoint.method} {endpoint.path}"


# ------------------------------------------------------------------ 事实回填


def ground_facts(
    data: dict[str, Any], state: PlannerState
) -> tuple[TargetAPI, RepositoryContext, list[str]]:
    """用真实事实回填 target_api 与 repository，并审计 LLM 的判断字段。

    事实字段（名称、版本、base_url、端点总数、仓库根目录、源码/测试目录）直接
    取自 PlannerState —— LLM 写什么都无法改变它们；它写错了也不会被采纳，
    更不会把幻觉带进下游。

    LLM 的判断字段（http_client / http_client_module / existing_modules /
    inspected_files）会被逐条对照证据：路径不存在就丢弃并记 warning。
    """
    warnings: list[str] = []

    target_api = TargetAPI(
        name=state.api.name,
        version=state.api.version,
        base_url=state.api.base_url,
        total_endpoints=len(state.api.endpoints),
    )

    raw_repo = data.get("repository")
    if not isinstance(raw_repo, dict):
        if raw_repo is not None:
            warnings.append(
                f"repository 字段应为对象，实际是 {type(raw_repo).__name__}，已按仓库事实重建"
            )
        else:
            warnings.append("DeepSeek 未给出 repository 字段，已按仓库事实重建")
        raw_repo = {}

    known = known_repository_files(state)
    evidence_paths = {
        _normalize_path(match.path) for evidence in state.evidence for match in evidence.matches
    }

    # HTTP 客户端库名：不校验（可能是仓库确实没有的新库），由 audit_strategy 检查依赖声明
    http_client = raw_repo.get("http_client")
    if http_client is not None and not isinstance(http_client, str):
        warnings.append("repository.http_client 不是字符串，已置空")
        http_client = None

    # HTTP 客户端所在模块：路径字段，必须真实存在
    client_module = raw_repo.get("http_client_module")
    if client_module is not None:
        if not isinstance(client_module, str) or _normalize_path(client_module) not in known:
            warnings.append(
                f"DeepSeek 声称 HTTP 客户端位于 {client_module!r}，但该文件不在仓库中，已清除"
            )
            client_module = None
        else:
            client_module = _normalize_path(client_module)

    # 可复用模块：模块名应能在真实文件路径中找到（pkg.mod → pkg/mod）
    #
    # 两种写法都接受：模型既可能给点号模块名（demo_project.client），也可能给文件
    # 路径（demo_project/client.py）。只认前者会把后者的真实模块误报成"找不到"。
    existing_modules: list[str] = []
    raw_modules = raw_repo.get("existing_modules")
    if isinstance(raw_modules, list):
        for module in raw_modules:
            name = str(module).strip()
            if not name or name in existing_modules:
                continue
            needle = name.removesuffix(".py").replace(".", "/").lower()
            if any(needle in path.lower() for path in known):
                existing_modules.append(name)
            else:
                warnings.append(f"DeepSeek 声称仓库中可复用模块 {name!r}，但找不到对应文件，已丢弃")
    elif raw_modules is not None:
        warnings.append("repository.existing_modules 不是数组，已忽略")

    # 决策依据文件：必须是**真实存在**的仓库文件。
    #
    # 注意这里校验的是"文件存在"而不是"出现在检索命中里"：模型完全可以引用一个
    # 真实存在、但本轮关键词没命中的文件（例如 pyproject.toml）。把后者也判成幻觉
    # 会产出误导性告警——本项目就曾因此把真实文件误报为编造。
    inspected: list[str] = []
    raw_inspected = raw_repo.get("inspected_files")
    if isinstance(raw_inspected, list):
        for path in raw_inspected:
            normalized = _normalize_path(str(path))
            if not normalized:
                continue
            if normalized in known and normalized not in inspected:
                inspected.append(normalized)
            elif normalized not in known:
                warnings.append(f"DeepSeek 声称依据了 {normalized!r}，但该文件不在仓库中，已丢弃")
    elif raw_inspected is not None:
        warnings.append("repository.inspected_files 不是数组，已忽略")
    if not inspected:
        # 兜底：模型没给时，直接用真实检索证据，保证该字段反映实际发生过的事
        inspected = [path for path in sorted(evidence_paths)]

    repository = RepositoryContext(
        root=state.project.root,
        name=state.project.name,
        source_dirs=list(state.project.source_dirs),
        test_dirs=list(state.project.test_dirs),
        http_client=http_client,
        http_client_module=client_module,
        existing_modules=existing_modules,
        inspected_files=inspected,
    )
    return target_api, repository, warnings


# ------------------------------------------------------------------ 审计


def enforce_constraints(
    plan: IntegrationPlan, state: PlannerState
) -> tuple[IntegrationPlan, list[str]]:
    """约束层：把 LLM 的判断逐条对照证据，不合规的丢弃并记录 warning。

    所有修正都通过 ``model_copy`` 返回**新对象**，不修改传入的 plan——
    审计函数不应有副作用，否则"丢弃了哪几条"就无从复核。
    """
    warnings: list[str] = []
    files_to_create, files_to_modify, path_warnings = audit_paths(plan, state)
    warnings.extend(path_warnings)

    files_to_create, test_warnings = audit_test_files(plan, files_to_create)
    warnings.extend(test_warnings)

    files_to_create, exception_warnings = audit_exception_module(plan, files_to_create)
    warnings.extend(exception_warnings)

    files_to_create, model_warnings = audit_models_module(plan, files_to_create)
    warnings.extend(model_warnings)
    warnings.extend(audit_strategy(plan, state))

    credential_warnings, required_env_vars = audit_credentials(plan, state)
    warnings.extend(credential_warnings)

    update: dict[str, Any] = {
        "files_to_create": files_to_create,
        "files_to_modify": files_to_modify,
    }
    if required_env_vars is not None and plan.authentication is not None:
        update["authentication"] = AuthenticationPlan(
            **{
                **plan.authentication.model_dump(),
                "required_env_vars": required_env_vars,
            }
        )
    return plan.model_copy(update=update), warnings


def audit_paths(
    plan: IntegrationPlan, state: PlannerState
) -> tuple[list[PlannedFile], list[FileModification], list[str]]:
    """校验文件路径：越界 / 不存在 / 已存在，一律丢弃并记录 warning。

    返回 (files_to_create, files_to_modify, warnings)。
    """
    warnings: list[str] = []
    known = known_repository_files(state)

    files_to_create: list[PlannedFile] = []
    seen_create: set[str] = set()
    for item in plan.files_to_create:
        problem = validate_relative_path(item.path)
        if problem:
            warnings.append(f"files_to_create 中 {item.path!r} 不合法（{problem}），已丢弃")
            continue
        path = _normalize_path(item.path)
        if path in known:
            warnings.append(
                f"files_to_create 中的 {path!r} 已存在于仓库中，不能作为新建文件，已丢弃"
            )
            continue
        if path in seen_create:
            continue
        seen_create.add(path)
        files_to_create.append(item.model_copy(update={"path": path}))

    files_to_modify: list[FileModification] = []
    seen_modify: set[str] = set()
    for item in plan.files_to_modify:
        problem = validate_relative_path(item.path)
        if problem:
            warnings.append(f"files_to_modify 中 {item.path!r} 不合法（{problem}），已丢弃")
            continue
        path = _normalize_path(item.path)
        if path not in known:
            warnings.append(
                f"files_to_modify 中的 {path!r} 不在仓库文件列表中，无法修改不存在的文件，已丢弃"
            )
            continue
        if path in seen_modify:
            continue
        seen_modify.add(path)
        files_to_modify.append(item.model_copy(update={"path": path}))

    return files_to_create, files_to_modify, warnings


def audit_test_files(
    plan: IntegrationPlan, files_to_create: list[PlannedFile]
) -> tuple[list[PlannedFile], list[str]]:
    """测试文件的唯一出处是 testing_strategy。

    Code Generator 的分工是：files_to_create → 生成完整文件；testing_strategy →
    生成测试骨架。同一个路径两边都写，会让同一文件被生成两次、内容还不一样，
    直接导致 pytest collection error。所以这里强制去重，把测试文件从
    files_to_create 中摘掉（保留 testing_strategy 里的定义）。
    """
    strategy = plan.testing_strategy
    test_paths = {
        _normalize_path(spec.path)
        for spec in (*strategy.unit_tests, *strategy.integration_tests, *strategy.contract_tests)
    }
    if not test_paths:
        return files_to_create, []

    warnings: list[str] = []
    kept: list[PlannedFile] = []
    for item in files_to_create:
        normalized = _normalize_path(item.path)
        if normalized in test_paths:
            warnings.append(
                f"files_to_create 中的 {normalized!r} 同时出现在 testing_strategy 中，"
                "已从 files_to_create 移除（测试文件只由 testing_strategy 生成，"
                "否则同一文件会被生成两次并导致测试收集失败）"
            )
            continue
        kept.append(item)
    return kept, warnings


def _ensure_sibling_module(
    files_to_create: list[PlannedFile], *, kind: str, suffix: str, purpose: str
) -> tuple[list[PlannedFile], str | None]:
    """客户端旁边"必须有"的模块，缺了就补一个同目录的兄弟模块。

    返回 (files_to_create, 新增的路径)；已有该 kind、或没有客户端模块可挂靠时，
    路径为 None（不做无中生有的补充）。
    """
    if any(item.kind == kind for item in files_to_create):
        return files_to_create, None
    client = next((item for item in files_to_create if item.kind == "client"), None)
    if client is None:
        return files_to_create, None
    path = _sibling_module_path(client.path, suffix)
    if path is None:
        return files_to_create, None
    return [*files_to_create, PlannedFile(path=path, kind=kind, purpose=purpose)], path


def audit_exception_module(
    plan: IntegrationPlan, files_to_create: list[PlannedFile]
) -> tuple[list[PlannedFile], list[str]]:
    """异常类必须有独立的落点，否则生成器会写出 ``from None import ...``。

    Code Generator 的分工是：异常类只在 kind="exceptions" 的文件里**定义**，
    客户端与测试都从那里 import（客户端正文无条件 ``raise <exception_type>``）。
    因此只要计划里有客户端模块，就必须同时有异常模块。LLM 倾向于把异常类"顺带"
    写进客户端文件，而生成器并不按那个假设工作——缺了异常模块，生成的测试文件
    是语法错误，pytest 直接收集失败（实测：``from None import PetstoreRateLimitError``）。
    """
    exception_type = plan.error_handling.exception_type
    files, added = _ensure_sibling_module(
        files_to_create,
        kind="exceptions",
        suffix="exceptions",
        purpose=f"HTTP 状态码到异常类型的映射，基类 {exception_type}",
    )
    if added is None:
        return files, []
    return files, [
        f"计划声明了 error_handling（基类 {exception_type}）但没有 kind='exceptions' 的文件；"
        f"已补充 {added!r}（异常类必须在独立模块中定义，客户端与测试都从这里 import）"
    ]


def _references_models(endpoints: list[PlannedEndpoint]) -> bool:
    """端点是否会引用 schema 模型类（决定生成器是否强依赖 models 模块）。

    与 Code Generator 的 ``_referenced_models`` 同源：请求体 schema_ref、
    响应 schema_ref、数组响应的 items.$ref 三者任一存在即成立。
    """
    for planned in endpoints:
        endpoint = planned.endpoint
        if endpoint.request_body is not None and endpoint.request_body.schema_ref:
            return True
        for response in endpoint.responses:
            if response.schema_ref:
                return True
            items = (response.json_schema or {}).get("items")
            if isinstance(items, dict) and items.get("$ref"):
                return True
    return False


def audit_models_module(
    plan: IntegrationPlan, files_to_create: list[PlannedFile]
) -> tuple[list[PlannedFile], list[str]]:
    """端点引用了 schema 时，模型类必须有独立模块，否则客户端引用未定义的名字。

    客户端正文会生成 ``return Pet.model_validate(response.json())``，但导入语句只在
    存在 kind="models" 文件时才写。缺了模型模块，客户端能生成、能编译，直到被调用
    才 ``NameError: name 'Pet' is not defined``（实测：集成测试 11 failed）。
    """
    if not _references_models(plan.endpoints):
        return files_to_create, []
    files, added = _ensure_sibling_module(
        files_to_create,
        kind="models",
        suffix="models",
        purpose="请求/响应的模型类，与 OpenAPI schema 对应",
    )
    if added is None:
        return files, []
    return files, [
        f"端点引用了 OpenAPI schema，但计划里没有 kind='models' 的文件；"
        f"已补充 {added!r}（客户端会调用 <模型>.model_validate(...)，"
        "模型类必须在独立模块中定义并导入）"
    ]


def _sibling_module_path(client_path: str, suffix: str) -> str | None:
    """按客户端模块推出同目录的兄弟模块路径（petstore_client.py → petstore_exceptions.py）。"""
    normalized = _normalize_path(client_path)
    if not normalized.endswith(".py") or "/" not in normalized:
        return None
    parent, _, filename = normalized.rpartition("/")
    stem = filename[: -len(".py")]
    if stem.endswith("_client"):
        stem = stem[: -len("_client")]
    if not stem:
        return None
    return f"{parent}/{stem}_{suffix}.py"


def audit_strategy(plan: IntegrationPlan, state: PlannerState) -> list[str]:
    """检查策略字段自洽：客户端模块是否可落地、依赖是否补齐。"""
    warnings: list[str] = []
    strategy = plan.integration_strategy

    planned_paths = {_normalize_path(item.path) for item in plan.files_to_create}
    planned_paths |= {_normalize_path(item.path) for item in plan.files_to_modify}
    client_module = _normalize_path(strategy.client_module)
    if client_module and client_module not in planned_paths:
        warnings.append(
            f"integration_strategy.client_module 指向 {client_module!r}，"
            "但它既不在 files_to_create 也不在 files_to_modify 中"
        )

    declared = {name.lower() for name in state.project.dependencies}
    client = (strategy.http_client or "").strip()
    if client and client.lower() not in declared:
        proposed = {item.name.lower() for item in plan.dependencies}
        if client.lower() not in proposed:
            warnings.append(
                f"计划使用 HTTP 客户端 {client!r}，但它既不在项目依赖中，"
                "也没有出现在 dependencies 里——存在缺少依赖声明的风险"
            )

    if strategy.approach == "extend_existing_client":
        if not plan.files_to_modify:
            warnings.append(
                "integration_strategy.approach 为 extend_existing_client，"
                "但 files_to_modify 为空——没有可扩展的既有文件"
            )
        if not state.project.python_files:
            warnings.append("目标仓库中没有 Python 文件，无法扩展既有客户端")

    if strategy.approach == "new_client_module" and not plan.files_to_create:
        warnings.append(
            "integration_strategy.approach 为 new_client_module，但 files_to_create 为空"
        )

    return warnings


def audit_credentials(
    plan: IntegrationPlan, state: PlannerState
) -> tuple[list[str], list[str] | None]:
    """检查凭据处理：只允许环境变量，且计划中不得出现密钥字面量。

    返回 (warnings, 清洗后的 required_env_vars)。第二个元素为 None 表示无需修正。

    只使用高置信度的密钥模式（sk- / ghp_ / AKIA / xox / Bearer <token>）；
    刻意不做"字段名里带 token 就报警"这类猜测式检测，避免把正常描述刷成噪音。
    """
    warnings: list[str] = []

    if state.api.auth is not None and plan.authentication is None:
        warnings.append("API 声明了安全方案，但计划中 authentication 为空，缺少认证实现方案")
    if state.api.auth is None and plan.authentication is not None:
        warnings.append("API 未声明安全方案，但计划中给出了认证方案，请确认是否必要")

    cleaned_env_vars: list[str] | None = None
    if plan.authentication is not None:
        original = list(plan.authentication.required_env_vars)
        kept = [name for name in original if ENV_VAR_PATTERN.match(name.strip())]
        if kept != original:
            removed = [name for name in original if name not in kept]
            warnings.append(
                "authentication.required_env_vars 中不符合环境变量命名规范的条目已移除"
                f"（疑似硬编码凭据）：{removed}"
            )
            cleaned_env_vars = kept
        if not kept:
            warnings.append(
                "authentication 未声明任何 required_env_vars，凭据来源不明确（禁止硬编码）"
            )

    for location, text in _iter_plan_strings(plan):
        match = CREDENTIAL_IN_TEXT_PATTERN.search(text)
        if match:
            sample = sanitize(match.group(0), limit=12)
            warnings.append(
                f"计划字段 {location} 中出现疑似密钥字面量（{sample}…），凭据必须从环境变量读取"
            )
    return warnings, cleaned_env_vars


def _iter_plan_strings(plan: IntegrationPlan):
    """遍历计划中的字符串字段，产出 (字段路径, 文本)。

    跳过 endpoints：它们是 API 文档的原样拷贝，不含本项目的凭据，
    扫一遍纯粹是浪费。
    """

    def walk(node: Any, path: str):
        if isinstance(node, str):
            yield path, node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield from walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                yield from walk(value, f"{path}[{index}]")

    yield from walk(plan.model_dump(exclude={"endpoints"}), "plan")


# ------------------------------------------------------------------ 证据补齐


def _search(state: PlannerState, query: str) -> None:
    """执行一次受限检索并记录证据。"""
    state.record(
        search_code(
            state.project.root,
            query,
            context_lines=1,
            max_results=MAX_EVIDENCE_MATCHES,
            max_total_chars=MAX_EVIDENCE_SNIPPET_CHARS,
        )
    )


def _collect_evidence(state: PlannerState, *, max_queries: int = MAX_EVIDENCE_QUERIES) -> None:
    """调用方没有提供证据时，做少量**受限**定向检索来补齐。

    - 调用方已经给了证据就完全不动——尊重上游 Retrieval 的结果。
    - 只走 repository.code_search（本身有 max_results / max_total_chars 上限），
      不做全仓库遍历、不直接读文件、不缓存仓库快照。
    - 定向检索全部落空时追加一条宽泛查询兜底：让模型至少看到项目的真实代码风格，
      而不是完全空手规划。仍然只取少量片段，不会退化成"把仓库发给模型"。
    """
    if state.evidence:
        return
    for query in _evidence_queries(state)[:max_queries]:
        _search(state, query)
    if not any(evidence.matches for evidence in state.evidence):
        _search(state, FALLBACK_EVIDENCE_QUERY)


def _evidence_queries(state: PlannerState) -> list[str]:
    """按"HTTP 客户端 → 凭据读取 → 测试约定 → 领域关键词"的顺序构造检索词。"""
    queries: list[str] = []
    declared = {name.lower() for name in state.project.dependencies}
    declared_clients = [client for client in KNOWN_HTTP_CLIENTS if client in declared]
    for client in declared_clients or list(KNOWN_HTTP_CLIENTS[:2]):
        queries.append(f"import {client}")
    queries.append("os.environ")
    queries.append("def test_")
    queries.extend(_domain_keywords(state.api, limit=2))
    return queries


def _domain_keywords(api: APIInfo, *, limit: int = 2) -> list[str]:
    """从 API 名称/描述/标签中提取可用于检索的领域词（长度 >= 3，去掉通用词）。"""
    keywords: list[str] = []
    for source in (api.name, api.description or ""):
        for token in re.split(r"[^0-9a-zA-Z]+", source.lower()):
            if len(token) >= 4 and token not in STOPWORDS and token not in keywords:
                keywords.append(token)
    for endpoint in api.endpoints:
        for tag in endpoint.tags:
            tag = tag.strip().lower()
            if len(tag) >= 3 and tag not in STOPWORDS and tag not in keywords:
                keywords.append(tag)
    return keywords[:limit]


# ------------------------------------------------------------------ 兼容导出

__all__ = [
    "DeepSeekPlanner",
    "DeepSeekPlannerError",
    "DeepSeekPlannerParseError",
    "DeepSeekPlannerValidationError",
    "audit_credentials",
    "audit_exception_module",
    "audit_models_module",
    "audit_paths",
    "audit_strategy",
    "audit_test_files",
    "enforce_constraints",
    "ground_facts",
    "known_repository_files",
    "parse_plan_json",
    "resolve_endpoints",
    "sanitize",
    "validate_plan",
    "validate_relative_path",
]
