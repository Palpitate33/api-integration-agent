"""Integration Planner：把 API 信息与仓库信息转化为可执行的 IntegrationPlan。

当前实现是确定性（规则驱动）的，不调用任何 LLM：
    - 先固定数据结构与 Planner 接口，后续 LLM Planner 实现同一协议即可替换。
    - 让整个规划过程可以在无外部服务依赖的情况下被完整测试。

硬约束（本模块只分析、只规划）：
    - 只读：仓库信息来自 repository.scanner（结构）与 repository.code_search
      （受长度限制的检索），不写入、不创建、不删除任何文件。
    - 不执行 shell 命令。
    - 不生成代码：代码生成属于 tools/ 阶段。
    - API 信息与仓库信息严格分离，分别落到 IntegrationPlan.target_api 与 .repository。
"""

import re
from typing import Protocol, runtime_checkable

from integration_agent.agent.models import (
    AuthenticationPlan,
    DependencyRequirement,
    ErrorHandlingPlan,
    ErrorHandlingRule,
    FileModification,
    IntegrationPlan,
    IntegrationStrategy,
    PlannedEndpoint,
    PlannedFile,
    RepositoryContext,
    Risk,
    TargetAPI,
    TestingStrategy,
    TestSpec,
)
from integration_agent.agent.state import PlannerState
from integration_agent.api import APIEndpoint, APIInfo
from integration_agent.repository import ProjectStructure, search_code

# 未在目标项目中发现 HTTP 客户端时默认引入的库
DEFAULT_HTTP_CLIENT = "httpx"
DEFAULT_HTTP_CLIENT_CONSTRAINT = ">=0.27"

# 仓库中可能已在使用的 HTTP 客户端（顺序即检测优先级）
KNOWN_HTTP_CLIENTS = ("httpx", "requests", "aiohttp", "urllib3")

DEFAULT_MAX_ENDPOINTS = 20
MAX_EVIDENCE_MATCHES = 3
MAX_DOMAIN_MODULES = 3

# 生成关键词时过滤掉的通用词，避免把 "api"、"service" 这类词当成领域特征
STOPWORDS = frozenset(
    {
        "api",
        "apis",
        "the",
        "and",
        "for",
        "with",
        "into",
        "from",
        "client",
        "clients",
        "service",
        "services",
        "integration",
        "integrate",
        "project",
        "demo",
        "test",
        "tests",
        "example",
        "sample",
        "using",
    }
)


class PlanningError(ValueError):
    """无法基于给定输入制定集成计划时抛出。"""


@runtime_checkable
class IntegrationPlanner(Protocol):
    """Planner 接口：确定性实现与未来的 LLM 实现共用同一契约。"""

    def plan(self, state: PlannerState) -> IntegrationPlan:
        """基于输入状态产出集成方案。"""
        ...


class DeterministicPlanner:
    """基于规则的确定性 Planner：同样输入必定产出同样计划。

    仅通过 repository.code_search 做受长度限制的检索，
    不读取整个仓库，也不修改仓库。
    """

    def __init__(
        self,
        *,
        max_endpoints: int = DEFAULT_MAX_ENDPOINTS,
        default_http_client: str = DEFAULT_HTTP_CLIENT,
    ) -> None:
        self.max_endpoints = max_endpoints
        self.default_http_client = default_http_client

    def plan(self, state: PlannerState) -> IntegrationPlan:
        api, project = state.api, state.project
        if not api.endpoints:
            raise PlanningError(f"API '{api.name}' 未定义任何端点（paths 为空），无法制定集成计划")

        slug = api_slug(api.name)
        package = _target_package(project, slug)
        http_client, client_module = self._detect_http_client(state)
        domain_modules = self._find_domain_modules(state)
        selected = self._select_endpoints(api, state.request)
        dependencies = _plan_dependencies(project, http_client, self.default_http_client)
        authentication = _plan_authentication(api, slug)
        documented_codes = {
            response.status_code for item in selected for response in item.endpoint.responses
        }
        error_handling = _plan_error_handling(documented_codes, authentication, slug)
        testing = _plan_testing_strategy(
            selected, slug, project, http_client or self.default_http_client
        )
        inspected = _inspected_files(state)
        keywords = _keywords(state.request)

        return IntegrationPlan(
            integration_goal=_integration_goal(api, project, state.request),
            target_api=TargetAPI(
                name=api.name,
                version=api.version,
                base_url=api.base_url,
                total_endpoints=len(api.endpoints),
            ),
            repository=RepositoryContext(
                root=project.root,
                name=project.name,
                source_dirs=list(project.source_dirs),
                test_dirs=list(project.test_dirs),
                http_client=http_client,
                http_client_module=client_module,
                existing_modules=domain_modules,
                inspected_files=inspected,
            ),
            integration_strategy=IntegrationStrategy(
                approach="extend_existing_client" if http_client else "new_client_module",
                summary=_strategy_summary(api, project, http_client, domain_modules),
                client_module=f"{package}/{slug}_client.py",
                http_client=http_client or self.default_http_client,
            ),
            files_to_create=_plan_files_to_create(
                api, slug, package, project, authentication, http_client
            ),
            files_to_modify=_plan_files_to_modify(
                project, package, slug, dependencies, http_client, client_module, domain_modules
            ),
            dependencies=dependencies,
            endpoints=selected,
            authentication=authentication,
            error_handling=error_handling,
            testing_strategy=testing,
            assumptions=_plan_assumptions(
                api, state, selected, http_client, domain_modules, authentication, keywords
            ),
            risks=_plan_risks(
                api, selected, dependencies, authentication, documented_codes, project
            ),
        )

    def _detect_http_client(self, state: PlannerState) -> tuple[str | None, str | None]:
        """检测仓库已在使用的 HTTP 客户端，返回 (库名, 使用该库的模块路径)。

        先看依赖声明（确定性、零成本），再看代码中的 import；找到库之后
        再做一次受限检索定位它的使用位置。
        """
        declared = {_dependency_name(item) for item in state.project.dependencies}
        client = next((name for name in KNOWN_HTTP_CLIENTS if name in declared), None)
        if client is None:
            client = self._sniff_client_from_code(state)
        if client is None:
            return None, None
        return client, self._locate_client_module(state, client)

    def _sniff_client_from_code(self, state: PlannerState) -> str | None:
        """在代码中查找已知 HTTP 客户端的 import 语句。"""
        for client in KNOWN_HTTP_CLIENTS:
            result = search_code(
                state.project.root,
                client,
                context_lines=0,
                max_results=MAX_EVIDENCE_MATCHES,
            )
            state.record(result)
            if any(_is_import_of(match.snippet, client) for match in result.matches):
                return client
        return None

    def _locate_client_module(self, state: PlannerState, client: str) -> str | None:
        """定位使用该客户端的模块，供 files_to_modify 复用其配置。"""
        for pattern in (f"import {client}", f"from {client}"):
            result = search_code(state.project.root, pattern, context_lines=0, max_results=1)
            state.record(result)
            if result.matches:
                return result.matches[0].path
        return None

    def _find_domain_modules(self, state: PlannerState) -> list[str]:
        """按 API 领域关键词在仓库中寻找可复用的既有模块。"""
        found: list[str] = []
        for keyword in _domain_keywords(state.api):
            result = search_code(state.project.root, keyword, max_results=MAX_EVIDENCE_MATCHES)
            state.record(result)
            for match in result.matches:
                if _is_domain_module(match.path) and match.path not in found:
                    found.append(match.path)
        return found[:MAX_DOMAIN_MODULES]

    def _select_endpoints(self, api: APIInfo, request: str) -> list[PlannedEndpoint]:
        """挑选本次集成要使用的端点。

        关键词命中数降序；同分保持文档顺序，保证结果确定。未指定关键词时
        退回文档顺序。超过上限的部分被截断，并在 risks 中提示。
        """
        keywords = _keywords(request)
        scored: list[tuple[int, int, APIEndpoint]] = []
        for index, endpoint in enumerate(api.endpoints):
            haystack = _endpoint_haystack(endpoint)
            score = sum(1 for keyword in keywords if keyword in haystack)
            scored.append((score, index, endpoint))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            PlannedEndpoint(endpoint=endpoint, purpose=_endpoint_purpose(endpoint))
            for _, _, endpoint in scored[: self.max_endpoints]
        ]


def plan_integration(
    api: APIInfo, project: ProjectStructure, *, request: str = ""
) -> IntegrationPlan:
    """便捷入口：一次性完成规划（不保留证据）。

    需要保留检索证据时，改用 DeterministicPlanner().plan(PlannerState(...))。
    """
    return DeterministicPlanner().plan(PlannerState(api=api, project=project, request=request))


def api_slug(name: str) -> str:
    """把 API 名称转换成 snake_case 的文件名/包名片段。

    去掉 api/service/rest/openapi/client 这类通用后缀，避免生成
    demo_petstore_api_client.py 这种冗余文件名；全部由通用词组成时退回 external_api。
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    slug = re.sub(r"(_?(api|service|rest|openapi|client))+$", "", slug).strip("_")
    return slug or "external_api"


def _pascal(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_") if part)


def _keywords(text: str) -> list[str]:
    """把自由文本切成关键词：小写、去重、去过短词与停用词，保持出现顺序。"""
    # 保留 CJK 区间，中文需求描述也能切出可用关键词
    tokens = re.split(r"[^0-9a-zA-Z一-鿿]+", text.lower())
    return [token for token in dict.fromkeys(tokens) if len(token) >= 3 and token not in STOPWORDS]


def _domain_keywords(api: APIInfo) -> list[str]:
    """从 API 名称与端点标签中提取领域关键词。"""
    parts = [api.name]
    for endpoint in api.endpoints:
        parts.extend(endpoint.tags)
    return _keywords(" ".join(parts))[:MAX_DOMAIN_MODULES]


def _dependency_name(declaration: str) -> str:
    """从依赖声明中取出包名："httpx>=0.27" -> "httpx"。"""
    return re.split(r"[<>=!~;\[\s]", declaration.strip(), maxsplit=1)[0].lower()


def _is_import_of(snippet: str, module: str) -> bool:
    """判断检索片段（格式为 "行号: 代码"）是否为该模块的 import 语句。"""
    _, _, line = snippet.partition(": ")
    return f"import {module}" in line or f"from {module}" in line


def _is_domain_module(path: str) -> bool:
    """判断命中路径是否是可复用的业务模块（排除包初始化文件与测试目录）。"""
    if not path.endswith(".py") or path.endswith("/__init__.py"):
        return False
    return not any(part in {"tests", "test"} for part in path.split("/")[:-1])


def _endpoint_haystack(endpoint: APIEndpoint) -> str:
    parts = [endpoint.operation_id or "", endpoint.path, endpoint.summary or ""]
    parts.extend(endpoint.tags)
    return " ".join(parts).lower()


def _endpoint_purpose(endpoint: APIEndpoint) -> str:
    return endpoint.summary or endpoint.operation_id or f"{endpoint.method} {endpoint.path}"


def _inspected_files(state: PlannerState) -> list[str]:
    """汇总本次规划实际检索过的文件（数量受 evidence 限制约束）。"""
    inspected: list[str] = []
    for evidence in state.evidence:
        for path in evidence.paths:
            if path not in inspected:
                inspected.append(path)
    return inspected


def _package_exists(project: ProjectStructure, package: str) -> bool:
    prefix = f"{package}/"
    return any(path.startswith(prefix) for path in project.python_files)


def _target_package(project: ProjectStructure, slug: str) -> str:
    """决定新代码放入哪个包。

    1. source_dirs 中本身是包（含 __init__.py）→ 直接复用；
    2. src/lib 布局 → 复用其下已有的包，否则在容器下新建；
    3. 根目录已存在与项目同名的包 → 复用（如 demo-project 的 demo_project/）；
    4. 其他情况 → 在仓库根新建 <slug> 包。
    """
    for name in project.source_dirs:
        if f"{name}/__init__.py" in project.python_files:
            return name
    for container in ("src", "lib"):
        if container not in project.source_dirs:
            continue
        for path in project.python_files:
            if path.startswith(f"{container}/") and path.endswith("/__init__.py"):
                return path[: -len("/__init__.py")]
        return f"{container}/{slug}"
    root_package = project.name.replace("-", "_")
    if root_package and f"{root_package}/__init__.py" in project.python_files:
        return root_package
    return slug


def _dependency_manifest(project: ProjectStructure) -> str | None:
    for candidate in ("pyproject.toml", "requirements.txt"):
        if candidate in project.manifest_files:
            return candidate
    return None


def _integration_goal(api: APIInfo, project: ProjectStructure, request: str) -> str:
    if request.strip():
        return f"将 {api.name}（v{api.version}）集成到 {project.name}：{request.strip()}"
    return f"将 {api.name}（v{api.version}）集成到 {project.name}"


def _strategy_summary(
    api: APIInfo,
    project: ProjectStructure,
    http_client: str | None,
    domain_modules: list[str],
) -> str:
    if http_client:
        base = f"复用 {project.name} 中已有的 {http_client} 客户端，新建独立模块封装 {api.name}"
    else:
        base = f"{project.name} 中未发现 HTTP 客户端，新建基于 {DEFAULT_HTTP_CLIENT} 的独立模块"
    if domain_modules:
        return f"{base}；并接入既有模块 {', '.join(domain_modules)}"
    return base


def _plan_dependencies(
    project: ProjectStructure, http_client: str | None, default_client: str
) -> list[DependencyRequirement]:
    """规划运行时依赖：HTTP 客户端 + 生成模型所需的 pydantic。"""
    declared = {_dependency_name(item) for item in project.dependencies}
    client = http_client or default_client
    installed = client in declared
    requirements = [
        DependencyRequirement(
            name=client,
            version=None if installed else DEFAULT_HTTP_CLIENT_CONSTRAINT,
            reason="HTTP 客户端" if installed else "项目未声明 HTTP 客户端，需要新增",
            already_installed=installed,
        )
    ]
    if "pydantic" not in declared:
        requirements.append(
            DependencyRequirement(
                name="pydantic",
                version=">=2.0",
                reason="生成的请求/响应模型",
                already_installed=False,
            )
        )
    return requirements


def _plan_authentication(api: APIInfo, slug: str) -> AuthenticationPlan | None:
    """根据 OpenAPI securityScheme 规划认证方式与凭据位置。"""
    auth = api.auth
    if auth is None:
        return None
    prefix = slug.upper()
    if auth.type == "apiKey":
        header = auth.name or "X-API-Key"
        return AuthenticationPlan(
            scheme="apiKey",
            strategy=f"在每个请求的 {auth.location or 'header'} 中附加 {header}",
            config_location="环境变量（由 config 模块读取，禁止硬编码或提交到版本库）",
            required_env_vars=[f"{prefix}_API_KEY"],
            header_name=header,
        )
    if auth.type == "http" and (auth.scheme or "").lower() == "basic":
        return AuthenticationPlan(
            scheme="http-basic",
            strategy="使用 HTTP Basic：用户名/密码经 base64 编码放入 Authorization 头",
            config_location="环境变量",
            required_env_vars=[f"{prefix}_USERNAME", f"{prefix}_PASSWORD"],
            header_name="Authorization",
        )
    if auth.type == "http":
        credential = auth.bearer_format or auth.scheme or "Bearer"
        return AuthenticationPlan(
            scheme=f"http-{auth.scheme or 'bearer'}",
            strategy=f"在 Authorization 头中携带 {credential} 凭据",
            config_location="环境变量",
            required_env_vars=[f"{prefix}_TOKEN"],
            header_name="Authorization",
        )
    if auth.type == "oauth2":
        return AuthenticationPlan(
            scheme="oauth2",
            strategy="按 client credentials 流程获取 access token，并在过期前刷新",
            config_location="环境变量 + 内存中的令牌缓存（不落盘）",
            required_env_vars=[f"{prefix}_CLIENT_ID", f"{prefix}_CLIENT_SECRET"],
            header_name="Authorization",
        )
    return AuthenticationPlan(
        scheme=auth.type,
        strategy=f"按 OpenAPI securityScheme '{auth.type}' 的要求注入凭据",
        config_location="环境变量",
        required_env_vars=[f"{prefix}_TOKEN"],
    )


def _plan_error_handling(
    documented_codes: set[str],
    authentication: AuthenticationPlan | None,
    slug: str,
) -> ErrorHandlingPlan:
    """为 400 / 401 / 403 / 404 / 429 / 5xx 逐项规定处理策略。"""
    auth_hint = ""
    if authentication is not None and authentication.required_env_vars:
        auth_hint = f"（检查 {authentication.required_env_vars[0]}）"
    has_5xx = any(code.startswith("5") for code in documented_codes)
    rules = [
        ErrorHandlingRule(
            status_code="400",
            category="validation",
            strategy="解析错误响应体中的校验细节并抛出请求参数异常，不重试",
            documented_in_spec="400" in documented_codes,
        ),
        ErrorHandlingRule(
            status_code="401",
            category="authentication",
            strategy=f"凭据缺失或已失效{auth_hint}，抛出认证异常，不重试",
            documented_in_spec="401" in documented_codes,
        ),
        ErrorHandlingRule(
            status_code="403",
            category="authorization",
            strategy="权限不足：抛出授权异常并保留原始响应，不重试",
            documented_in_spec="403" in documented_codes,
        ),
        ErrorHandlingRule(
            status_code="404",
            category="not_found",
            strategy="资源不存在：抛出资源不存在异常，由调用方决定是否降级处理",
            documented_in_spec="404" in documented_codes,
        ),
        ErrorHandlingRule(
            status_code="429",
            category="rate_limit",
            strategy="优先按 Retry-After 头退避重试，无该头时指数退避，超过上限抛出限流异常",
            retryable=True,
            documented_in_spec="429" in documented_codes,
        ),
        ErrorHandlingRule(
            status_code="5xx",
            category="server_error",
            strategy="幂等请求（GET/PUT/DELETE）指数退避重试，POST/PATCH 不自动重试",
            retryable=True,
            documented_in_spec=has_5xx,
        ),
    ]
    notes = ["所有错误都必须保留原始状态码与响应体，便于排查"]
    if "429" in documented_codes:
        notes.append("文档已声明 429：重试上限与退避基数需按文档中的限流说明确定")
    return ErrorHandlingPlan(exception_type=f"{_pascal(slug)}APIError", rules=rules, notes=notes)


def _plan_files_to_create(
    api: APIInfo,
    slug: str,
    package: str,
    project: ProjectStructure,
    authentication: AuthenticationPlan | None,
    http_client: str | None,
) -> list[PlannedFile]:
    client = http_client or DEFAULT_HTTP_CLIENT
    files = [
        PlannedFile(
            path=f"{package}/{slug}_client.py",
            kind="client",
            purpose=(
                f"封装 {api.name} 的 HTTP 调用：每个端点一个方法，"
                f"统一处理认证、超时与错误映射（基于 {client}）"
            ),
        ),
        PlannedFile(
            path=f"{package}/{slug}_models.py",
            kind="models",
            purpose="请求/响应的 Pydantic 模型，与 OpenAPI schema 对应",
        ),
        PlannedFile(
            path=f"{package}/{slug}_exceptions.py",
            kind="exceptions",
            purpose=f"HTTP 状态码到异常类型的映射，基类 {_pascal(slug)}APIError",
        ),
    ]
    if authentication is not None or api.base_url is None:
        files.append(
            PlannedFile(
                path=f"{package}/{slug}_config.py",
                kind="config",
                purpose="base_url 与认证凭据的读取与校验（从环境变量加载）",
            )
        )
    if not _package_exists(project, package):
        files.append(
            PlannedFile(
                path=f"{package}/__init__.py",
                kind="package_init",
                purpose="声明新包并导出客户端入口",
            )
        )
    return files


def _plan_files_to_modify(
    project: ProjectStructure,
    package: str,
    slug: str,
    dependencies: list[DependencyRequirement],
    http_client: str | None,
    client_module: str | None,
    domain_modules: list[str],
) -> list[FileModification]:
    modifications: list[FileModification] = []
    new_dependencies = [item for item in dependencies if not item.already_installed]
    manifest = _dependency_manifest(project)
    if new_dependencies and manifest:
        modifications.append(
            FileModification(
                path=manifest,
                reason="声明新增的运行时依赖",
                changes=[f"{item.name}{item.version or ''}" for item in new_dependencies],
            )
        )
    if client_module and http_client:
        modifications.append(
            FileModification(
                path=client_module,
                reason=f"复用该模块已配置的 {http_client} 客户端（超时/重试/代理），避免重复配置",
                changes=["抽取或复用既有的 transport/session 配置"],
            )
        )
    for module in domain_modules:
        modifications.append(
            FileModification(
                path=module,
                reason="接入新集成的调用，复用既有领域逻辑而非重复实现",
                changes=["调用新客户端并复用既有数据处理流程"],
            )
        )
    if _package_exists(project, package):
        modifications.append(
            FileModification(
                path=f"{package}/__init__.py",
                reason="导出新的客户端入口，保持包内 API 一致",
                changes=[f"导出 {slug} 客户端与异常类型"],
            )
        )
    return modifications


def _plan_testing_strategy(
    selected: list[PlannedEndpoint],
    slug: str,
    project: ProjectStructure,
    http_client: str,
) -> TestingStrategy:
    test_dir = project.test_dirs[0] if project.test_dirs else "tests"
    names = [item.operation_id or f"{item.method} {item.path}" for item in selected]
    return TestingStrategy(
        unit_tests=[
            TestSpec(
                path=f"{test_dir}/test_{slug}_client.py",
                purpose="不发起网络请求，验证请求构造、参数序列化与响应解析",
                covers=[f"{name}：请求与响应处理" for name in names] + ["错误码到异常的映射"],
            )
        ],
        integration_tests=[
            TestSpec(
                path=f"{test_dir}/test_{slug}_integration.py",
                purpose=f"用 {http_client} 的 MockTransport 走完整链路，不依赖外部网络",
                covers=["端到端调用", "认证凭据注入", "重试与退避行为"],
            )
        ],
        contract_tests=[
            TestSpec(
                path=f"{test_dir}/test_{slug}_contract.py",
                purpose="用 OpenAPI 文档中的 schema 校验响应结构，捕捉服务端契约漂移",
                covers=[f"{name}：响应结构与 OpenAPI schema 一致" for name in names],
            )
        ],
    )


def _plan_assumptions(
    api: APIInfo,
    state: PlannerState,
    selected: list[PlannedEndpoint],
    http_client: str | None,
    domain_modules: list[str],
    authentication: AuthenticationPlan | None,
    keywords: list[str],
) -> list[str]:
    assumptions = [
        f"API 规范来自 {api.name} v{api.version}，假定集成期间保持稳定",
    ]
    if len(selected) < len(api.endpoints):
        if keywords:
            detail = f"按关键词 {keywords} 优先选取"
        else:
            detail = "按文档顺序选取"
        assumptions.append(f"本次集成覆盖 {len(selected)}/{len(api.endpoints)} 个端点，{detail}")
    if http_client:
        assumptions.append(f"复用仓库中已有的 {http_client}，与其保持一致的调用风格")
    else:
        assumptions.append(
            f"仓库中未发现既有 HTTP 客户端，新建独立模块并使用 {DEFAULT_HTTP_CLIENT}"
        )
    if domain_modules:
        assumptions.append(f"复用既有模块 {', '.join(domain_modules)}，不重复实现领域逻辑")
    if authentication is not None:
        assumptions.append("凭据通过环境变量提供，不写入代码或版本库")
    else:
        assumptions.append("该 API 无需认证（OpenAPI 未声明根级 security）")
    if api.base_url is None:
        assumptions.append("OpenAPI 未声明 servers，base_url 必须由使用方显式配置")
    if not state.request.strip():
        assumptions.append("未提供额外需求描述，端点选择按文档顺序进行")
    assumptions.append("超时与重试参数先取保守默认值，集成后按实测调整")
    return assumptions


def _plan_risks(
    api: APIInfo,
    selected: list[PlannedEndpoint],
    dependencies: list[DependencyRequirement],
    authentication: AuthenticationPlan | None,
    documented_codes: set[str],
    project: ProjectStructure,
) -> list[Risk]:
    risks: list[Risk] = []
    if api.base_url is None:
        risks.append(
            Risk(
                description="OpenAPI 未声明 base_url，集成后无法确定实际服务地址",
                severity="high",
                mitigation="在 config 中强制要求显式配置，并在启动时校验",
            )
        )
    if len(selected) < len(api.endpoints):
        risks.append(
            Risk(
                description=(
                    f"API 共 {len(api.endpoints)} 个端点，本次仅集成 {len(selected)} 个，"
                    f"其余端点未覆盖"
                ),
                severity="medium",
                mitigation="确认优先级后分批集成",
            )
        )
    new_dependencies = [item for item in dependencies if not item.already_installed]
    if new_dependencies:
        names = ", ".join(item.name for item in new_dependencies)
        risks.append(
            Risk(
                description=f"新增依赖 {names} 可能与既有依赖产生版本冲突",
                severity="medium",
                mitigation="安装后运行完整测试套件验证",
            )
        )
    if _dependency_manifest(project) is None:
        risks.append(
            Risk(
                description="未找到 pyproject.toml / requirements.txt 依赖清单，新增依赖无处声明",
                severity="medium",
                mitigation="先为项目补齐依赖清单文件",
            )
        )
    if authentication is not None:
        risks.append(
            Risk(
                description="缺少可用凭据时，集成测试无法覆盖真实调用路径",
                severity="medium",
                mitigation="提供 sandbox 凭据，或使用 MockTransport 覆盖请求链路",
            )
        )
    if "429" in documented_codes:
        risks.append(
            Risk(
                description="API 存在限流（429），高频调用可能被拒绝",
                severity="medium",
                mitigation="实现带退避的重试并限制并发",
            )
        )
    if not documented_codes or not any(code.startswith("5") for code in documented_codes):
        risks.append(
            Risk(
                description="文档未声明 5xx 响应，真实服务行为可能与预期不一致",
                severity="low",
                mitigation="保留原始响应体，按实测结果补充错误映射",
            )
        )
    return risks
