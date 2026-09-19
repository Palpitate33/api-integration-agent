"""Integration Planner 单元测试。

覆盖用户要求的 8 个场景：
    1. 正常生成 IntegrationPlan          5. 需要新增 dependency
    2. API 没有合适 endpoint             6. API 使用 authentication
    3. repository 中存在已有 HTTP client  7. API 存在 429 rate limit
    4. repository 中存在 payment/service 模块  8. Planner 输出结构化 Pydantic Model

以及 Planner 的硬约束：不修改仓库、不创建文件、不执行 shell 命令、
不把整个仓库塞进上下文。
"""

import json
from pathlib import Path

import pytest

from integration_agent.agent import (
    DeterministicPlanner,
    IntegrationPlan,
    IntegrationPlanner,
    PlannerState,
    PlanningError,
    api_slug,
    plan_integration,
)
from integration_agent.agent.planner import MAX_EVIDENCE_MATCHES
from integration_agent.api import parse_openapi, parse_openapi_text
from integration_agent.repository import ProjectStructure, scan_repository

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PETSTORE_PATH = EXAMPLES / "openapi" / "petstore.yaml"
DEMO_PROJECT_PATH = EXAMPLES / "demo_project"

SAMPLE_API = parse_openapi_text(
    """
openapi: 3.0.3
info: {title: Sample API, version: "1.0"}
servers: [{url: https://sample.example.com}]
paths:
  /items:
    get:
      operationId: listItems
      tags: [items]
      parameters:
        - {name: limit, in: query, schema: {type: integer}}
      responses:
        "200": {description: ok}
"""
)

NO_BASE_URL_API = parse_openapi_text(
    """
openapi: 3.0.3
info: {title: Sample API, version: "1.0"}
paths:
  /items:
    get:
      operationId: listItems
      responses:
        "200": {description: ok}
"""
)

BEARER_API = parse_openapi_text(
    """
openapi: 3.0.3
info: {title: Sample API, version: "1.0"}
servers: [{url: https://sample.example.com}]
security: [{bearerAuth: []}]
paths:
  /items:
    get:
      operationId: listItems
      responses:
        "200": {description: ok}
components:
  securitySchemes:
    bearerAuth: {type: http, scheme: bearer, bearerFormat: JWT}
"""
)

RATE_LIMITED_API = parse_openapi_text(
    """
openapi: 3.0.3
info: {title: Rate Limited API, version: "1"}
servers: [{url: https://limited.example.com}]
paths:
  /jobs:
    post:
      operationId: submitJob
      responses:
        "202": {description: accepted}
        "429": {description: too many requests}
        "500": {description: server error}
"""
)

PAYMENT_API = parse_openapi_text(
    """
openapi: 3.0.3
info: {title: Acme Payment API, version: "2.1"}
servers: [{url: https://payments.example.com}]
paths:
  /charges:
    post:
      operationId: createCharge
      tags: [payments]
      responses:
        "201": {description: created}
"""
)

PAYMENT_MODULE = '''"""Payment domain logic（既有模块，集成时应被复用）。"""

from dataclasses import dataclass


@dataclass
class Payment:
    """一笔支付记录。"""

    amount: float
    currency: str


def format_payment(payment: Payment) -> str:
    """把支付记录格式化成展示文本。"""
    return f"{payment.amount} {payment.currency}"
'''

HTTP_MODULE = '''"""共享 HTTP 工具（既有模块）。"""

import httpx


def build_client() -> httpx.Client:
    """构造带统一超时配置的客户端。"""
    return httpx.Client(timeout=10.0)
'''

REQUESTS_MODULE = '''"""既有的 HTTP 封装。"""

import requests


def get(url: str) -> requests.Response:
    return requests.get(url, timeout=10)
'''


def _make_project(
    root: Path,
    files: dict[str, str] | None = None,
    *,
    name: str = "sample-project",
    dependencies: list[str] | None = None,
) -> ProjectStructure:
    """在临时目录中构造一个可扫描的 Python 项目。"""
    root.mkdir(parents=True, exist_ok=True)
    deps = ", ".join(json.dumps(item) for item in dependencies or [])
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\ndependencies = [{deps}]\n',
        encoding="utf-8",
    )
    for relative, content in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return scan_repository(root)


# ---------------------------------------------------------------- 场景 1：正常生成


def test_plan_generated_from_real_fixtures() -> None:
    """用 examples 中的真实 fixture 生成完整计划。"""
    plan = plan_integration(parse_openapi(PETSTORE_PATH), scan_repository(DEMO_PROJECT_PATH))

    assert isinstance(plan, IntegrationPlan)
    assert plan.target_api.name == "Demo Petstore API"
    assert plan.target_api.version == "1.0.0"
    assert plan.target_api.base_url == "https://petstore.example.com/v1"
    assert plan.target_api.total_endpoints == 3

    assert plan.repository.name == "demo-project"
    # demo fixture 显式声明了 httpx：Planner 应识别为已有客户端并复用
    assert plan.repository.http_client == "httpx"
    assert plan.integration_strategy.approach == "extend_existing_client"
    assert plan.integration_strategy.client_module == "demo_project/demo_petstore_client.py"
    assert plan.integration_strategy.http_client == "httpx"

    assert [item.operation_id for item in plan.endpoints] == ["listPets", "createPet", "getPet"]
    assert plan.endpoints[0].method == "GET"
    assert plan.endpoints[0].path == "/pets"
    assert plan.error_handling.exception_type == "DemoPetstoreAPIError"

    created = {item.path: item.kind for item in plan.files_to_create}
    assert created == {
        "demo_project/demo_petstore_client.py": "client",
        "demo_project/demo_petstore_models.py": "models",
        "demo_project/demo_petstore_exceptions.py": "exceptions",
        "demo_project/demo_petstore_config.py": "config",
    }
    modified = {item.path for item in plan.files_to_modify}
    assert modified == {"pyproject.toml", "demo_project/__init__.py"}

    assert plan.integration_goal.startswith("将 Demo Petstore API（v1.0.0）集成到 demo-project")
    assert plan.assumptions
    assert plan.testing_strategy.unit_tests[0].path == "tests/test_demo_petstore_client.py"


def test_goal_includes_user_request() -> None:
    api = parse_openapi(PETSTORE_PATH)
    project = scan_repository(DEMO_PROJECT_PATH)
    plan = plan_integration(api, project, request="只接入查询接口")
    assert "只接入查询接口" in plan.integration_goal


def test_plan_is_deterministic() -> None:
    """同样输入必须产出完全相同的计划（无哈希随机、无时间戳）。"""
    api = parse_openapi(PETSTORE_PATH)
    project = scan_repository(DEMO_PROJECT_PATH)
    assert plan_integration(api, project) == plan_integration(api, project)


# --------------------------------------------------------- 场景 2：没有合适 endpoint


def test_api_without_endpoints_cannot_be_planned(tmp_path: Path) -> None:
    empty_api = parse_openapi_text(
        'openapi: 3.0.3\ninfo: {title: Empty API, version: "1"}\npaths: {}\n'
    )
    project = _make_project(tmp_path / "repo")
    with pytest.raises(PlanningError, match="未定义任何端点"):
        plan_integration(empty_api, project)


def test_planner_does_not_touch_repository_on_error(tmp_path: Path) -> None:
    """规划失败时也不得触碰仓库。"""
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    empty_api = parse_openapi_text(
        'openapi: 3.0.3\ninfo: {title: Empty API, version: "1"}\npaths: {}\n'
    )
    before = _snapshot(tmp_path / "repo")
    with pytest.raises(PlanningError):
        plan_integration(empty_api, project)
    assert _snapshot(tmp_path / "repo") == before


# --------------------------------------------------- 场景 3：仓库已有 HTTP client


def test_reuses_declared_http_client(tmp_path: Path) -> None:
    project = _make_project(
        tmp_path / "repo",
        {
            "src/sample_app/__init__.py": '"""Sample app."""\n',
            "src/sample_app/http.py": HTTP_MODULE,
        },
        dependencies=["httpx>=0.27"],
    )
    plan = plan_integration(SAMPLE_API, project)

    assert plan.repository.http_client == "httpx"
    assert plan.repository.http_client_module == "src/sample_app/http.py"
    assert plan.integration_strategy.approach == "extend_existing_client"
    assert plan.integration_strategy.client_module == "src/sample_app/sample_client.py"

    dependencies = {item.name: item for item in plan.dependencies}
    assert dependencies["httpx"].already_installed is True
    assert dependencies["httpx"].version is None

    reason = next(
        item.reason for item in plan.files_to_modify if item.path == "src/sample_app/http.py"
    )
    assert "复用" in reason


def test_detects_http_client_from_existing_import(tmp_path: Path) -> None:
    """依赖清单没写、但代码里已在用 requests 时同样要识别出来。"""
    project = _make_project(
        tmp_path / "repo",
        {
            "app/__init__.py": '"""App."""\n',
            "app/http.py": REQUESTS_MODULE,
        },
    )
    plan = plan_integration(SAMPLE_API, project)

    assert plan.repository.http_client == "requests"
    assert plan.repository.http_client_module == "app/http.py"
    assert plan.integration_strategy.approach == "extend_existing_client"
    assert plan.dependencies[0].name == "requests"
    assert plan.dependencies[0].already_installed is False


# ------------------------------------------- 场景 4：仓库已有 payment/service 模块


def test_reuses_existing_domain_module(tmp_path: Path) -> None:
    project = _make_project(
        tmp_path / "repo",
        {
            "payments/__init__.py": '"""Payments package."""\n',
            "payments/service.py": PAYMENT_MODULE,
            "tests/test_payment.py": "def test_payment() -> None:\n    pass\n",
        },
    )
    plan = plan_integration(PAYMENT_API, project)

    # 只保留业务模块：__init__.py 与 tests/ 下的命中被剔除
    assert plan.repository.existing_modules == ["payments/service.py"]
    assert any("payments/service.py" in item for item in plan.assumptions)
    assert any("复用" in item.reason for item in plan.files_to_modify)
    assert plan.integration_strategy.summary.endswith("并接入既有模块 payments/service.py")


# ----------------------------------------------------- 场景 5：需要新增 dependency


def test_missing_dependencies_are_planned(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = plan_integration(SAMPLE_API, project)

    dependencies = {item.name: item for item in plan.dependencies}
    assert dependencies["httpx"].already_installed is False
    assert dependencies["httpx"].version == ">=0.27"
    assert dependencies["pydantic"].already_installed is False

    manifest = next(item for item in plan.files_to_modify if item.path == "pyproject.toml")
    assert manifest.changes == ["httpx>=0.27", "pydantic>=2.0"]
    assert any("httpx" in risk.description for risk in plan.risks)


def test_no_new_dependency_when_client_already_declared(tmp_path: Path) -> None:
    project = _make_project(
        tmp_path / "repo",
        {"src/app/__init__.py": '"""App."""\n'},
        dependencies=["httpx>=0.27", "pydantic>=2.0"],
    )
    plan = plan_integration(SAMPLE_API, project)

    assert all(item.already_installed for item in plan.dependencies)
    assert not any(item.path == "pyproject.toml" for item in plan.files_to_modify)


# -------------------------------------------------------- 场景 6：API 使用认证


def test_api_key_authentication(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = plan_integration(parse_openapi(PETSTORE_PATH), project)

    auth = plan.authentication
    assert auth is not None
    assert auth.scheme == "apiKey"
    assert auth.header_name == "X-API-Key"
    assert auth.required_env_vars == ["DEMO_PETSTORE_API_KEY"]
    assert "环境变量" in auth.config_location

    # 凭据需要 config 模块承载
    assert "config" in {item.kind for item in plan.files_to_create}
    rule = next(r for r in plan.error_handling.rules if r.status_code == "401")
    assert "DEMO_PETSTORE_API_KEY" in rule.strategy


def test_bearer_authentication(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = plan_integration(BEARER_API, project)

    auth = plan.authentication
    assert auth is not None
    assert auth.scheme == "http-bearer"
    assert auth.header_name == "Authorization"
    assert auth.required_env_vars == ["SAMPLE_TOKEN"]


def test_no_authentication_plan_without_security(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = plan_integration(SAMPLE_API, project)

    assert plan.authentication is None
    assert "config" not in {item.kind for item in plan.files_to_create}
    assert any("无需认证" in item for item in plan.assumptions)


# ------------------------------------------------------ 场景 7：API 存在 429 限流


def test_rate_limit_is_planned(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = plan_integration(RATE_LIMITED_API, project)

    rules = {rule.status_code: rule for rule in plan.error_handling.rules}
    assert rules["429"].category == "rate_limit"
    assert rules["429"].retryable is True
    assert rules["429"].documented_in_spec is True
    assert "Retry-After" in rules["429"].strategy

    # 文档声明的 5xx / 未声明的 400 要区分开
    assert rules["5xx"].documented_in_spec is True
    assert rules["400"].documented_in_spec is False

    assert any("429" in note for note in plan.error_handling.notes)
    assert any("限流" in risk.description for risk in plan.risks)


def test_all_error_categories_always_covered(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = plan_integration(SAMPLE_API, project)

    assert [rule.status_code for rule in plan.error_handling.rules] == [
        "400",
        "401",
        "403",
        "404",
        "429",
        "5xx",
    ]
    assert all(rule.strategy for rule in plan.error_handling.rules)


# -------------------------------------------- 场景 8：结构化 Pydantic Model 输出


def test_plan_is_structured_and_roundtrips() -> None:
    plan = plan_integration(parse_openapi(PETSTORE_PATH), scan_repository(DEMO_PROJECT_PATH))

    payload = plan.model_dump()
    required = {
        "integration_goal",
        "target_api",
        "integration_strategy",
        "files_to_create",
        "files_to_modify",
        "dependencies",
        "endpoints",
        "authentication",
        "error_handling",
        "testing_strategy",
        "assumptions",
        "risks",
    }
    assert required <= set(payload)

    # 可被 Code Generator 直接反序列化
    assert IntegrationPlan.model_validate_json(plan.model_dump_json()) == plan


def test_core_information_is_in_typed_fields_not_free_text() -> None:
    """方法、路径、参数、依赖名、文件路径等核心信息必须是结构化字段。"""
    plan = plan_integration(parse_openapi(PETSTORE_PATH), scan_repository(DEMO_PROJECT_PATH))

    list_pets = plan.endpoints[0]
    assert (list_pets.method, list_pets.path, list_pets.operation_id) == (
        "GET",
        "/pets",
        "listPets",
    )
    parameter = list_pets.endpoint.parameters[0]
    assert (parameter.name, parameter.location, parameter.schema_type) == (
        "limit",
        "query",
        "integer",
    )

    assert plan.dependencies[0].name == "httpx"
    assert plan.error_handling.rules[0].category == "validation"
    assert plan.files_to_create[0].kind == "client"
    assert plan.testing_strategy.unit_tests[0].covers


# ------------------------------------------------------------- Planner 硬约束


def test_planner_does_not_modify_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    project = _make_project(
        root,
        {"app/__init__.py": '"""App."""\n', "app/http.py": REQUESTS_MODULE},
    )
    before = _snapshot(root)

    DeterministicPlanner().plan(PlannerState(api=SAMPLE_API, project=project))

    assert _snapshot(root) == before


def test_planner_satisfies_protocol(tmp_path: Path) -> None:
    """实现同一协议，后续 LLM Planner 可以直接替换。"""
    planner = DeterministicPlanner()
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    assert isinstance(planner, IntegrationPlanner)
    state = PlannerState(api=SAMPLE_API, project=project)
    assert isinstance(planner.plan(state), IntegrationPlan)


def test_evidence_is_bounded(tmp_path: Path) -> None:
    """仓库内容只以受长度限制的检索片段进入状态，且可被审计。"""
    root = tmp_path / "repo"
    project = _make_project(
        root,
        {
            "app/__init__.py": '"""App."""\n',
            "app/http.py": REQUESTS_MODULE,
            "app/more.py": "import requests\n",
        },
    )
    state = PlannerState(api=SAMPLE_API, project=project)
    DeterministicPlanner().plan(state)

    assert state.evidence
    for evidence in state.evidence:
        assert len(evidence.matches) <= MAX_EVIDENCE_MATCHES
        assert all(len(match.snippet) <= 400 for match in evidence.matches)
    assert sum(item.total_chars for item in state.evidence) < 10_000


def test_planner_reads_only_python_sources(tmp_path: Path) -> None:
    """证据里不应出现被扫描仓库之外的文件。"""
    root = tmp_path / "repo"
    project = _make_project(root, {"app/http.py": REQUESTS_MODULE})
    state = PlannerState(api=SAMPLE_API, project=project)
    DeterministicPlanner().plan(state)

    inspected = {path for evidence in state.evidence for path in evidence.paths}
    assert inspected == {"app/http.py"}


# ------------------------------------------------------------------ 端点选择


def test_endpoint_selection_follows_request_keywords(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    api = parse_openapi(PETSTORE_PATH)

    # 未指定需求时按文档顺序；指定需求后命中的端点排到最前
    assert [item.operation_id for item in plan_integration(api, project).endpoints] == [
        "listPets",
        "createPet",
        "getPet",
    ]
    focused = DeterministicPlanner(max_endpoints=1).plan(
        PlannerState(api=api, project=project, request="create a new pet")
    )
    assert [item.operation_id for item in focused.endpoints] == ["createPet"]
    assert any("create" in item for item in focused.assumptions)


def test_max_endpoints_truncates_and_reports_risk(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = DeterministicPlanner(max_endpoints=1).plan(
        PlannerState(api=parse_openapi(PETSTORE_PATH), project=project)
    )

    assert len(plan.endpoints) == 1
    assert plan.target_api.total_endpoints == 3
    assert any("仅集成 1 个" in risk.description for risk in plan.risks)


def test_missing_base_url_is_high_risk(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "repo", {"app/__init__.py": '"""App."""\n'})
    plan = plan_integration(NO_BASE_URL_API, project)

    assert plan.target_api.base_url is None
    assert "config" in {item.kind for item in plan.files_to_create}
    assert any(risk.severity == "high" for risk in plan.risks)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text('"""App."""\n', encoding="utf-8")
    project = scan_repository(root)

    plan = plan_integration(SAMPLE_API, project)

    assert not any(item.path == "pyproject.toml" for item in plan.files_to_modify)
    assert any("依赖清单" in risk.description for risk in plan.risks)


# ------------------------------------------------------------------- 辅助函数


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Demo Petstore API", "demo_petstore"),
        ("Acme Payment API", "acme_payment"),
        ("Stripe", "stripe"),
        ("Pet Service", "pet"),
        ("", "external_api"),
        ("   ", "external_api"),
    ],
)
def test_api_slug(name: str, expected: str) -> None:
    assert api_slug(name) == expected


def test_api_slug_never_returns_a_bare_generic_name() -> None:
    assert api_slug("API") == "external_api"


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """记录目录下所有文件的大小与修改时间，用于证明 Planner 没有写仓库。"""
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
