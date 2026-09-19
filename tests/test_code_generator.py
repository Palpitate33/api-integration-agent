"""Code Generator 单元测试。

覆盖要求的 10 个场景：
    1. files_to_create → GeneratedFile(action="create")
    2. GET / POST endpoint 被正确消费
    3. authentication 影响生成代码
    4. 429 retryable=true 影响生成代码（以及 retryable=false 不生成 retry）
    5. 新增 dependency / 已安装不重复添加
    6. files_to_modify → 结构化 modification
    7. testing strategy 影响生成的测试 artifact
    8. 确定性：相同 plan 连续生成结果一致
    9. 不修改 Repository（文件快照证明）
    10. Pydantic 契约：isinstance + 结构化字段 + 往返序列化

以及生成代码的语法有效性（compile）、Protocol 一致性、安全边界等。
"""

import json
from pathlib import Path

import pytest

from integration_agent.agent import IntegrationPlan, plan_integration
from integration_agent.api import parse_openapi, parse_openapi_text
from integration_agent.generation import (
    CodeGenerator,
    DependencyChange,
    DeterministicCodeGenerator,
    GeneratedArtifacts,
    GeneratedFile,
    generate_code,
)
from integration_agent.repository import scan_repository

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
      parameters:
        - {name: limit, in: query, schema: {type: integer}}
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


@pytest.fixture(scope="module")
def petstore_plan() -> IntegrationPlan:
    return plan_integration(parse_openapi(PETSTORE_PATH), scan_repository(DEMO_PROJECT_PATH))


@pytest.fixture(scope="module")
def petstore_artifacts(petstore_plan: IntegrationPlan) -> GeneratedArtifacts:
    return generate_code(petstore_plan)


def _make_project(
    root: Path,
    files: dict[str, str] | None = None,
    *,
    dependencies: list[str] | None = None,
) -> None:
    """构造可扫描的临时项目。"""
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if dependencies is not None:
        deps = ", ".join(json.dumps(item) for item in dependencies)
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "sample-project"\nversion = "0.1.0"\ndependencies = [{deps}]\n',
            encoding="utf-8",
        )
    elif not (root / "pyproject.toml").exists() and not list(root.iterdir()):
        (root / "pyproject.toml").write_text(
            '[project]\nname = "sample-project"\nversion = "0.1.0"\ndependencies = []\n',
            encoding="utf-8",
        )


def _plan(api, root: Path) -> IntegrationPlan:
    _make_project(root, {"app/__init__.py": '"""App."""\n'})
    return plan_integration(api, scan_repository(root))


def _file(artifacts: GeneratedArtifacts, path: str) -> GeneratedFile:
    return next(item for item in artifacts.files if item.path == path)


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --------------------------------------------- Test 1：files_to_create → create


def test_files_to_create_become_create_artifacts(
    petstore_plan: IntegrationPlan, petstore_artifacts: GeneratedArtifacts
) -> None:
    created = {item.path: item for item in petstore_artifacts.files if item.action == "create"}
    for planned in petstore_plan.files_to_create:
        assert planned.path in created
        artifact = created[planned.path]
        assert artifact.action == "create"
        assert artifact.content
        assert artifact.purpose == planned.purpose


# ------------------------------------------------------ Test 2：GET / POST 消费


def test_get_and_post_endpoints_consumed(petstore_artifacts: GeneratedArtifacts) -> None:
    client = _file(petstore_artifacts, "demo_project/demo_petstore_client.py").content
    assert "def list_pets(self, *, limit: int | None = None) -> list[Pet]:" in client
    assert "def create_pet(self, *, payload: NewPet) -> Pet:" in client
    assert "def get_pet(self, *, pet_id: str) -> Pet:" in client
    assert '"GET", url' in client
    assert '"POST", url' in client
    assert '"/pets/{petId}"' in client


def test_parameters_consumed(petstore_artifacts: GeneratedArtifacts) -> None:
    client = _file(petstore_artifacts, "demo_project/demo_petstore_client.py").content
    assert 'params["limit"] = limit' in client
    assert '.replace("{petId}", str(pet_id))' in client
    assert "json=payload.model_dump(exclude_none=True)" in client


# ---------------------------------------------------------- Test 3：Authentication


def test_api_key_authentication_affects_generated_code(
    petstore_plan: IntegrationPlan, petstore_artifacts: GeneratedArtifacts
) -> None:
    config = _file(petstore_artifacts, "demo_project/demo_petstore_config.py").content
    assert (
        'BASE_URL = os.environ.get("DEMO_PETSTORE_BASE_URL", "https://petstore.example.com/v1")'
    ) in config
    assert 'API_KEY = os.environ.get("DEMO_PETSTORE_API_KEY")' in config

    client = _file(petstore_artifacts, "demo_project/demo_petstore_client.py").content
    assert 'headers["X-API-Key"] = self._api_key' in client
    assert "config.API_KEY" in client
    assert "DEMO_PETSTORE_API_KEY" in client  # 401 提示带环境变量名
    assert "test-key" not in client  # 不硬编码任何测试凭据


def test_bearer_authentication_affects_generated_code(tmp_path: Path) -> None:
    artifacts = generate_code(_plan(BEARER_API, tmp_path / "repo"))
    client = _file(artifacts, "app/sample_client.py").content
    assert 'headers["Authorization"] = f"Bearer {self._token}"' in client
    config = _file(artifacts, "app/sample_config.py").content
    assert 'TOKEN = os.environ.get("SAMPLE_TOKEN")' in config


# --------------------------------------------------- Test 4：429 retryable 消费


def test_rate_limit_retry_generated(tmp_path: Path) -> None:
    artifacts = generate_code(_plan(RATE_LIMITED_API, tmp_path / "repo"))
    client = _file(artifacts, "app/rate_limited_client.py").content
    assert "def _is_retryable" in client
    assert "if status_code in {429}:" in client
    assert "Retry-After" in client
    assert "time.sleep(self._backoff_delay" in client
    assert "def submit_job" in client
    assert (
        "class RateLimitedRateLimitError"
        in _file(artifacts, "app/rate_limited_exceptions.py").content
    )
    integration = _file(artifacts, "tests/test_rate_limited_integration.py").content
    assert "test_retry_and_backoff" in integration
    assert "pytest.raises" in integration


def test_no_retry_when_not_retryable(petstore_plan: IntegrationPlan) -> None:
    data = petstore_plan.model_dump()
    for rule in data["error_handling"]["rules"]:
        rule["retryable"] = False
    plan = IntegrationPlan.model_validate(data)

    artifacts = generate_code(plan)
    client = _file(artifacts, "demo_project/demo_petstore_client.py").content
    assert "time.sleep" not in client
    assert "_is_retryable" not in client
    assert "Retry-After" not in client
    integration = _file(artifacts, "tests/test_demo_petstore_integration.py").content
    assert "test_retry_and_backoff" not in integration


# ------------------------------------------------------------ Test 5：Dependency


def test_missing_dependency_becomes_add(
    petstore_plan: IntegrationPlan, petstore_artifacts: GeneratedArtifacts
) -> None:
    changes = {item.name: item for item in petstore_artifacts.dependency_changes}
    # demo fixture 已声明 httpx（already_installed），不再产生重复的 DependencyChange
    assert set(changes) == {"pydantic"}
    assert changes["pydantic"].action == "add"
    assert changes["pydantic"].version == ">=2.0"
    assert all(item.reason for item in petstore_artifacts.dependency_changes)
    httpx_declared = next(item for item in petstore_plan.dependencies if item.name == "httpx")
    assert httpx_declared.already_installed is True


def test_installed_dependency_not_duplicated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _make_project(
        root,
        {"app/__init__.py": '"""App."""\n'},
        dependencies=["httpx>=0.27", "pydantic>=2.0"],
    )
    plan = plan_integration(SAMPLE_API, scan_repository(root))
    artifacts = generate_code(plan)
    assert all(item.already_installed for item in plan.dependencies)
    assert artifacts.dependency_changes == []


# --------------------------------------------------------------- Test 6：Modify


def test_files_to_modify_become_structured_modifications(
    petstore_plan: IntegrationPlan, petstore_artifacts: GeneratedArtifacts
) -> None:
    by_path = {item.path: item for item in petstore_artifacts.files if item.action == "modify"}
    for planned in petstore_plan.files_to_modify:
        assert planned.path in by_path
        artifact = by_path[planned.path]
        assert artifact.action == "modify"
        assert artifact.purpose == planned.reason
        assert artifact.changes == planned.changes
        assert artifact.insertion_point
        assert "class DemoPetstoreClient" not in artifact.content  # 不重写整个文件

    manifest = by_path["pyproject.toml"]
    assert manifest.insertion_point == "[project] 的 dependencies 列表内"
    # httpx 已在 fixture 中声明，只剩 pydantic 需要追加
    assert '"pydantic>=2.0",' in manifest.content
    assert '"httpx>=0.27",' not in manifest.content

    init = by_path["demo_project/__init__.py"]
    assert init.insertion_point == "文件末尾（导出区）"
    assert "from .demo_petstore_client import DemoPetstoreClient" in init.content


# ----------------------------------------------------- Test 7：Testing Strategy


def test_testing_strategy_generates_test_artifacts(
    petstore_plan: IntegrationPlan, petstore_artifacts: GeneratedArtifacts
) -> None:
    paths = {item.path for item in petstore_artifacts.files}
    for spec in (
        petstore_plan.testing_strategy.unit_tests
        + petstore_plan.testing_strategy.integration_tests
        + petstore_plan.testing_strategy.contract_tests
    ):
        assert spec.path in paths
        assert _file(petstore_artifacts, spec.path).purpose == spec.purpose

    unit = _file(petstore_artifacts, "tests/test_demo_petstore_client.py").content
    assert "MockTransport" in unit
    assert "test_list_pets" in unit
    assert "test_error_code_to_exception_mapping" in unit
    assert "issubclass" in unit

    integration = _file(petstore_artifacts, "tests/test_demo_petstore_integration.py").content
    assert "test_end_to_end_call" in integration
    assert "test_authentication_injection" in integration
    assert 'request.headers.get("X-API-Key") == "test-key"' in integration

    contract = _file(petstore_artifacts, "tests/test_demo_petstore_contract.py").content
    assert "test_list_pets_response_matches_schema" in contract
    assert "isinstance" in contract


# ------------------------------------------------------------- Test 8：确定性


def test_deterministic_output(
    petstore_plan: IntegrationPlan, petstore_artifacts: GeneratedArtifacts
) -> None:
    generator = DeterministicCodeGenerator()
    assert generator.generate(petstore_plan) == petstore_artifacts
    assert generate_code(petstore_plan) == petstore_artifacts
    assert generator.generate(petstore_plan) == generator.generate(petstore_plan)


# ------------------------------------------------- Test 9：不修改 Repository


def test_generator_does_not_modify_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    plan = _plan(SAMPLE_API, root)
    before = _snapshot(root)
    generate_code(plan)
    generate_code(plan)
    assert _snapshot(root) == before


# ------------------------------------------------------ Test 10：Pydantic 契约


def test_generated_artifacts_contract(petstore_artifacts: GeneratedArtifacts) -> None:
    assert isinstance(petstore_artifacts, GeneratedArtifacts)
    assert all(isinstance(item, GeneratedFile) for item in petstore_artifacts.files)
    assert all(isinstance(item, DependencyChange) for item in petstore_artifacts.dependency_changes)
    assert petstore_artifacts.summary
    assert GeneratedArtifacts.model_validate_json(petstore_artifacts.model_dump_json()) == (
        petstore_artifacts
    )

    created = petstore_artifacts.created_files
    modified = petstore_artifacts.modified_files
    assert created and modified
    assert all(item.action == "create" for item in created)
    assert all(item.action == "modify" for item in modified)
    assert {item.action for item in petstore_artifacts.files} == {"create", "modify"}


# ------------------------------------------------------------- 附加场景与边界


def test_protocol_conformance() -> None:
    assert isinstance(DeterministicCodeGenerator(), CodeGenerator)


def test_generated_python_files_compile(petstore_artifacts: GeneratedArtifacts) -> None:
    """所有生成的 .py 文件必须是语法有效的 Python。"""
    for item in petstore_artifacts.created_files:
        if item.path.endswith(".py"):
            compile(item.content, item.path, "exec")


def test_models_module_contains_referenced_schemas(
    petstore_artifacts: GeneratedArtifacts,
) -> None:
    models = _file(petstore_artifacts, "demo_project/demo_petstore_models.py").content
    assert "class Pet(BaseModel):" in models
    assert "class NewPet(BaseModel):" in models
    assert 'extra="allow"' in models


def test_exceptions_module_covers_all_rules(petstore_artifacts: GeneratedArtifacts) -> None:
    exceptions = _file(petstore_artifacts, "demo_project/demo_petstore_exceptions.py").content
    assert "class DemoPetstoreAPIError(Exception):" in exceptions
    for suffix in (
        "ValidationError",
        "AuthenticationError",
        "AuthorizationError",
        "NotFoundError",
        "RateLimitError",
        "ServerError",
    ):
        # 精确到类声明的左括号，防止 "ServerErrorError" 这类双重后缀被前缀断言漏掉
        assert f"class DemoPetstore{suffix}(" in exceptions
    assert "ServerErrorError" not in exceptions  # 回归：server_error 后缀去重


def test_no_auth_no_config_module(tmp_path: Path) -> None:
    artifacts = generate_code(_plan(SAMPLE_API, tmp_path / "repo"))
    client = _file(artifacts, "app/sample_client.py").content
    assert "os.environ" not in client
    assert 'base_url: str = "https://sample.example.com"' in client
    assert not any(item.path.endswith("_config.py") for item in artifacts.files)


def test_missing_base_url_generates_config_and_warning(tmp_path: Path) -> None:
    artifacts = generate_code(_plan(NO_BASE_URL_API, tmp_path / "repo"))
    config = _file(artifacts, "app/sample_config.py").content
    assert 'BASE_URL = os.environ.get("SAMPLE_BASE_URL")' in config
    client = _file(artifacts, "app/sample_client.py").content
    assert "缺少 base_url" in client
    assert any("SAMPLE_BASE_URL" in warning for warning in artifacts.warnings)


def test_warnings_report_dependencies_and_secrets(
    petstore_artifacts: GeneratedArtifacts,
) -> None:
    assert any("新增依赖需要" in item for item in petstore_artifacts.warnings)
    assert any("环境变量注入" in item for item in petstore_artifacts.warnings)


def test_package_init_generated_for_new_package(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "empty-project"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )
    plan = plan_integration(SAMPLE_API, scan_repository(root))
    artifacts = generate_code(plan)
    init = _file(artifacts, "sample/__init__.py")
    assert init.action == "create"
    assert "from .sample_client import SampleClient" in init.content
    assert _file(artifacts, "sample/sample_client.py").action == "create"


def test_summary_describes_output(petstore_artifacts: GeneratedArtifacts) -> None:
    summary = petstore_artifacts.summary
    assert "Demo Petstore API" in summary
    assert "3 个端点" in summary
    assert "1 项依赖" in summary


def test_advisory_modifications_get_warning(tmp_path: Path) -> None:
    """既有业务/HTTP 模块的修改是建议性片段，必须提示人工确认。"""
    root = tmp_path / "repo"
    _make_project(
        root,
        {
            "app/__init__.py": '"""App."""\n',
            "app/http.py": '"""Shared HTTP helpers."""\n\nimport requests\n',
        },
    )
    plan = plan_integration(SAMPLE_API, scan_repository(root))
    artifacts = generate_code(plan)
    assert any("人工确认" in warning for warning in artifacts.warnings)
    http_mod = _file(artifacts, "app/http.py")
    assert http_mod.action == "modify"
    assert "transport" in http_mod.content
