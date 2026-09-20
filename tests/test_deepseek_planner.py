"""DeepSeek Planner 单元测试。

**本文件不调用真实 DeepSeek API。** 所有用例都通过 FakeLLMClient（或等价的假客户端）
驱动，返回确定性的 JSON 字符串——真实 API 只允许出现在 scripts/ 下的显式 smoke / e2e 脚本。

覆盖用户要求的 8 个场景：
    1. 合法 JSON → 生成合法 IntegrationPlan
    2. Markdown 代码围栏包裹的 JSON → 正常解析
    3. 非法 JSON → DeepSeekPlannerParseError
    4. JSON 合法但模式错误 → DeepSeekPlannerValidationError
    5. 幻觉文件/端点 → 被丢弃并记录 warning（不静默接受）
    6. LLM 报 401 时错误信息不泄漏 API Key
    7. 上下文超限 → prompt 被裁剪，且体积有硬上限
    8. 全部使用 FakeLLMClient，无真实网络调用

以及本实现特有的约束：
    - 事实字段由程序回填，LLM 无法改写（不能"发明项目结构"）
    - 端点必须能在真实 APIInfo 中解析
    - 路径安全（绝对路径 / .. / 不存在 / 已存在）
    - 不静默 fallback：LLM 失败直接抛异常
    - OUTPUT_SKELETON 与 IntegrationPlan 字段不漂移
"""

import json
from pathlib import Path

import pytest

from integration_agent.agent import (
    DeepSeekPlanner,
    DeepSeekPlannerError,
    DeepSeekPlannerParseError,
    DeepSeekPlannerValidationError,
    DeterministicPlanner,
    IntegrationPlan,
    IntegrationPlanner,
    PlannerState,
)
from integration_agent.agent.deepseek_planner import (
    known_repository_files,
    parse_plan_json,
    sanitize,
    validate_relative_path,
)
from integration_agent.agent.prompt import (
    DEFAULT_MAX_PROMPT_CHARS,
    EVIDENCE_PRIORITY,
    OUTPUT_SKELETON,
    SYSTEM_PROMPT,
    build_planner_prompt,
)
from integration_agent.api import parse_openapi_text
from integration_agent.repair import FakeLLMClient
from integration_agent.repository import CodeMatch, ProjectStructure, SearchResult, scan_repository

# ---------------------------------------------------------------- Fixtures

PETSTORE = parse_openapi_text(
    """
openapi: 3.0.3
info:
  title: Demo Petstore API
  version: 1.0.0
  description: 一个用于集成演示的宠物商店 API
servers: [{url: https://petstore.example.com/v1}]
security: [{apiKeyAuth: []}]
paths:
  /pets:
    get:
      operationId: listPets
      tags: [pets]
      summary: 列出所有宠物
      responses:
        "200": {description: ok}
        "429": {description: rate limited}
    post:
      operationId: createPet
      tags: [pets]
      summary: 创建一只宠物
      responses:
        "201": {description: created}
  /pets/{petId}:
    get:
      operationId: getPet
      tags: [pets]
      summary: 获取单只宠物
      responses:
        "200": {description: ok}
        "404": {description: not found}
components:
  securitySchemes:
    apiKeyAuth: {type: apiKey, in: header, name: X-API-Key}
"""
)

# 与 PETSTORE 相同，但端点带 schema 引用（数组 items.$ref / 请求体 $ref / 响应 $ref），
# 用来触发"客户端正文必须引用模型类"的代码路径。
PETSTORE_WITH_SCHEMAS = parse_openapi_text(
    """
openapi: 3.0.3
info:
  title: Demo Petstore API
  version: 1.0.0
servers: [{url: https://petstore.example.com/v1}]
security: [{apiKeyAuth: []}]
paths:
  /pets:
    get:
      operationId: listPets
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                type: array
                items: {$ref: "#/components/schemas/Pet"}
    post:
      operationId: createPet
      requestBody:
        required: true
        content:
          application/json:
            schema: {$ref: "#/components/schemas/NewPet"}
      responses:
        "201":
          description: created
          content:
            application/json:
              schema: {$ref: "#/components/schemas/Pet"}
  /pets/{petId}:
    get:
      operationId: getPet
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema: {$ref: "#/components/schemas/Pet"}
components:
  schemas:
    Pet: {type: object, properties: {id: {type: integer}, name: {type: string}}}
    NewPet: {type: object, properties: {name: {type: string}}}
  securitySchemes:
    apiKeyAuth: {type: apiKey, in: header, name: X-API-Key}
"""
)

NO_AUTH_API = parse_openapi_text(
    """
openapi: 3.0.3
info: {title: Open API, version: "1.0"}
servers: [{url: https://open.example.com}]
paths:
  /things:
    get:
      operationId: listThings
      responses:
        "200": {description: ok}
"""
)

CLIENT_MODULE = '''"""既有的 HTTP 客户端封装（集成时应复用）。"""

import httpx


def build_client() -> httpx.Client:
    return httpx.Client(timeout=10.0)
'''

TEST_MODULE = '''"""既有测试约定（集成时应沿用）。"""


def test_placeholder() -> None:
    assert True
'''


def _make_project(
    root: Path,
    files: dict[str, str] | None = None,
    *,
    name: str = "demo-project",
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


@pytest.fixture
def project(tmp_path: Path) -> ProjectStructure:
    return _make_project(
        tmp_path / "demo_project",
        {
            "demo_project/__init__.py": "",
            "demo_project/client.py": CLIENT_MODULE,
            "tests/test_client.py": TEST_MODULE,
        },
        dependencies=["httpx"],
    )


@pytest.fixture
def state(project: ProjectStructure) -> PlannerState:
    return PlannerState(api=PETSTORE, project=project, request="集成宠物查询与创建接口")


# ---------------------------------------------------------------- 响应构造


def _plan_payload(**overrides: object) -> dict:
    """一份能被校验通过的 LLM 计划载荷；用例只覆盖自己关心的字段。"""
    payload: dict = {
        "integration_goal": "把 Demo Petstore API 集成进 demo-project",
        "target_api": {"name": "whatever", "version": "0", "base_url": None, "total_endpoints": 0},
        "repository": {
            "root": "whatever",
            "name": "whatever",
            "http_client": "httpx",
            "http_client_module": "demo_project/client.py",
            "existing_modules": ["demo_project.client"],
            "inspected_files": ["demo_project/client.py"],
        },
        "integration_strategy": {
            "approach": "extend_existing_client",
            "summary": "在既有 client.py 旁新增 Petstore 客户端",
            "client_module": "demo_project/petstore_client.py",
            "http_client": "httpx",
        },
        "files_to_create": [
            {
                "path": "demo_project/petstore_client.py",
                "purpose": "封装 Petstore 端点",
                "kind": "client",
            },
            {
                "path": "demo_project/petstore_exceptions.py",
                "purpose": "状态码到异常的映射",
                "kind": "exceptions",
            },
        ],
        "files_to_modify": [
            {
                "path": "demo_project/client.py",
                "reason": "复用其 httpx 超时配置",
                "changes": ["导出一个共享的 build_client"],
            }
        ],
        "dependencies": [
            {"name": "httpx", "reason": "HTTP 调用", "already_installed": True},
        ],
        "endpoints": [
            {"method": "GET", "path": "/pets", "purpose": "列出宠物"},
            {"method": "POST", "path": "/pets", "purpose": "创建宠物"},
        ],
        "authentication": {
            "scheme": "apiKey",
            "strategy": "在请求头附加 X-API-Key",
            "config_location": "环境变量",
            "required_env_vars": ["DEMO_PETSTORE_API_KEY"],
            "header_name": "X-API-Key",
        },
        "error_handling": {
            "exception_type": "PetstoreAPIError",
            "rules": [
                {
                    "status_code": "429",
                    "category": "rate_limit",
                    "strategy": "指数退避重试",
                    "retryable": True,
                    "documented_in_spec": True,
                }
            ],
            "notes": [],
        },
        "testing_strategy": {
            "unit_tests": [
                {
                    "path": "tests/test_petstore_client.py",
                    "purpose": "覆盖端点封装",
                    "covers": ["listPets"],
                }
            ],
            "integration_tests": [],
            "contract_tests": [],
        },
        "assumptions": ["假设服务端返回 UTC 时间"],
        "risks": [{"description": "限流", "severity": "medium", "mitigation": "退避重试"}],
        "warnings": [],
    }
    payload.update(overrides)
    return payload


def _dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------- 场景 1：合法 JSON


def test_valid_json_produces_plan(state: PlannerState) -> None:
    """合法 JSON → 完整可用的 IntegrationPlan。"""
    fake = FakeLLMClient(_dumps(_plan_payload()))
    plan = DeepSeekPlanner(fake).plan(state)

    assert isinstance(plan, IntegrationPlan)
    assert plan.integration_goal == "把 Demo Petstore API 集成进 demo-project"
    assert plan.integration_strategy.approach == "extend_existing_client"
    assert [item.path for item in plan.files_to_create] == [
        "demo_project/petstore_client.py",
        "demo_project/petstore_exceptions.py",
    ]
    assert [item.path for item in plan.files_to_modify] == ["demo_project/client.py"]
    assert plan.error_handling.exception_type == "PetstoreAPIError"
    assert plan.authentication is not None
    assert plan.authentication.required_env_vars == ["DEMO_PETSTORE_API_KEY"]
    assert plan.risks[0].severity == "medium"
    assert fake.calls == 1, "一次规划只应调用模型一次"


def test_planner_satisfies_protocol(state: PlannerState) -> None:
    """DeepSeekPlanner 实现 IntegrationPlanner 协议，可与 DeterministicPlanner 互换。"""
    planner = DeepSeekPlanner(FakeLLMClient(_dumps(_plan_payload())))
    assert isinstance(planner, IntegrationPlanner)


def test_llm_is_not_given_environment_access(state: PlannerState) -> None:
    """Planner 只把 prompt 交给 LLMClient，绝不自己读环境变量或碰网络。"""
    fake = FakeLLMClient(_dumps(_plan_payload()))
    DeepSeekPlanner(fake).plan(state)

    assert len(fake.prompts) == 1
    prompt = fake.prompts[0]
    # 环境变量名可以出现在 prompt 里（作为配置约定），但绝不能出现 Key 的取值
    assert "sk-" not in prompt
    assert "Authorization: Bearer" not in prompt


# ---------------------------------------------------------------- 场景 2：代码围栏


def test_json_wrapped_in_markdown_fence(state: PlannerState) -> None:
    """```json 围栏包裹的响应也能解析，但最终仍必须经过 json.loads。"""
    raw = f"```json\n{_dumps(_plan_payload())}\n```"
    plan = DeepSeekPlanner(FakeLLMClient(raw)).plan(state)
    assert plan.integration_goal.startswith("把 Demo Petstore API")


def test_json_with_surrounding_prose(state: PlannerState) -> None:
    """模型在 JSON 前后多写一句话时，只取最外层对象。"""
    raw = f"Here is the plan:\n{_dumps(_plan_payload())}\nHope this helps!"
    plan = DeepSeekPlanner(FakeLLMClient(raw)).plan(state)
    assert isinstance(plan, IntegrationPlan)


# ---------------------------------------------------------------- 场景 3：非法 JSON


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \n  ",
        "not json at all",
        '{"integration_goal": "未闭合的字符串',
        "[1, 2, 3]",
    ],
)
def test_invalid_json_raises_parse_error(state: PlannerState, raw: str) -> None:
    """非法 JSON / 空响应 / 顶层非对象 → DeepSeekPlannerParseError。"""
    with pytest.raises(DeepSeekPlannerParseError):
        DeepSeekPlanner(FakeLLMClient(raw)).plan(state)


def test_parse_plan_json_rejects_non_object() -> None:
    with pytest.raises(DeepSeekPlannerParseError):
        parse_plan_json('["a", "b"]')


def test_fenced_non_json_still_fails(state: PlannerState) -> None:
    """围栏里不是 JSON 时不能被"清理"成合法结果。"""
    with pytest.raises(DeepSeekPlannerParseError):
        DeepSeekPlanner(FakeLLMClient("```json\n{broken\n```")).plan(state)


# ---------------------------------------------------------------- 场景 4：模式错误


def test_schema_violation_raises_validation_error(state: PlannerState) -> None:
    """JSON 合法但不满足 IntegrationPlan 约束 → DeepSeekPlannerValidationError。"""
    payload = _plan_payload()
    payload["integration_strategy"]["approach"] = "rewrite_everything"
    with pytest.raises(DeepSeekPlannerValidationError) as excinfo:
        DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)
    # 错误信息要能定位到具体字段，便于排查
    assert "integration_strategy.approach" in str(excinfo.value)


def test_missing_required_section_raises(state: PlannerState) -> None:
    """缺少核心判断（integration_strategy）→ 校验失败，而不是被默默补默认值。"""
    payload = _plan_payload()
    del payload["integration_strategy"]
    with pytest.raises(DeepSeekPlannerValidationError):
        DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)


def test_validation_error_does_not_echo_input_values(state: PlannerState) -> None:
    """校验错误只报告字段与原因，不回显模型给的原始值。"""
    payload = _plan_payload()
    payload["error_handling"]["exception_type"] = 12345  # 类型错误
    with pytest.raises(DeepSeekPlannerValidationError) as excinfo:
        DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)
    assert "error_handling.exception_type" in str(excinfo.value)
    assert "12345" not in str(excinfo.value)


# ---------------------------------------------------------------- 场景 5：幻觉


def test_hallucinated_files_are_dropped_with_warnings(state: PlannerState) -> None:
    """模型编造的文件既不能进入计划，也不能被悄悄接受。"""
    payload = _plan_payload()
    payload["files_to_modify"] = [
        {"path": "demo_project/ghost.py", "reason": "凭记忆编造", "changes": []},
        {"path": "demo_project/client.py", "reason": "真实存在", "changes": ["ok"]},
    ]
    payload["files_to_create"] = [
        {"path": "pyproject.toml", "purpose": "已存在，不能算新建", "kind": "other"},
        {"path": "demo_project/new_client.py", "purpose": "合法新文件", "kind": "client"},
        {"path": "demo_project/new_exceptions.py", "purpose": "异常类落点", "kind": "exceptions"},
    ]
    payload["repository"] = {
        "root": "whatever",
        "name": "whatever",
        "http_client": "httpx",
        "http_client_module": "demo_project/ghost_http.py",
        "existing_modules": ["demo_project.client", "demo_project.ghost"],
        "inspected_files": ["demo_project/client.py", "demo_project/never_read.py"],
    }
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.path for item in plan.files_to_modify] == ["demo_project/client.py"]
    assert [item.path for item in plan.files_to_create] == [
        "demo_project/new_client.py",
        "demo_project/new_exceptions.py",
    ]
    assert plan.repository.http_client_module is None
    assert plan.repository.existing_modules == ["demo_project.client"]
    assert plan.repository.inspected_files == ["demo_project/client.py"]

    joined = "\n".join(plan.warnings)
    assert "demo_project/ghost.py" in joined
    assert "pyproject.toml" in joined
    assert "demo_project/ghost_http.py" in joined
    assert "demo_project.ghost" in joined
    assert "demo_project/never_read.py" in joined


def test_real_repo_files_are_not_flagged_as_hallucinations(state: PlannerState) -> None:
    """回归：引用真实存在但本轮检索没命中的文件，不是幻觉。

    曾经这里只认"检索命中过的文件"，于是 pyproject.toml 这类真实文件被误报为
    编造，告警本身就是错的。
    """
    payload = _plan_payload()
    payload["repository"]["inspected_files"] = ["pyproject.toml", "demo_project/client.py"]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert plan.repository.inspected_files == ["pyproject.toml", "demo_project/client.py"]
    assert not any("pyproject.toml" in warning for warning in plan.warnings)
    assert not any("client.py" in warning for warning in plan.warnings)


def test_deepseek_planner_accepts_manifest_file_in_files_to_modify(state: PlannerState) -> None:
    """回归：pyproject.toml 在仓库中真实存在，不能被当成"不存在的文件"丢掉。"""
    payload = _plan_payload()
    payload["files_to_modify"] = [
        {"path": "pyproject.toml", "reason": "补充 pytest 依赖", "changes": ["加 pytest"]},
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.path for item in plan.files_to_modify] == ["pyproject.toml"]


def test_test_file_listed_in_both_places_is_deduplicated(state: PlannerState) -> None:
    """回归：测试文件同时出现在 files_to_create 与 testing_strategy 会被生成两次。

    Code Generator 对两个列表分别生成"完整文件"和"测试骨架"，同一路径出现两次
    会产出内容冲突的两个版本，直接导致 pytest collection error。
    """
    payload = _plan_payload()
    payload["files_to_create"] = [
        {"path": "demo_project/petstore_client.py", "purpose": "客户端", "kind": "client"},
        {
            "path": "tests/test_petstore_client.py",
            "purpose": "又在 testing_strategy 里声明了一次",
            "kind": "other",
        },
        {"path": "demo_project/petstore_exceptions.py", "purpose": "异常类", "kind": "exceptions"},
    ]
    payload["testing_strategy"]["unit_tests"] = [
        {"path": "tests/test_petstore_client.py", "purpose": "单元测试", "covers": ["listPets"]}
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.path for item in plan.files_to_create] == [
        "demo_project/petstore_client.py",
        "demo_project/petstore_exceptions.py",
    ]
    assert plan.testing_strategy.unit_tests[0].path == "tests/test_petstore_client.py"
    assert any("testing_strategy" in warning for warning in plan.warnings)


def test_non_test_files_are_untouched_by_dedup(state: PlannerState) -> None:
    """去重只针对与 testing_strategy 重合的路径，不影响其他文件。"""
    payload = _plan_payload()
    payload["files_to_create"] = [
        {"path": "demo_project/petstore_client.py", "purpose": "客户端", "kind": "client"},
        {"path": "demo_project/petstore_exceptions.py", "purpose": "异常", "kind": "exceptions"},
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.path for item in plan.files_to_create] == [
        "demo_project/petstore_client.py",
        "demo_project/petstore_exceptions.py",
    ]
    assert not any("testing_strategy" in warning for warning in plan.warnings)


def test_missing_exceptions_module_is_added(state: PlannerState) -> None:
    """回归：只声明客户端、没有异常模块的计划会生成 ``from None import ...``。

    Code Generator 只在 kind="exceptions" 的文件里定义异常类，客户端与测试都从
    那里 import；缺了它，生成的测试文件是语法错误（实测：
    ``from None import PetstoreRateLimitError`` → pytest collection error）。
    LLM 常把异常类顺带写进客户端文件，这里必须补回独立模块。
    """
    payload = _plan_payload()
    payload["files_to_create"] = [
        {"path": "demo_project/petstore_client.py", "purpose": "客户端", "kind": "client"}
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.path for item in plan.files_to_create] == [
        "demo_project/petstore_client.py",
        "demo_project/petstore_exceptions.py",
    ]
    added = plan.files_to_create[-1]
    assert added.kind == "exceptions"
    assert "PetstoreAPIError" in added.purpose
    assert any("exceptions" in warning for warning in plan.warnings)


def test_declared_exceptions_module_is_left_alone(state: PlannerState) -> None:
    """模型自己给了异常模块时不得重复添加，也不应产生 warning。"""
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(_plan_payload()))).plan(state)

    kinds = [item.kind for item in plan.files_to_create]
    assert kinds.count("exceptions") == 1
    assert not any("kind='exceptions'" in warning for warning in plan.warnings)


def test_plan_without_client_module_is_not_given_exceptions(state: PlannerState) -> None:
    """没有客户端模块时不做无中生有的补充。"""
    payload = _plan_payload()
    payload["files_to_create"] = [
        {"path": "demo_project/models.py", "purpose": "数据模型", "kind": "models"}
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.path for item in plan.files_to_create] == ["demo_project/models.py"]


def test_existing_module_given_as_file_path_is_accepted(state: PlannerState) -> None:
    """回归：模型把 existing_modules 写成文件路径也是有效引用。

    原实现只把点号模块名换算成 pkg/mod，文件路径写法会变成 demo_project/client/py，
    于是真实存在的模块被误报成"找不到对应文件"。
    """
    payload = _plan_payload()
    payload["repository"] = {
        "root": "whatever",
        "name": "whatever",
        "http_client": "httpx",
        "http_client_module": "demo_project/client.py",
        "existing_modules": ["demo_project.client", "demo_project/client.py"],
        "inspected_files": ["demo_project/client.py"],
    }
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert plan.repository.existing_modules == ["demo_project.client", "demo_project/client.py"]
    assert not any("可复用模块" in warning for warning in plan.warnings)


def test_missing_models_module_is_added_when_endpoints_reference_schemas(
    state: PlannerState,
) -> None:
    """回归：端点引用了 schema，客户端就会生成 ``Pet.model_validate(...)``。

    但模型的 import 只在存在 kind="models" 文件时才写。缺了模型模块，客户端照样
    生成、照样能编译，直到被调用才 NameError（实测：集成测试 11 failed，
    ``NameError: name 'Pet' is not defined``）。
    """
    payload = _plan_payload()
    payload["files_to_create"] = [
        {"path": "demo_project/petstore_client.py", "purpose": "客户端", "kind": "client"},
        {"path": "demo_project/petstore_exceptions.py", "purpose": "异常", "kind": "exceptions"},
    ]
    state = PlannerState(
        api=PETSTORE_WITH_SCHEMAS, project=state.project, request="集成宠物查询与创建接口"
    )
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert plan.endpoints, "本用例依赖端点带 schema 引用"
    assert [item.path for item in plan.files_to_create] == [
        "demo_project/petstore_client.py",
        "demo_project/petstore_exceptions.py",
        "demo_project/petstore_models.py",
    ]
    assert plan.files_to_create[-1].kind == "models"
    assert any("models" in warning for warning in plan.warnings)


def test_plan_without_schema_references_needs_no_models_module(state: PlannerState) -> None:
    """端点不引用任何 schema 时不得无中生有地补模型模块。"""
    payload = _plan_payload()
    payload["endpoints"] = [{"method": "GET", "path": "/pets", "purpose": "列出"}]
    payload["files_to_create"] = [
        {"path": "demo_project/petstore_client.py", "purpose": "客户端", "kind": "client"},
        {"path": "demo_project/petstore_exceptions.py", "purpose": "异常", "kind": "exceptions"},
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.kind for item in plan.files_to_create] == ["client", "exceptions"]


def test_hallucinated_endpoint_is_dropped(state: PlannerState) -> None:
    """端点必须能在真实 APIInfo 中解析，编造的端点被丢弃。"""
    payload = _plan_payload()
    payload["endpoints"] = [
        {"method": "GET", "path": "/pets", "purpose": "真实"},
        {"method": "DELETE", "path": "/pets/{petId}", "purpose": "文档里没有 DELETE"},
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.operation_id for item in plan.endpoints] == ["listPets"]
    assert any("DELETE /pets/{petId}" in warning for warning in plan.warnings)


def test_endpoint_schema_comes_from_spec_not_model(state: PlannerState) -> None:
    """模型只给标识；参数/响应 schema 一律从 API 文档取，杜绝凭记忆编造。"""
    payload = _plan_payload()
    payload["endpoints"] = [{"method": "get", "path": "/pets", "purpose": "小写 method 也应识别"}]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    endpoint = plan.endpoints[0].endpoint
    assert endpoint.operation_id == "listPets"
    assert endpoint.summary == "列出所有宠物"
    assert {item.status_code for item in endpoint.responses} == {"200", "429"}


def test_endpoints_are_capped(state: PlannerState) -> None:
    """选择过多端点时按上限截断并告警。"""
    payload = _plan_payload()
    payload["endpoints"] = [
        {"method": "GET", "path": "/pets", "purpose": "a"},
        {"method": "POST", "path": "/pets", "purpose": "b"},
        {"method": "GET", "path": "/pets/{petId}", "purpose": "c"},
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload)), max_endpoints=2).plan(state)

    assert len(plan.endpoints) == 2
    assert any("超过上限" in warning for warning in plan.warnings)


# ---------------------------------------------------------------- 事实回填


def test_facts_are_grounded_against_state(state: PlannerState) -> None:
    """事实字段由程序回填：模型声称的一切都无法改变 API / 仓库的真实信息。"""
    payload = _plan_payload()
    payload["target_api"] = {
        "name": "Imaginary API",
        "version": "9.9.9",
        "base_url": "https://evil.example.com",
        "total_endpoints": 999,
    }
    payload["repository"]["root"] = "/tmp/imaginary"
    payload["repository"]["name"] = "not-the-real-project"
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert plan.target_api.name == "Demo Petstore API"
    assert plan.target_api.version == "1.0.0"
    assert plan.target_api.base_url == "https://petstore.example.com/v1"
    assert plan.target_api.total_endpoints == 3
    assert plan.repository.root == state.project.root
    assert plan.repository.name == "demo-project"


def test_missing_repository_section_is_rebuilt_and_warned(state: PlannerState) -> None:
    """模型完全省略 repository 时按事实重建，并明确告知信息有缺失。"""
    payload = _plan_payload()
    del payload["repository"]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert plan.repository.name == "demo-project"
    assert plan.repository.http_client is None
    assert any("repository" in warning for warning in plan.warnings)


# ---------------------------------------------------------------- 路径安全


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "C:\\Windows\\system32\\evil.py", "C:relative.py", "../../secrets.py", ""],
)
def test_unsafe_paths_are_rejected(path: str) -> None:
    assert validate_relative_path(path) is not None


@pytest.mark.parametrize("path", ["demo_project/client.py", "src/a/b.py", "./src/a.py"])
def test_safe_paths_are_accepted(path: str) -> None:
    assert validate_relative_path(path) is None


def test_unsafe_path_in_plan_is_dropped(state: PlannerState) -> None:
    """越界路径即使模型写了也不能落地。"""
    payload = _plan_payload()
    payload["files_to_create"] = [
        {"path": "../../outside.py", "purpose": "越界", "kind": "other"},
        {"path": "demo_project/ok.py", "purpose": "合法", "kind": "client"},
        {"path": "demo_project/ok_exceptions.py", "purpose": "异常类落点", "kind": "exceptions"},
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert [item.path for item in plan.files_to_create] == [
        "demo_project/ok.py",
        "demo_project/ok_exceptions.py",
    ]
    assert any("../../outside.py" in warning for warning in plan.warnings)


# ---------------------------------------------------------------- 凭据安全


def test_invalid_env_var_names_are_stripped(state: PlannerState) -> None:
    """疑似硬编码凭据不能混进 required_env_vars。"""
    payload = _plan_payload()
    payload["authentication"]["required_env_vars"] = [
        "DEMO_PETSTORE_API_KEY",
        "sk-live-abcdef1234567890",
        "lowercase-name",
    ]
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert plan.authentication is not None
    assert plan.authentication.required_env_vars == ["DEMO_PETSTORE_API_KEY"]
    assert any("required_env_vars" in warning for warning in plan.warnings)


def test_literal_secret_in_plan_text_is_flagged(state: PlannerState) -> None:
    """计划正文里出现密钥字面量要告警，而不是当成正常描述。"""
    payload = _plan_payload()
    payload["integration_goal"] = "使用 sk-live-abcdef1234567890 直接调用"
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert any("疑似密钥字面量" in warning for warning in plan.warnings)


def test_missing_required_env_vars_is_flagged(state: PlannerState) -> None:
    """声明了认证却不给出环境变量来源 → 有硬编码风险。"""
    payload = _plan_payload()
    payload["authentication"]["required_env_vars"] = []
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert any("required_env_vars" in warning for warning in plan.warnings)


def test_auth_declared_in_spec_but_absent_in_plan_is_flagged(state: PlannerState) -> None:
    """API 声明了安全方案，计划却没有认证方案 → 告警。"""
    payload = _plan_payload()
    payload["authentication"] = None
    plan = DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)

    assert any("安全方案" in warning for warning in plan.warnings)


# ---------------------------------------------------------------- 场景 6：不泄漏 Key


class _FailingLLM:
    """模拟 DeepSeek 返回 401 的客户端；错误信息里带着 Key（最坏情况）。"""

    def __init__(self, message: str) -> None:
        self.message = message
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        raise RuntimeError(self.message)


def test_api_key_is_not_leaked_on_401(state: PlannerState) -> None:
    """401 invalid api key：异常信息必须脱敏，且不包含完整 prompt。"""
    secret = "sk-live-abcdef1234567890abcdef"
    llm = _FailingLLM(f"DeepSeek API 返回 HTTP 401: invalid api key ({secret})")

    with pytest.raises(DeepSeekPlannerError) as excinfo:
        DeepSeekPlanner(llm).plan(state)

    message = str(excinfo.value)
    assert secret not in message
    assert "sk-live" not in message
    assert "***" in message
    # 完整 prompt 不应出现在异常里（其中含仓库代码片段）
    assert "REPOSITORY EVIDENCE" not in message


def test_bearer_token_is_not_leaked(state: PlannerState) -> None:
    llm = _FailingLLM("HTTP 401 Authorization: Bearer abcdef1234567890abcdefghij")
    with pytest.raises(DeepSeekPlannerError) as excinfo:
        DeepSeekPlanner(llm).plan(state)

    message = str(excinfo.value)
    assert "abcdef1234567890abcdefghij" not in message
    assert "***" in message


def test_exception_chain_does_not_reintroduce_the_secret(state: PlannerState) -> None:
    """异常链本身不能成为泄漏通道。

    Pipeline 会把 traceback.format_exc() 写进 PipelineResult 返回给前端。
    如果这里用 `raise ... from exc`，底层那条**未脱敏**的原始异常会随 traceback
    一起泄漏出去——sanitize 就成了摆设。本用例锁定 `from None` 这一行为。
    """
    import traceback

    secret = "sk-live-abcdef1234567890abcdef"
    llm = _FailingLLM(f"DeepSeek API 返回 HTTP 401: invalid api key ({secret})")

    try:
        DeepSeekPlanner(llm).plan(state)
    except DeepSeekPlannerError:
        rendered = traceback.format_exc()

    assert secret not in rendered
    assert "***" in rendered


def test_validation_traceback_does_not_dump_raw_input(state: PlannerState) -> None:
    """Pydantic ValidationError 的 traceback 会打印 input_value，必须被掐断。"""
    import traceback

    payload = _plan_payload()
    payload["integration_strategy"]["approach"] = "leaky-value-should-not-appear"
    try:
        DeepSeekPlanner(FakeLLMClient(_dumps(payload))).plan(state)
    except DeepSeekPlannerValidationError:
        rendered = traceback.format_exc()

    assert "leaky-value-should-not-appear" not in rendered


def test_sanitize_flattens_and_truncates() -> None:
    """脱敏函数：抹凭据 + 压平换行 + 限长，避免异常信息失控。"""
    cleaned = sanitize("一行\n二行 sk-abcdef123456 x" * 100)
    assert "\n" not in cleaned
    assert "sk-abcdef123456" not in cleaned
    assert len(cleaned) <= 300


# ---------------------------------------------------------------- 不静默 fallback


def test_no_silent_fallback_on_llm_failure(state: PlannerState) -> None:
    """LLM 失败必须抛异常，绝不能悄悄退回确定性计划。"""
    with pytest.raises(DeepSeekPlannerError):
        DeepSeekPlanner(FakeLLMClient("")).plan(state)


def test_no_silent_fallback_on_empty_response(state: PlannerState) -> None:
    """空响应同样不能被当成"没有计划就是计划"。"""
    with pytest.raises(DeepSeekPlannerError):
        DeepSeekPlanner(FakeLLMClient([])).plan(state)


# ---------------------------------------------------------------- 场景 7：上下文限制


def _state_with_evidence(project: ProjectStructure, entries: list[tuple[str, str]]) -> PlannerState:
    """构造带证据的状态；entries 为 (文件路径, 片段) 列表。"""
    state = PlannerState(api=PETSTORE, project=project, request="集成宠物接口")
    for index, (path, snippet) in enumerate(entries):
        state.record(
            SearchResult(
                query=f"query-{index}",
                matches=[CodeMatch(path=path, line_number=index + 1, snippet=snippet)],
                files_scanned=1,
            )
        )
    return state


def test_prompt_stays_within_budget_with_huge_evidence(project: ProjectStructure) -> None:
    """证据再多，prompt 体积也有硬上限。"""
    entries = [(f"demo_project/module_{index}.py", "x = 1\n" * 2000) for index in range(60)]
    state = _state_with_evidence(project, entries)

    prompt = build_planner_prompt(state)
    assert len(prompt) <= DEFAULT_MAX_PROMPT_CHARS


def test_prompt_stays_within_budget_with_huge_request(project: ProjectStructure) -> None:
    """超长 request 不能突破上限；且被压缩的是需求原文，不是输出契约。"""
    state = PlannerState(api=PETSTORE, project=project, request="请集成 " * 20_000)
    prompt = build_planner_prompt(state)

    assert len(prompt) <= DEFAULT_MAX_PROMPT_CHARS
    assert "truncated" in prompt
    # 关键：OUTPUT FORMAT / PLANNING RULES 必须活下来——模型不知道输出结构，
    # 再完整的上下文也换不回一个能解析的计划
    assert "OUTPUT FORMAT" in prompt
    assert "PLANNING RULES" in prompt


def test_evidence_is_trimmed_by_priority(project: ProjectStructure) -> None:
    """证据超限时优先保留 HTTP 客户端 / 认证类文件，先丢弃无关文件。"""
    entries = []
    for index in range(40):
        entries.append((f"demo_project/noise_{index}.py", "y = 1\n" * 400))
    entries.append(("demo_project/http_client.py", "z = 1\n" * 400))
    entries.append(("demo_project/auth_token.py", "w = 1\n" * 400))
    state = _state_with_evidence(project, entries)

    prompt = build_planner_prompt(state)
    assert len(prompt) <= DEFAULT_MAX_PROMPT_CHARS
    assert "demo_project/http_client.py" in prompt
    assert "demo_project/auth_token.py" in prompt


def test_planner_prompt_includes_all_required_sections(state: PlannerState) -> None:
    """prompt 必须包含五类必要上下文：请求 / API / 结构 / 约束 / 证据。"""
    prompt = build_planner_prompt(state)

    for marker in (
        "INTEGRATION REQUEST",
        "API INFORMATION",
        "PROJECT STRUCTURE",
        "PROJECT CONSTRAINTS",
        "REPOSITORY EVIDENCE",
        "PLANNING RULES",
        "OUTPUT FORMAT",
    ):
        assert marker in prompt
    assert state.request in prompt
    assert "Demo Petstore API" in prompt
    assert "listPets" in prompt


def test_prompt_never_dumps_whole_repository(state: PlannerState) -> None:
    """prompt 里不能出现"整个仓库源码"——只能出现受控的检索片段。"""
    state.request = "集成"
    prompt = build_planner_prompt(state)
    # 证据只应包含检索命中的少量文件，而不是全部 python_files
    assert len(prompt) < DEFAULT_MAX_PROMPT_CHARS
    assert "python_file_count" in prompt  # 只给数量，不给全文


def test_evidence_priority_table_is_ordered() -> None:
    """优先级表顺序即保留顺序：HTTP 客户端 > 认证 > 业务 > 模型 > 测试 > 配置。"""
    labels = [name for name, _ in EVIDENCE_PRIORITY]
    assert labels == [
        "http_client",
        "authentication",
        "service",
        "model",
        "test",
        "config",
    ]


# ---------------------------------------------------------------- 骨架不漂移


def test_output_skeleton_matches_integration_plan_fields() -> None:
    """prompt 里的输出骨架必须与 IntegrationPlan 字段完全一致，防止两边漂移。"""
    assert set(OUTPUT_SKELETON) == set(IntegrationPlan.model_fields)


@pytest.mark.parametrize(
    ("skeleton_key", "model_name"),
    [
        ("target_api", "TargetAPI"),
        ("repository", "RepositoryContext"),
        ("integration_strategy", "IntegrationStrategy"),
        ("authentication", "AuthenticationPlan"),
        ("error_handling", "ErrorHandlingPlan"),
        ("testing_strategy", "TestingStrategy"),
    ],
)
def test_output_skeleton_nested_keys_match_models(skeleton_key: str, model_name: str) -> None:
    """嵌套骨架的键也要与对应模型字段一致。"""
    import integration_agent.agent.models as models

    model = getattr(models, model_name)
    assert set(OUTPUT_SKELETON[skeleton_key]) == set(model.model_fields)


def test_system_prompt_states_core_rules() -> None:
    """System Prompt 必须明确"不是代码生成器"与"证据优先于假设"。"""
    assert "NOT a code generator" in SYSTEM_PROMPT
    assert "Evidence has priority over assumptions" in SYSTEM_PROMPT
    assert "Do not output Markdown" in SYSTEM_PROMPT


def test_prompt_contains_lowercase_json_keyword(state: PlannerState) -> None:
    """DeepSeek 的 response_format=json_object 要求 prompt 中出现小写 "json"。

    这是真实的 API 契约，不是风格问题——少了它结构化输出会被服务端拒绝。
    """
    assert "json" in build_planner_prompt(state)


def test_prompt_exposes_schema_names(project: ProjectStructure) -> None:
    """schema 名必须出现在 prompt 里，否则模型的判断会与生成器的行为矛盾。

    Code Generator 会把响应体校验成模型类（``Pet.model_validate(...)``）。若 prompt
    隐去 schema，模型会得出"没有 schema，不需要 models 模块"的结论并写进 warnings，
    而生成器照旧生成模型引用——计划与代码互相矛盾。
    """
    state = PlannerState(
        api=PETSTORE_WITH_SCHEMAS, project=project, request="集成宠物查询与创建接口"
    )
    prompt = build_planner_prompt(state)

    api_section = prompt.split("PROJECT STRUCTURE")[0]
    assert '"Pet"' in api_section
    assert '"NewPet"' in api_section, "请求体 schema 也要给出，否则模型不知道要建模型模块"
    assert '"models"' in prompt, "必须告诉模型生成器要求 kind=models 文件"


# ---------------------------------------------------------------- 证据收集


def test_provided_evidence_is_not_recollected(project: ProjectStructure) -> None:
    """调用方已提供证据时不重复检索（尊重上游 Retrieval）。"""
    state = _state_with_evidence(project, [("demo_project/client.py", "import httpx")])
    before = len(state.evidence)

    DeepSeekPlanner(FakeLLMClient(_dumps(_plan_payload()))).plan(state)
    assert len(state.evidence) == before


def test_evidence_is_collected_when_absent(project: ProjectStructure) -> None:
    """没有证据时做少量受限检索补充，而不是把仓库整体交给模型。"""
    state = PlannerState(api=PETSTORE, project=project, request="集成宠物接口")
    assert state.evidence == []

    DeepSeekPlanner(FakeLLMClient(_dumps(_plan_payload()))).plan(state)

    assert state.evidence, "应当补充检索证据"
    assert len(state.evidence) <= 6
    # 检索命中的确实是仓库里真实存在的文件
    assert all(path in known_repository_files(state) for path in state.evidence[0].paths)


def test_fallback_query_supplies_evidence_when_all_queries_miss(tmp_path: Path) -> None:
    """回归：定向检索全部落空时用一条宽泛查询兜底，模型不至于完全空手规划。

    占位模块（只有 docstring + def，没有任何 import）正是这种情形。
    """
    project = _make_project(
        tmp_path / "placeholder",
        {
            "placeholder/__init__.py": "",
            "placeholder/service.py": (
                '"""占位服务模块。"""\n\n\ndef get_profile(user_id: str) -> dict:\n'
                '    """返回占位数据。"""\n    return {"id": user_id}\n'
            ),
        },
    )
    state = PlannerState(api=PETSTORE, project=project, request="集成宠物接口")
    assert state.evidence == []

    DeepSeekPlanner(FakeLLMClient(_dumps(_plan_payload()))).plan(state)

    assert any(evidence.matches for evidence in state.evidence), "兜底查询应当命中真实代码"
    assert "placeholder/service.py" in known_repository_files(state)


def test_known_repository_files_covers_scan_and_evidence(project: ProjectStructure) -> None:
    state = _state_with_evidence(project, [("demo_project/client.py", "import httpx")])
    known = known_repository_files(state)

    assert "pyproject.toml" in known
    assert "demo_project/client.py" in known
    assert "demo_project/ghost.py" not in known


# ---------------------------------------------------------------- 兼容性


def test_deterministic_planner_has_empty_warnings(project: ProjectStructure) -> None:
    """新增的 warnings 字段不影响既有确定性 Planner。"""
    plan = DeterministicPlanner().plan(PlannerState(api=PETSTORE, project=project))
    assert plan.warnings == []


def test_deepseek_planner_is_not_used_by_default(state: PlannerState) -> None:
    """默认仍是确定性 Planner：DeepSeekPlanner 必须显式注入。"""
    default_plan = DeterministicPlanner().plan(state)
    llm_plan = DeepSeekPlanner(FakeLLMClient(_dumps(_plan_payload()))).plan(state)

    assert default_plan.integration_goal != llm_plan.integration_goal


def test_planner_rejects_non_llm_objects() -> None:
    """结构化协议校验：没有 generate() 的对象不能当 LLM 用。"""
    for bad in (None, object(), "just a string", 42):
        with pytest.raises(DeepSeekPlannerError):
            DeepSeekPlanner(bad)


def test_planner_accepts_any_generate_object(state: PlannerState) -> None:
    """任何实现 generate(prompt) -> str 的对象都可以注入（不依赖具体类型）。"""

    class Custom:
        def generate(self, prompt: str) -> str:
            return _dumps(_plan_payload())

    plan = DeepSeekPlanner(Custom()).plan(state)
    assert isinstance(plan, IntegrationPlan)


def test_same_input_and_output_is_deterministic(state: PlannerState) -> None:
    """同一状态 + 同一 LLM 输出 → 完全相同的计划（可重放）。"""
    raw = _dumps(_plan_payload())
    first = DeepSeekPlanner(FakeLLMClient(raw)).plan(state)
    second = DeepSeekPlanner(FakeLLMClient(raw)).plan(state)

    assert first.model_dump() == second.model_dump()
