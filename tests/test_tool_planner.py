"""ToolUsingPlanner 的测试（E2 Commit 5）。

**本文件不调用真实 DeepSeek，也不发任何网络请求。** Agent 回路由
FakeToolCallingClient 驱动，工具用真实注册表（四个只读工具）或记录调用的 spy。

覆盖用户要求的 20 个场景：
    1.  实现 IntegrationPlanner 协议
    2.  初始状态正确进入 AgentLoop（prompt / ToolContext / 工具 schema）
    3.  可以调用 inspect_project
    4.  可以调用 search_code
    5.  可以调用 read_file
    6.  可以调用 inspect_api
    7.  最终 final JSON 正确解析为 IntegrationPlan
    8.  非法 final JSON → 报错而不是半个计划
    9.  endpoint 幻觉被过滤
    10. 文件幻觉被过滤
    11. 事实字段由 PlannerState 回填（LLM 改写无效）
    12. validate_plan 生效
    13. enforce_constraints 生效
    14. warnings 正确记录
    15. LLM 不执行工具
    16. planner 不写真实 Repository
    17. 不重新解析 OpenAPI
    18. 不重新扫描 Repository
    19. 用户 request 被传进模型
    20. 工具调用预算生效
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from integration_agent.agent import tool_planner as tool_planner_module
from integration_agent.agent.agent_loop import AgentLoopConfig, AgentLoopRunner, AgentRunResult
from integration_agent.agent.llm import (
    AssistantTurn,
    ChatMessage,
    FakeToolCallingClient,
    ToolCallRequest,
)
from integration_agent.agent.models import IntegrationPlan
from integration_agent.agent.planner import IntegrationPlanner, PlanningError
from integration_agent.agent.state import PlannerState
from integration_agent.agent.tool_planner import ToolUsingPlanner, parse_final_plan
from integration_agent.api import parse_openapi_text
from integration_agent.repository import scan_repository
from integration_agent.tools import ToolContext, ToolRegistry, ToolResult, ToolSpec

PETSTORE = parse_openapi_text(
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
      tags: [pets]
      summary: 列出所有宠物
      responses:
        "200": {description: ok}
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
components:
  securitySchemes:
    apiKeyAuth: {type: apiKey, in: header, name: X-API-Key}
"""
)

REQUEST = "把宠物接口集成进示例项目"

# ToolUsingPlanner 允许 import 的东西：它不该有任何"重新解析/重新扫描"的入口
ALLOWED_PLANNER_IMPORTS = {
    "logging",
    "pathlib",
    "typing",
    "integration_agent.agent.agent_loop",
    "integration_agent.agent.deepseek_planner",
    "integration_agent.agent.llm",
    "integration_agent.agent.models",
    "integration_agent.agent.planner",
    "integration_agent.agent.prompt",
    "integration_agent.agent.state",
    "integration_agent.agent.tool_prompt",
    "integration_agent.tools",
    "integration_agent.tools.registry",
}
FORBIDDEN_PLANNER_IMPORTS = {
    "integration_agent.api.parser",
    "integration_agent.repository.scanner",
    "subprocess",
    "socket",
    "urllib",
    "urllib.request",
}


# ------------------------------------------------------------------ helpers


def _project_tree(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sample-project"\nversion = "0.1.0"\ndependencies = ["httpx>=0.27"]\n',
        encoding="utf-8",
    )
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "client.py").write_text("def fetch_pets():\n    return []\n", encoding="utf-8")


@pytest.fixture()
def state(tmp_path: Path) -> PlannerState:
    _project_tree(tmp_path)
    return PlannerState(api=PETSTORE, project=scan_repository(tmp_path), request=REQUEST)


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return snapshot


def _payload(**overrides) -> dict:
    """一份完整、合法、无幻觉的 IntegrationPlan 载荷。"""
    payload: dict = {
        "integration_goal": "把 Demo Petstore API 集成进 sample-project",
        # 事实字段刻意写错：ground_facts 必须把它们全部覆盖掉
        "target_api": {
            "name": "Wrong API",
            "version": "0.0.0",
            "base_url": "https://wrong.example.com",
            "total_endpoints": 999,
        },
        "repository": {
            "root": "/tmp/wrong-root",
            "name": "wrong-project",
            "source_dirs": ["wrong_src"],
            "test_dirs": ["wrong_tests"],
            "http_client": "httpx",
            "http_client_module": "pkg/client.py",
            "existing_modules": [],
            "inspected_files": [],
        },
        "integration_strategy": {
            "approach": "new_client_module",
            "summary": "新建独立客户端模块封装 petstore 调用",
            "client_module": "sample/petstore_client.py",
            "http_client": "httpx",
        },
        "files_to_create": [
            {
                "path": "sample/petstore_client.py",
                "purpose": "封装 HTTP 调用",
                "kind": "client",
            },
            {
                "path": "sample/petstore_exceptions.py",
                "purpose": "HTTP 状态码到异常的映射",
                "kind": "exceptions",
            },
            {
                "path": "sample/__init__.py",
                "purpose": "声明新包",
                "kind": "package_init",
            },
        ],
        "files_to_modify": [
            {
                "path": "pyproject.toml",
                "reason": "声明新增的运行时依赖",
                "changes": ["httpx>=0.27"],
            }
        ],
        "dependencies": [
            {
                "name": "httpx",
                "version": None,
                "reason": "HTTP 客户端",
                "already_installed": True,
            }
        ],
        "endpoints": [{"method": "GET", "path": "/pets", "purpose": "列出宠物"}],
        "authentication": {
            "scheme": "apiKey",
            "strategy": "在每个请求的 header 中附加 X-API-Key",
            "config_location": "环境变量",
            "required_env_vars": ["PETSTORE_API_KEY"],
            "header_name": "X-API-Key",
        },
        "error_handling": {"exception_type": "PetstoreAPIError", "rules": [], "notes": []},
        "testing_strategy": {"unit_tests": [], "integration_tests": [], "contract_tests": []},
        "assumptions": [],
        "risks": [],
        "warnings": [],
    }
    payload.update(overrides)
    return payload


def _final_turn(payload: dict | None = None) -> AssistantTurn:
    return AssistantTurn(
        content=json.dumps({"final": payload if payload is not None else _payload()}),
        finish_reason="stop",
    )


def _raw_final_turn(content: str) -> AssistantTurn:
    return AssistantTurn(content=content, finish_reason="stop")


def _call(call_id: str, name: str, arguments: str = "{}") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _tool_turn(*calls: ToolCallRequest) -> AssistantTurn:
    return AssistantTurn(tool_calls=list(calls))


def _tool_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [message for message in messages if message.role == "tool"]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class _SpyTool:
    """记录调用的只读假工具，用来证明"到底有没有被执行"。"""

    def __init__(self, name: str = "spy") -> None:
        self.spec = ToolSpec(name=name, description=f"spy tool {name}")
        self.invocations: list[dict] = []

    def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append(dict(args))
        return ToolResult(call_id="", tool=self.spec.name, ok=True, content="ok")


class _RecordingLoop:
    """不做任何事、只记录 Planner 传进来了什么的 AgentLoop 实现。"""

    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload if payload is not None else _payload()
        self.captured: dict = {}

    def run(self, *, system_prompt, user_prompt, context, registry, llm) -> AgentRunResult:
        self.captured = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": context,
            "registry": registry,
            "llm": llm,
        }
        return AgentRunResult(
            status="completed",
            final_message=ChatMessage(
                role="assistant", content=json.dumps({"final": self._payload})
            ),
            turns=1,
        )


# ------------------------------------------------------------------ 1~2 契约与初始状态


def test_01_tool_using_planner_implements_the_integration_planner_protocol(state) -> None:
    planner = ToolUsingPlanner(FakeToolCallingClient(_final_turn()))

    assert isinstance(planner, IntegrationPlanner)
    assert callable(planner.plan)


def test_02_initial_state_reaches_the_agent_loop(state) -> None:
    loop = _RecordingLoop()
    client = FakeToolCallingClient(_final_turn())
    planner = ToolUsingPlanner(client, loop=loop)

    plan = planner.plan(state)

    captured = loop.captured
    # ToolContext 只承载只读事实，且直接复用 PlannerState，不重新扫描
    assert isinstance(captured["context"], ToolContext)
    assert captured["context"].api is state.api
    assert captured["context"].project is state.project
    assert captured["context"].project_root == Path(state.project.root)
    # known_files 是"确实存在"的集合：.py 文件清单 + 清单文件（pyproject.toml 等）
    assert captured["context"].known_files == {
        *state.project.python_files,
        *state.project.manifest_files,
    }
    assert "sk-" not in captured["context"].model_dump_json().lower()
    # 初始上下文包含 API 基础信息与项目结构摘要，但不包含任何工具的执行结果
    assert "Demo Petstore API" in captured["user_prompt"]
    assert "sample-project" in captured["user_prompt"]
    assert REQUEST in captured["user_prompt"]
    assert "inspect_project" in captured["system_prompt"]
    # 注册表原样传下去，工具定义是四个只读工具（按名字排序，顺序稳定）
    assert [spec.name for spec in captured["registry"].specs()] == [
        "inspect_api",
        "inspect_project",
        "read_file",
        "search_code",
    ]
    assert isinstance(plan, IntegrationPlan)


# ------------------------------------------------------------------ 3~6 四个工具都能用


def test_03_the_agent_can_call_inspect_project(state) -> None:
    client = FakeToolCallingClient([_tool_turn(_call("c1", "inspect_project")), _final_turn()])

    plan = ToolUsingPlanner(client).plan(state)

    assert "name: sample-project" in _tool_messages(client.requests[1])[0].content
    assert isinstance(plan, IntegrationPlan)


def test_04_the_agent_can_call_search_code(state) -> None:
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "search_code", '{"query": "def fetch_pets"}')),
            _final_turn(),
        ]
    )

    ToolUsingPlanner(client).plan(state)

    assert "pkg/client.py" in _tool_messages(client.requests[1])[0].content


def test_05_the_agent_can_call_read_file(state) -> None:
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "read_file", '{"path": "pkg/client.py"}')),
            _final_turn(),
        ]
    )

    ToolUsingPlanner(client).plan(state)

    assert "def fetch_pets" in _tool_messages(client.requests[1])[0].content


def test_06_the_agent_can_call_inspect_api(state) -> None:
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "inspect_api", '{"mode": "endpoints"}')),
            _final_turn(),
        ]
    )

    ToolUsingPlanner(client).plan(state)

    content = _tool_messages(client.requests[1])[0].content
    assert "GET /pets" in content
    assert "listPets" in content


# ------------------------------------------------------------------ 7~8 最终 JSON


def test_07_the_final_envelope_is_parsed_into_an_integration_plan(state) -> None:
    client = FakeToolCallingClient(_final_turn())

    plan = ToolUsingPlanner(client).plan(state)

    assert isinstance(plan, IntegrationPlan)
    assert plan.integration_goal.startswith("把 Demo Petstore API 集成进")
    assert [item.path for item in plan.files_to_create][0] == "sample/petstore_client.py"
    assert plan.error_handling.exception_type == "PetstoreAPIError"
    assert plan.warnings == []  # 干净的输入不该产出噪音


@pytest.mark.parametrize(
    "content",
    [
        "这不是 JSON",
        "",
        '{"plan": {}}',  # 包络键写错
        '{"final": "not an object"}',  # 包络里不是对象
        "[1, 2, 3]",  # 顶层不是对象
    ],
    ids=["prose", "empty", "wrong-key", "not-an-object", "array"],
)
def test_08_an_invalid_final_envelope_raises(state, content: str) -> None:
    client = FakeToolCallingClient(_raw_final_turn(content))

    with pytest.raises(PlanningError):
        ToolUsingPlanner(client).plan(state)


def test_08b_parse_final_plan_extracts_only_the_envelope() -> None:
    assert parse_final_plan('{"final": {"a": 1}}') == {"a": 1}
    with pytest.raises(PlanningError):
        parse_final_plan('{"a": 1}')


def test_08c_a_structurally_incomplete_plan_raises(state) -> None:
    payload = _payload()
    del payload["integration_goal"]
    client = FakeToolCallingClient(_final_turn(payload))

    with pytest.raises(PlanningError):
        ToolUsingPlanner(client).plan(state)


# ------------------------------------------------------------------ 9~11 grounding


def test_09_hallucinated_endpoints_are_dropped(state) -> None:
    payload = _payload(
        endpoints=[
            {"method": "GET", "path": "/pets", "purpose": "真实存在的端点"},
            {"method": "GET", "path": "/users", "purpose": "编造的端点"},
            {"method": "DELETE", "path": "/pets", "purpose": "同 path 但方法不存在"},
        ]
    )
    client = FakeToolCallingClient(_final_turn(payload))

    plan = ToolUsingPlanner(client).plan(state)

    assert [(item.method, item.path) for item in plan.endpoints] == [("GET", "/pets")]
    assert any("/users" in warning for warning in plan.warnings)
    assert any("DELETE /pets" in warning for warning in plan.warnings)


def test_10_hallucinated_files_are_dropped(state) -> None:
    payload = _payload(
        files_to_modify=[
            {"path": "pyproject.toml", "reason": "真实存在", "changes": []},
            {"path": "src/ghost.py", "reason": "不存在的文件", "changes": []},
        ]
    )
    client = FakeToolCallingClient(_final_turn(payload))

    plan = ToolUsingPlanner(client).plan(state)

    assert [item.path for item in plan.files_to_modify] == ["pyproject.toml"]
    assert any("src/ghost.py" in warning for warning in plan.warnings)
    # 绝不因为模型提到就创建一个文件
    assert not (Path(state.project.root) / "src" / "ghost.py").exists()


def test_10b_files_that_already_exist_cannot_be_created(state) -> None:
    payload = _payload(
        files_to_create=[
            {"path": "sample/petstore_client.py", "purpose": "新文件", "kind": "client"},
            {"path": "pkg/client.py", "purpose": "已存在的文件", "kind": "other"},
        ]
    )
    client = FakeToolCallingClient(_final_turn(payload))

    plan = ToolUsingPlanner(client).plan(state)

    assert "pkg/client.py" not in [item.path for item in plan.files_to_create]
    assert any("pkg/client.py" in warning for warning in plan.warnings)


def test_11_fact_fields_come_from_the_planner_state_not_the_model(state) -> None:
    client = FakeToolCallingClient(_final_turn())

    plan = ToolUsingPlanner(client).plan(state)

    assert plan.target_api.name == state.api.name
    assert plan.target_api.version == state.api.version
    assert plan.target_api.base_url == state.api.base_url
    assert plan.target_api.total_endpoints == len(state.api.endpoints)
    assert plan.repository.root == state.project.root
    assert plan.repository.name == state.project.name
    assert plan.repository.source_dirs == list(state.project.source_dirs)
    # 模型写的 "Wrong API" / "/tmp/wrong-root" 一个字都没进计划
    dumped = plan.model_dump_json()
    assert "Wrong API" not in dumped
    assert "/tmp/wrong-root" not in dumped


# ------------------------------------------------------------------ 12~14 校验与告警


def test_12_validate_plan_is_reused(state) -> None:
    """ "integration_strategy" 取一个非法枚举值 → Pydantic 层拒绝，而不是放过去。"""
    payload = _payload()
    payload["integration_strategy"]["approach"] = "rewrite_everything"
    client = FakeToolCallingClient(_final_turn(payload))

    with pytest.raises(PlanningError) as excinfo:
        ToolUsingPlanner(client).plan(state)

    assert "approach" in str(excinfo.value)


def test_13_enforce_constraints_moves_test_files_out_of_files_to_create(state) -> None:
    payload = _payload()
    payload["files_to_create"].append(
        {"path": "tests/test_petstore_client.py", "purpose": "单元测试", "kind": "other"}
    )
    payload["testing_strategy"]["unit_tests"] = [
        {"path": "tests/test_petstore_client.py", "purpose": "不发起网络请求", "covers": []}
    ]
    client = FakeToolCallingClient(_final_turn(payload))

    plan = ToolUsingPlanner(client).plan(state)

    assert "tests/test_petstore_client.py" not in [item.path for item in plan.files_to_create]
    assert plan.testing_strategy.unit_tests[0].path == "tests/test_petstore_client.py"
    assert any("testing_strategy" in warning for warning in plan.warnings)


def test_14_loop_warnings_and_grounding_warnings_are_both_recorded(state) -> None:
    payload = _payload(endpoints=[{"method": "GET", "path": "/users", "purpose": "编造的端点"}])
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "ghost_tool")),  # 未注册的工具 → Agent Loop 告警
            _final_turn(payload),
        ]
    )

    plan = ToolUsingPlanner(client).plan(state)

    assert any(warning.startswith("Agent Loop：") for warning in plan.warnings)
    assert any("/users" in warning for warning in plan.warnings)
    assert plan.endpoints == []


def test_14b_model_declared_uncertainty_is_kept(state) -> None:
    payload = _payload(warnings=["不确定仓库是否有统一的日志约定"])
    client = FakeToolCallingClient(_final_turn(payload))

    plan = ToolUsingPlanner(client).plan(state)

    assert "不确定仓库是否有统一的日志约定" in plan.warnings


# ------------------------------------------------------------------ 15~18 安全边界


def test_15_the_model_never_executes_tools_itself(state) -> None:
    spy = _SpyTool()
    client = FakeToolCallingClient(_final_turn())

    plan = ToolUsingPlanner(client, registry=ToolRegistry([spy])).plan(state)

    # 工具定义提供给了模型，但模型只回了一个最终回答 —— 什么都没被执行
    assert [item["function"]["name"] for item in client.tools[0]] == ["spy"]
    assert spy.invocations == []
    assert isinstance(plan, IntegrationPlan)


def test_15b_the_model_cannot_reach_a_tool_through_a_stray_argument(state) -> None:
    """arguments 里的路径不是"执行许可"：越界路径由工具自己拒绝，不是由模型决定。"""
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "read_file", '{"path": "../../etc/passwd"}')),
            _final_turn(),
        ]
    )

    ToolUsingPlanner(client).plan(state)

    observation = _tool_messages(client.requests[1])[0].content
    assert "def fetch_pets" not in observation


def test_16_the_planner_does_not_touch_the_repository(state) -> None:
    root = Path(state.project.root)
    before = _snapshot(root)
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "inspect_project")),
            _tool_turn(_call("c2", "search_code", '{"query": "def fetch_pets"}')),
            _tool_turn(_call("c3", "read_file", '{"path": "pkg/client.py"}')),
            _final_turn(),
        ]
    )

    ToolUsingPlanner(client).plan(state)

    assert _snapshot(root) == before


def test_17_the_planner_never_reparses_the_openapi_document(state, monkeypatch) -> None:
    import integration_agent.api as api_package
    import integration_agent.api.parser as api_parser

    def bomb(*args, **kwargs):
        raise AssertionError("ToolUsingPlanner 不得重新解析 OpenAPI")

    monkeypatch.setattr(api_parser, "parse_openapi_text", bomb)
    monkeypatch.setattr(api_package, "parse_openapi_text", bomb)
    client = FakeToolCallingClient(_final_turn())

    plan = ToolUsingPlanner(client).plan(state)

    # API 事实来自 PlannerState，与传入的对象是同一份
    assert plan.target_api.name == state.api.name

    imports = _imported_modules(Path(tool_planner_module.__file__))
    assert imports & FORBIDDEN_PLANNER_IMPORTS == set()
    assert imports <= ALLOWED_PLANNER_IMPORTS


def test_18_the_planner_does_not_scan_the_repository_by_itself(state, monkeypatch) -> None:
    import integration_agent.tools.repository_tools as repository_tools

    calls: list[str] = []
    original = repository_tools.scan_repository

    def counting_scan(root):
        calls.append(str(root))
        return original(root)

    monkeypatch.setattr(repository_tools, "scan_repository", counting_scan)

    # 模型不要求看项目结构 → 一次扫描都不该发生
    ToolUsingPlanner(FakeToolCallingClient(_final_turn())).plan(state)
    assert calls == []

    # 模型明确要求 inspect_project → 恰好一次，而且是工具发起的
    ToolUsingPlanner(
        FakeToolCallingClient([_tool_turn(_call("c1", "inspect_project")), _final_turn()])
    ).plan(state)
    assert len(calls) == 1
    assert calls[0] == state.project.root


# ------------------------------------------------------------------ 19~20 请求与预算


def test_19_the_user_request_reaches_the_model(state) -> None:
    client = FakeToolCallingClient(_final_turn())

    ToolUsingPlanner(client).plan(state)

    user_message = client.requests[0][1]
    assert user_message.role == "user"
    assert REQUEST in user_message.content
    assert client.requests[0][0].role == "system"


def test_20_the_tool_call_budget_is_enforced(state) -> None:
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "inspect_project")),
            _tool_turn(_call("c2", "inspect_api", '{"mode": "summary"}')),
        ]
    )
    planner = ToolUsingPlanner(client, config=AgentLoopConfig(max_tool_calls=1))

    with pytest.raises(PlanningError) as excinfo:
        planner.plan(state)

    assert "max_tool_calls" in str(excinfo.value)
    assert client.calls == 2


def test_20b_an_agent_loop_that_does_not_finish_raises_instead_of_falling_back(state) -> None:
    client = FakeToolCallingClient([_tool_turn(_call("c1", "ghost"))])
    planner = ToolUsingPlanner(client, config=AgentLoopConfig(max_turns=1))

    with pytest.raises(PlanningError) as excinfo:
        planner.plan(state)

    assert "max_turns" in str(excinfo.value)


# ------------------------------------------------------------------ 构造与协议护栏


def test_constructor_rejects_a_single_turn_llm(state) -> None:
    class _GenerateOnly:
        def generate(self, prompt: str) -> str:
            return "{}"

    with pytest.raises(PlanningError):
        ToolUsingPlanner(_GenerateOnly())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"max_endpoints": 0}, "max_endpoints"),
        ({"max_prompt_chars": 10}, "max_prompt_chars"),
    ],
)
def test_constructor_rejects_out_of_range_options(state, kwargs, expected) -> None:
    with pytest.raises(PlanningError) as excinfo:
        ToolUsingPlanner(FakeToolCallingClient(_final_turn()), **kwargs)
    assert expected in str(excinfo.value)


def test_the_default_loop_is_the_agent_loop_runner(state) -> None:
    planner = ToolUsingPlanner(FakeToolCallingClient(_final_turn()))

    assert isinstance(planner.loop, AgentLoopRunner)


def test_the_planner_module_has_no_write_capability() -> None:
    tree = ast.parse(Path(tool_planner_module.__file__).read_text(encoding="utf-8"))
    used_attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert used_attributes.isdisjoint(
        {
            "write_text",
            "write_bytes",
            "mkdir",
            "unlink",
            "rename",
            "rmdir",
            "remove",
            "system",
            "popen",
            "check_output",
            "eval",
            "exec",
        }
    )
    calls = [
        node.func
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert all(call.id != "open" for call in calls)


def test_the_prompt_carries_no_credentials(state, monkeypatch) -> None:
    """提示词是发给外部模型的，凭据一个字节都不该出现。

    刻意**不去搜 "authorization" / "bearer" / "apiKey" 这类词**：它们是
    OpenAPI securityScheme 与 OUTPUT_SKELETON 里的字段名/错误类别名
    （"bearer_format"、"authorization" 错误类型），描述的是认证*机制*，
    不是凭据*本身*。按词搜索只会得到一堆假阳性，证明不了任何事。

    真正能证明问题的是植入金丝雀：把这个计划自己声明的环境变量设成一个
    可识别的哨兵值，然后检查它有没有出现在发给模型的文本里。任何读取
    环境变量的代码路径都会让哨兵漏出来。

    模块层面还有一道结构性保证：tool_planner 的 import 白名单里没有 os
    （见 test_17），所以"读环境变量"这件事在本模块里无从发生。
    """
    monkeypatch.setenv("PETSTORE_API_KEY", "sk-CANARY-0000-do-not-leak")
    client = FakeToolCallingClient(_final_turn())

    ToolUsingPlanner(client).plan(state)

    system_prompt, user_prompt = client.requests[0][0].content, client.requests[0][1].content
    assert "sk-CANARY-0000-do-not-leak" not in system_prompt + user_prompt
    assert "CANARY" not in system_prompt + user_prompt
    # 正对照：认证*机制*照常进入提示词。API 文档里公开写着 type=apiKey、
    # name=X-API-Key、in=header —— 模型必须知道有这回事才能规划认证；
    # 但这些是 OpenAPI 规范里的公开标识，不是凭据。
    # 真正的凭据（密钥值、环境变量名）一个都不出现。
    assert '"type": "apiKey"' in user_prompt
    assert "X-API-Key" in user_prompt
    assert "PETSTORE_API_KEY" not in user_prompt


def test_the_prompt_is_a_deterministic_function_of_the_state(state) -> None:
    """同样的 state + 同样的注册表 → 逐字节相同的 prompt。

    这条同时是一道泄漏护栏：prompt 若是 Pure 的，就不可能偷偷掺进
    "这一次运行时环境里恰好存在"的东西（时间戳、机器名、环境变量）。
    """
    first = FakeToolCallingClient(_final_turn())
    second = FakeToolCallingClient(_final_turn())

    ToolUsingPlanner(first).plan(state)
    ToolUsingPlanner(second).plan(state)

    assert first.requests[0][0] == second.requests[0][0]
    assert first.requests[0][1] == second.requests[0][1]
    assert first.tools[0] == second.tools[0]
