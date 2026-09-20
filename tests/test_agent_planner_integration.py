"""E2 Commit 6 集成测试：API Request → Agent Planner → IntegrationPlan → 后续 Pipeline。

本文件验证的是**接线**，不是 Agent Loop（tests/test_agent_loop.py），也不是
ToolUsingPlanner 的解析细节（tests/test_tool_planner.py）。这里只问一件事：

    打开 use_agent_planner 之后，ToolUsingPlanner 是否真的替掉了确定性 Planner，
    它产出的 IntegrationPlan 是否真的流进了后续的 Code Generator / Test Runner /
    Repair Loop —— 而不是"多了一个配置字段，实际什么也没换"。

所有场景都注入 FakeToolCallingClient：不发网络请求、不需要真实 API Key、结果确定。
真实 DeepSeek 的端到端验证见 scripts/e2e_agent_planner.py（不参与 pytest）。

安全边界：
    - 全部只读：场景 7 用 size + mtime_ns + sha256 证明整条链路不碰仓库。
    - 不涉及 shell / subprocess / git / 网络。
    - 场景 5b 顺带证明"让 Agent 去跑 shell / 删文件"是不可能成功的：工具未注册。
"""

import hashlib
import json
from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from integration_agent import pipeline
from integration_agent.agent import (
    AgentRunResult,
    DeterministicPlanner,
    PlannerState,
    ToolUsingPlanner,
)
from integration_agent.agent.agent_loop import AgentLoopRunner
from integration_agent.agent.llm import AssistantTurn, FakeToolCallingClient, ToolCallRequest
from integration_agent.api import parse_openapi
from integration_agent.api_server.app import app
from integration_agent.pipeline import orchestrator
from integration_agent.repository import scan_repository

# 注意：api_server/__init__.py 把包属性 app 绑定为 FastAPI 实例，
# 因此用 import_module 直接取 app 模块对象，供 monkeypatch 使用。
APP_MODULE = import_module("integration_agent.api_server.app")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SPEC_PATH = EXAMPLES / "openapi" / "petstore.yaml"
PROJECT_PATH = EXAMPLES / "demo_project"

client = TestClient(app)

VALID_PAYLOAD = {
    "api_spec": "openapi/petstore.yaml",
    "project_path": "demo_project",
}
AGENT_PAYLOAD = {**VALID_PAYLOAD, "use_agent_planner": True}

# petstore.yaml 里共 3 个端点；确定性 Planner 会全部选中。
ALL_ENDPOINTS = 3
# Agent 只挑这一个——用来证明"流下去的是 Agent 的计划，不是确定性计划"。
AGENT_ENDPOINT = ("GET", "/pets")
KEEP_ONE = {f"{AGENT_ENDPOINT[0]} {AGENT_ENDPOINT[1]}"}

TOOL_NAMES = ["inspect_api", "inspect_project", "read_file", "search_code"]


# ------------------------------------------------------------------ 辅助


def _state(request: str = "") -> PlannerState:
    """与 Pipeline 内部同源的 PlannerState（真实解析 + 真实扫描，只读）。"""
    return PlannerState(
        api=parse_openapi(SPEC_PATH),
        project=scan_repository(PROJECT_PATH),
        request=request,
    )


def _flatten_endpoints(data: dict) -> list[dict[str, str]]:
    """把计划 dump 里的 PlannedEndpoint 压成 LLM 要回答的扁平形状。

    真实模型只回答 method / path / purpose，参数与 schema 一律从 APIInfo 取
    （见 deepseek_planner.resolve_endpoints）。构造假回答时必须还原这个形状，
    否则端点会被当成"不在 API 文档中"而丢弃——那样测的就是构造错误，不是链路。
    """
    return [
        {
            "method": item["endpoint"]["method"],
            "path": item["endpoint"]["path"],
            "purpose": item["purpose"],
        }
        for item in data["endpoints"]
    ]


def _plan_payload(state: PlannerState, *, keep: set[str] | None = None) -> dict:
    """以一个真实合法的计划为骨架，构造假模型的最终回答。

    骨架取自 DeterministicPlanner —— 目的是保证 validate_plan / enforce_constraints
    一定通过，从而让测试专注于"接线"而不是"手写一个合法的 IntegrationPlan"。
    keep 用来把端点裁成子集：最终计划里只剩 Agent 挑的那几个端点，是
    "Agent 的计划真的流到了下游"最直接的证据。
    """
    data = DeterministicPlanner().plan(state).model_dump()
    selected = _flatten_endpoints(data)
    if keep is not None:
        selected = [item for item in selected if f"{item['method'].upper()} {item['path']}" in keep]
    data["endpoints"] = selected
    return data


def _final_turn(data: dict) -> AssistantTurn:
    """最终回答：ToolUsingPlanner 只认 content 里的 {"final": {...}}。"""
    return AssistantTurn(content=json.dumps({"final": data}))


def _call(call_id: str, name: str, arguments: str = "{}") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _agent_turns(state: PlannerState, *, keep: set[str] | None = None) -> list[AssistantTurn]:
    """一次完整的四轮 Agent 回路：inspect_project → search_code → read_file → final。

    刻意用三个不同的只读工具，逐轮扩大观察面：结构 → 检索 → 单文件正文。
    第三个工具的参数取自第二个工具的观察结果（"placeholder" 命中 service.py），
    这也顺带证明 tool 消息确实按顺序回填给了"模型"。
    """
    return [
        AssistantTurn(tool_calls=[_call("call-1", "inspect_project")]),
        AssistantTurn(
            tool_calls=[
                _call("call-2", "search_code", '{"query": "placeholder", "max_results": 5}')
            ]
        ),
        AssistantTurn(
            tool_calls=[_call("call-3", "read_file", '{"path": "demo_project/service.py"}')]
        ),
        _final_turn(_plan_payload(state, keep=keep)),
    ]


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    """仓库指纹：相对路径 → (size, mtime_ns, sha256)。"""
    result: dict[str, tuple[int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns, digest)
    return result


def install_fake_llm(monkeypatch, turns) -> FakeToolCallingClient:
    """把 app 模块里的 DeepSeekLLMClient 换成 FakeToolCallingClient。

    只替换**构造出来的客户端**，不改 app 的任何接线代码：因此走的是真实的
    ToolUsingPlanner(llm_client) 构造路径，只是 llm 不发网络请求。
    """
    fake = FakeToolCallingClient(turns)
    monkeypatch.setattr(APP_MODULE, "DeepSeekLLMClient", lambda **kwargs: fake)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")
    return fake


class _RecordingLoop:
    """包住真实的 AgentLoopRunner：不重写回路，只在跑完后留下一份结果。

    §九 要求断言 AgentRunResult.status / tool_calls，而 PipelineResult 里不带
    AgentRunResult —— 要把它取出来只能靠这个 spy。回路本身仍然是
    ToolUsingPlanner 默认使用的那个 AgentLoopRunner，一行逻辑都没有复制。
    """

    def __init__(self, captured: dict) -> None:
        self._inner = AgentLoopRunner()
        self._captured = captured

    def run(self, **kwargs) -> AgentRunResult:
        result = self._inner.run(**kwargs)
        self._captured["run_result"] = result
        self._captured["context"] = kwargs["context"]
        self._captured["llm"] = kwargs["llm"]
        self._captured["registry"] = kwargs["registry"]
        return result


def recording_planner_class(captured: dict) -> type:
    """返回一个 ToolUsingPlanner 子类，只多注入一个记录型 loop。"""

    class RecordingPlanner(ToolUsingPlanner):
        def __init__(self, llm, **kwargs) -> None:
            super().__init__(llm, loop=_RecordingLoop(captured), **kwargs)

    return RecordingPlanner


class _RecordingGenerator:
    """包住真实的 DeterministicCodeGenerator，只记录它收到的 plan 是哪一个。"""

    def __init__(self, inner, sink: list) -> None:
        self._inner = inner
        self._sink = sink

    def generate(self, plan):
        self._sink.append(plan)
        return self._inner.generate(plan)


# =============================================== 场景 1：默认行为不变


def test_default_request_touches_no_llm_and_still_uses_the_deterministic_planner(
    monkeypatch,
) -> None:
    """两个开关都不开：仍然是确定性 Planner，且**根本不构造任何 LLM 客户端**。

    比"kwargs 里没有 planner"更强：把 LLM 客户端与两个 Planner 类都换成炸弹，
    只要有任何一条被摸到，测试就会炸出来。
    """

    def boom(*args, **kwargs):
        raise AssertionError("默认请求不应构造任何 LLM 组件")

    monkeypatch.setattr(APP_MODULE, "DeepSeekLLMClient", boom)
    monkeypatch.setattr(APP_MODULE, "ToolUsingPlanner", boom)
    monkeypatch.setattr(APP_MODULE, "DeepSeekPlanner", boom)
    # 客户端不构造，就不该读 Key；这里连环境变量都不给
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    response = client.post("/api/integrations/run", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert len(body["plan"]["endpoints"]) == ALL_ENDPOINTS
    assert body["patch"] is not None


def test_agent_planner_flag_defaults_to_false() -> None:
    """请求模型层：新字段默认 False，老请求体（不带该字段）行为完全不变。"""
    from integration_agent.api_server.models import IntegrationRunRequest

    request = IntegrationRunRequest.model_validate(VALID_PAYLOAD)

    assert request.use_agent_planner is False
    assert request.use_llm_planner is False
    assert request.use_llm is False
    # 老字段一个都没少
    assert request.api_spec == "openapi/petstore.yaml"
    assert request.project_path == "demo_project"
    assert request.request == ""
    assert request.max_iterations == 3
    assert request.demo_mode is False


# =============================================== 场景 2：真的选中了 Agent Planner


def test_use_agent_planner_really_selects_tool_using_planner(monkeypatch) -> None:
    """接线层：注入 Pipeline 的确实是 ToolUsingPlanner 实例，且它拿到的是同一个客户端。"""
    captured: dict = {}
    built: list[dict] = []

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return pipeline.PipelineResult(
            status="passed", api=None, project=None, repair_loop_result=None
        )

    fake = FakeToolCallingClient([])

    def fake_client(**kwargs):
        built.append(kwargs)
        return fake

    monkeypatch.setattr(APP_MODULE, "run_pipeline", fake_run)
    monkeypatch.setattr(APP_MODULE, "DeepSeekLLMClient", fake_client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")

    client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    planner = captured["planner"]
    assert isinstance(planner, ToolUsingPlanner), "use_agent_planner 必须真的替换 Planner"
    assert planner.llm is fake
    # 只开 Agent Planner 不连带启用修复
    assert "repair_applier" not in captured
    # 四个只读工具全部注册进去了（没有 if/elif 分派之外的额外工具）
    assert [spec.name for spec in planner.registry.specs()] == TOOL_NAMES
    # tool calling 与 response_format=json_object 在真实 API 上互斥：Agent Planner
    # 拿到的必须是 json_mode=False 的客户端（真实 API 实测 HTTP 400，见 scripts/
    # e2e_agent_planner.py）。这条断言把"客户端配置"钉死在测试里，防止悄悄回归。
    assert {"json_mode": False} in built, "Agent Planner 必须用 json_mode=False 的客户端"


def test_use_agent_planner_does_not_touch_the_repository(monkeypatch) -> None:
    """只是选中一个 Planner，不该产生任何副作用。"""
    before = _snapshot(PROJECT_PATH)
    install_fake_llm(monkeypatch, _agent_turns(_state(), keep=KEEP_ONE))

    client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert _snapshot(PROJECT_PATH) == before


# =============================================== 场景 3：多轮 tool calling 全链路


def test_multi_round_tool_calling_ends_with_a_plan(monkeypatch) -> None:
    """四轮：inspect_project → search_code → read_file → final IntegrationPlan。

    断言的是**真实跑过的那次 Agent 运行**（经 spy 取出的 AgentRunResult），
    不是"构造了一个 Planner 对象"。
    """
    captured: dict = {}
    state = _state()
    fake = install_fake_llm(
        monkeypatch,
        _agent_turns(state, keep=KEEP_ONE),
    )
    monkeypatch.setattr(APP_MODULE, "ToolUsingPlanner", recording_planner_class(captured))

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["plan"] is not None

    # ---- Agent Loop 真的跑完了 ----
    run: AgentRunResult = captured["run_result"]
    assert run.status == "completed"
    assert run.turns == 4, "三次工具调用 + 一次最终回答 = 四次 LLM 请求"
    assert run.call_count > 1, "必须真的发生了多轮 tool calling"
    assert [call.name for call in run.tool_calls] == ["inspect_project", "search_code", "read_file"]
    assert all(result.ok for result in run.tool_results), "工具调用必须全部成功"
    assert run.error is None

    # ---- 每一轮都把上一轮的 tool 结果原样带回去了 ----
    assert fake.calls == 4
    # 第一轮只有 system + user；之后每轮多一条 assistant + 一条 tool
    assert [len(item) for item in fake.requests] == [2, 4, 6, 8]
    observations = [message.content for message in fake.requests[-1] if message.role == "tool"]
    assert len(observations) == 3
    assert "placeholder" in observations[1], "search_code 的命中必须回到模型手上"
    assert "def get_user_profile" in observations[2], "read_file 的正文必须回到模型手上"
    # 工具定义每一轮都带上了，且是按名字排序的四个只读工具
    assert all([item["function"]["name"] for item in tools] == TOOL_NAMES for tools in fake.tools)

    # ---- 最终产出的 IntegrationPlan 来自 Agent 挑的端点 ----
    assert [item["endpoint"]["method"] for item in body["plan"]["endpoints"]] == [AGENT_ENDPOINT[0]]
    assert [item["endpoint"]["path"] for item in body["plan"]["endpoints"]] == [AGENT_ENDPOINT[1]]


# =============================================== 场景 4：计划真的进了 Code Generator


def test_agent_plan_is_the_one_handed_to_the_code_generator(monkeypatch) -> None:
    """Code Generator 收到的就是 ToolUsingPlanner 产出的那个 IntegrationPlan。

    用一个只做记录的 DeterministicCodeGenerator 包装器取到入参 plan：既不改
    Code Generator 的一行代码，也不用重新实现它。
    """
    captured_plans: list = []
    real_generator = orchestrator.DeterministicCodeGenerator()
    monkeypatch.setattr(
        orchestrator,
        "DeterministicCodeGenerator",
        lambda: _RecordingGenerator(real_generator, captured_plans),
    )
    state = _state()
    install_fake_llm(monkeypatch, _agent_turns(state, keep=KEEP_ONE))

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert len(captured_plans) == 1, "一次请求只生成一次代码"
    plan = captured_plans[0]

    # plan 与响应里序列化出来的是同一份
    assert plan.model_dump(mode="json") == body["plan"]
    # 而且就是 Agent 挑的那一个端点，不是确定性 Planner 的 3 个
    assert [(item.endpoint.method, item.endpoint.path) for item in plan.endpoints] == [
        AGENT_ENDPOINT
    ]
    # 端点详情（参数 / 响应 schema）来自真实 APIInfo，不是模型编的
    assert plan.endpoints[0].endpoint.operation_id == "listPets"
    # Code Generator 之后照常产出可用的 artifacts / patch
    assert body["artifacts"]["files"]
    assert body["patch"]["summary"]["created"] > 0


def test_agent_plan_changes_what_gets_generated(monkeypatch) -> None:
    """同一份输入，换一个 Agent 的计划，产物就该不一样——证明计划确实驱动了下游。"""
    state = _state()
    install_fake_llm(monkeypatch, _agent_turns(state, keep=KEEP_ONE))
    agent_body = client.post("/api/integrations/run", json=AGENT_PAYLOAD).json()

    baseline = client.post("/api/integrations/run", json=VALID_PAYLOAD).json()

    assert len(baseline["plan"]["endpoints"]) == ALL_ENDPOINTS
    assert len(agent_body["plan"]["endpoints"]) == 1
    assert agent_body["plan"]["endpoints"] != baseline["plan"]["endpoints"]
    assert agent_body["patch"]["unified_diff"] != baseline["patch"]["unified_diff"]
    assert agent_body["patch"]["unified_diff"], "Agent 的计划同样要产出可用的 diff"


# =============================================== 场景 5：规划失败不得进入 Repair Loop


@pytest.mark.parametrize("unknown_tool", ["run_shell", "delete_repository"])
def test_agent_failure_ends_at_plan_and_never_repairs(monkeypatch, unknown_tool: str) -> None:
    """Agent Planner 失败 → status=error / failed_stage=plan，且 Repair Loop 一次都没跑。

    两件事必须同时成立：
        1. 失败被如实返回（不静默退回确定性规划）；
        2. 规划阶段失败**不触发** Repair Loop —— Repair 只处理"代码生成之后的测试失败"。
    """
    repair_calls: list = []
    real_repair_loop = orchestrator.run_repair_loop

    def spy_repair_loop(*args, **kwargs):
        repair_calls.append(kwargs)
        return real_repair_loop(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "run_repair_loop", spy_repair_loop)
    # 模型连续两次要求调用一个不存在的工具（一次 shell、一次删除）→ tool_error
    install_fake_llm(
        monkeypatch,
        [
            AssistantTurn(tool_calls=[_call("call-1", unknown_tool)]),
            AssistantTurn(tool_calls=[_call("call-2", unknown_tool)]),
        ],
    )

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["failed_stage"] == "plan"
    assert body["plan"] is None, "失败时不得退回确定性计划"
    assert body["artifacts"] is None
    assert body["repair_loop_result"] is None
    assert repair_calls == [], "规划失败绝不能进入 Repair Loop"
    assert body["error"] == "Integration failed during plan."


def test_llm_error_ends_at_plan_without_leaking_the_exception(monkeypatch) -> None:
    """LLM 侧故障（如 401）同样收敛在 plan 阶段，异常原文与密钥都不外泄。"""

    class FailingClient:
        def chat(self, messages, **kwargs):
            raise RuntimeError("DeepSeek API 返回 HTTP 401：invalid api key sk-live-shouldnotleak")

    monkeypatch.setattr(APP_MODULE, "DeepSeekLLMClient", lambda **kwargs: FailingClient())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["failed_stage"] == "plan"
    assert body["plan"] is None
    assert body["repair_loop_result"] is None
    assert "sk-live-shouldnotleak" not in response.text
    assert "invalid api key" not in response.text


def test_repair_spy_sees_the_repair_loop_when_planning_succeeds(monkeypatch) -> None:
    """阳性对照：规划成功时 Repair Loop 必须被调用。

    没有这条对照，"规划失败时 repair_calls == []" 就可能是假的——万一 spy 根本没
    接上（比如 patch 错了模块），它同样会是空的。这条对照证明 spy 确实在观测真实调用。
    """
    repair_calls: list = []
    real_repair_loop = orchestrator.run_repair_loop

    def spy_repair_loop(*args, **kwargs):
        repair_calls.append(kwargs)
        return real_repair_loop(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "run_repair_loop", spy_repair_loop)
    install_fake_llm(monkeypatch, _agent_turns(_state(), keep=KEEP_ONE))

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.json()["status"] == "passed"
    assert len(repair_calls) == 1, "规划成功时 Repair Loop 必须被调用（且只调用一次）"


# =============================================== 场景 6：两个 Planner 开关互斥


def test_both_planner_flags_run_only_the_agent_planner(monkeypatch) -> None:
    """use_agent_planner + use_llm_planner 同时开启：只构造、只运行 ToolUsingPlanner。

    这里用的是 FakeToolCallingClient —— 它**没有** generate() 方法。如果 app 误选了
    DeepSeekPlanner，规划阶段会当场炸成 status=error；因此"跑通"本身就是选择正确的证明。
    """
    built: list[str] = []
    agent_class = recording_planner_class({})
    real_agent_class = ToolUsingPlanner

    def build_agent(llm, **kwargs):
        built.append("agent")
        return agent_class(llm, **kwargs)

    def build_deepseek(llm, **kwargs):  # pragma: no cover - 被调用即测试失败
        built.append("deepseek")
        raise AssertionError("两个开关同时开启时不应构造 DeepSeekPlanner")

    monkeypatch.setattr(APP_MODULE, "ToolUsingPlanner", build_agent)
    monkeypatch.setattr(APP_MODULE, "DeepSeekPlanner", build_deepseek)
    state = _state()
    fake = install_fake_llm(
        monkeypatch,
        _agent_turns(state, keep=KEEP_ONE),
    )

    response = client.post(
        "/api/integrations/run",
        json={**AGENT_PAYLOAD, "use_llm_planner": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert built == ["agent"], "一次请求只会构造一个 Planner"
    assert fake.calls == 4, "跑的是 Agent Planner（多轮 tool calling），不是 DeepSeekPlanner"
    assert len(body["plan"]["endpoints"]) == 1
    assert isinstance(agent_class, type) and issubclass(real_agent_class, ToolUsingPlanner)


# =============================================== 场景 7：整条链路不修改仓库


def test_the_whole_agent_request_leaves_the_repository_untouched(monkeypatch) -> None:
    """size + mtime_ns + sha256 三者全等：内容没变，也没被"写了又写回"。"""
    before = _snapshot(PROJECT_PATH)
    state = _state()
    install_fake_llm(monkeypatch, _agent_turns(state, keep=KEEP_ONE))

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["status"] == "passed"
    after = _snapshot(PROJECT_PATH)
    assert after == before, "Agent Planner 全链路不得修改目标仓库"
    assert set(after) == set(before)


# =============================================== 场景 8：真实 TestClient 端到端


def test_http_round_trip_with_use_agent_planner(monkeypatch) -> None:
    """真实 POST /api/integrations/run，注入 FakeToolCallingClient，不碰真实 DeepSeek。"""
    state = _state()
    fake = install_fake_llm(
        monkeypatch,
        _agent_turns(state, keep=KEEP_ONE),
    )

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    # 结构化 JSON 可被 PipelineResult 反序列化（round-trip）
    assert pipeline.PipelineResult.model_validate_json(response.text) is not None
    body = response.json()
    assert body["api"]["name"] == "Demo Petstore API"
    assert body["project"]["name"] == "demo-project"
    # 响应里既没有 Key，也没有内部细节
    assert "sk-test-not-a-real-key" not in response.text
    assert "DEEPSEEK_API_KEY" not in response.text
    assert "Traceback" not in response.text
    assert fake.calls == 4


def test_agent_planner_without_key_returns_structured_error(monkeypatch) -> None:
    """未配置 Key：与其他 LLM 开关完全一致的结构化 400，不做任何静默降级。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "LLM_NOT_CONFIGURED"
    assert "DEEPSEEK_API_KEY" in body["error"]["message"]
    assert "sk-" not in response.text


@pytest.mark.parametrize("value", ["yes", 1, 0, None, "", []])
def test_use_agent_planner_validates_exactly_like_the_existing_switches(value) -> None:
    """新字段与既有开关共用同一套 Pydantic 校验：逐值行为必须完全一致。

    注意 Pydantic 对 bool 是**宽松**的（"yes" / 1 → True，0 / "" → False），这是
    use_llm / use_llm_planner 已有的行为；本字段既没有收紧也没有放宽，因此不会
    让任何既有请求体变成非法，也不会多接受既有字段不接受的值。
    """
    from pydantic import ValidationError

    from integration_agent.api_server.models import IntegrationRunRequest

    def outcome(field: str):
        try:
            request = IntegrationRunRequest.model_validate({**VALID_PAYLOAD, field: value})
        except ValidationError:
            return "rejected"
        return getattr(request, field)

    new = outcome("use_agent_planner")
    assert new == outcome("use_llm_planner")
    assert new == outcome("use_llm")


# =============================================== 场景 9：复用 Pipeline 已解析的输入


def test_agent_planner_parses_and_scans_exactly_once(monkeypatch) -> None:
    """§六：Agent Planner 复用 Pipeline 已经解析好的 api / 已经扫描好的 project。

    计数证明：整条请求里 parse_openapi 恰好 1 次、scan_repository 恰好 1 次，
    都来自 Pipeline 自己。Agent Planner 没有第二次解析、也没有第二次扫描
    （inspect_project 是模型显式要求的补充观察，本场景刻意不调用它）。
    """
    calls = {"parse": 0, "scan": 0}
    real_parse = orchestrator.parse_openapi
    real_scan = orchestrator.scan_repository

    def counting_parse(*args, **kwargs):
        calls["parse"] += 1
        return real_parse(*args, **kwargs)

    def counting_scan(*args, **kwargs):
        calls["scan"] += 1
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "parse_openapi", counting_parse)
    monkeypatch.setattr(orchestrator, "scan_repository", counting_scan)
    state = _state()
    install_fake_llm(monkeypatch, _agent_turns(state, keep=KEEP_ONE))

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    assert calls["parse"] == 1, "OpenAPI 只解析一次（Pipeline 自己那次）"
    assert calls["scan"] == 1, "仓库只扫描一次（Pipeline 自己那次）"


def test_agent_context_comes_from_the_pipeline_state(monkeypatch) -> None:
    """Agent 拿到的 ToolContext 就是 Pipeline 的 api / project，不是新建的一份。"""
    captured: dict = {}
    install_fake_llm(
        monkeypatch,
        _agent_turns(_state(), keep=KEEP_ONE),
    )
    monkeypatch.setattr(APP_MODULE, "ToolUsingPlanner", recording_planner_class(captured))

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    context = captured["context"]
    body = response.json()
    # 与响应里序列化出来的 api / project 是同一份事实
    assert context.api.model_dump(mode="json") == body["api"]
    assert context.project.model_dump(mode="json") == body["project"]
    assert context.project_root == PROJECT_PATH
    # known_files 覆盖仓库里的 .py 与清单文件，供路径审计使用
    assert "demo_project/service.py" in context.known_files
    assert "pyproject.toml" in context.known_files
    # 注入的 llm 就是 app 构造的那一个，registry 是默认的四个只读工具
    assert isinstance(captured["llm"], FakeToolCallingClient)
    assert [spec.name for spec in captured["registry"].specs()] == TOOL_NAMES


# =============================================== 场景 10：Agent 的工具面是只读的


def test_agent_cannot_reach_shell_or_write_tools(monkeypatch) -> None:
    """§十三：Agent 只有四个只读工具，要 shell / 要写文件只能得到"工具未注册"。"""
    captured: dict = {}
    install_fake_llm(
        monkeypatch,
        [
            AssistantTurn(tool_calls=[_call("call-1", "run_shell", '{"command": "git push"}')]),
            _final_turn(_plan_payload(_state(), keep=KEEP_ONE)),
        ],
    )
    monkeypatch.setattr(APP_MODULE, "ToolUsingPlanner", recording_planner_class(captured))

    response = client.post("/api/integrations/run", json=AGENT_PAYLOAD)

    assert response.status_code == 200
    run: AgentRunResult = captured["run_result"]
    assert run.status == "completed", "被拒绝的工具调用不会让回路崩掉"
    rejected = run.tool_results[0]
    assert rejected.ok is False
    assert "工具未注册" in (rejected.error or "")
    assert run.tool_results[0].tool == "run_shell"
    # 拒绝之后仍然拿到了计划
    assert response.json()["plan"] is not None
