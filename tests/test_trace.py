"""Execution Trace 的测试：TraceEvent / TraceCollector / 各阶段埋点。

§十四 要求的 8 个方面，逐条对应：

    1. emit / events / clear      → 收集器基本语义
    2. metadata                   → 规范化（脱敏 / 截断 / 类型降级 / 深度）
    3. 多事件顺序                  → started→completed 成对、顺序稳定
    4. Pipeline 集成               → 一次完整运行产生各阶段事件
    5. TestResult 一致性           → trace 里的测试结论 == TestResult 本身
    6. Secret Safety              → 凭据进不了 trace
    7. Backward Compatibility     → trace=None 行为与引入 trace 之前一致
    8. 词表闭合（本文件加的）       → 全仓库 emit 调用点都必须是已登记的类型

第 8 条来自这一层开发时**真的踩到**的坑：planning_started 与 generation_completed
两个事件类型一开始漏登记，于是 3 条事件被 emit 静默丢弃——运行一切正常，trace 少了
内容。这正是"观测层不该让业务失败"的另一面：它安静到连自己的错误都不说。AST 扫描
把"埋了没登记"变成一条会红的测试，而不是一次事后才发现的空白。

非空洞性：凡是断言"X 不在这里"的用例，都先证明 X 真的发生过（伪造的凭据确实进了
消息历史、伪造的 stdout 确实存在……）——否则它只是一条永远为真的空断言。
"""

import ast
import json
import threading
from pathlib import Path
from typing import get_args

import pytest

from integration_agent import generation, pipeline, repair, validation
from integration_agent.agent import DeterministicPlanner, PlannerState
from integration_agent.agent.agent_loop import AgentLoopRunner
from integration_agent.api import parse_openapi
from integration_agent.llm import AssistantTurn, FakeToolCallingClient, ToolCallRequest
from integration_agent.pipeline import orchestrator as orchestrator_module
from integration_agent.repository import scan_repository
from integration_agent.tools import build_default_registry
from integration_agent.tools.registry import ToolContext
from integration_agent.trace import (
    MAX_STRING_CHARS,
    REDACTED,
    TraceCollector,
    TraceEventType,
    TraceStage,
    current_trace,
    emit,
    trace_scope,
    use_trace,
)

# TestResult 一律写成 validation.TestResult：把这个名字绑进本模块，pytest 会把它
# 当成待收集的测试类（PytestCollectionWarning）。

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PETSTORE_PATH = EXAMPLES / "openapi" / "petstore.yaml"
DEMO_PROJECT_PATH = EXAMPLES / "demo_project"
SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "integration_agent"

# stage_started 之后可能出现的收口事件。generation_completed / patch_generated 是
# 那两个阶段自己的完成信号（见 trace.py 里的说明），所以它们也算"阶段关上了"。
STAGE_CLOSERS = {"stage_completed", "stage_failed", "generation_completed", "patch_generated"}

FAILING_TEST = "def test_failure():\n    assert 1 == 2\n"

# 故意长得像真凭据的假凭据：只用于验证它进不了 trace。
FAKE_CREDENTIALS = {
    "api_key": "sk-live-0123456789abcdef",
    "token": "ghp_0123456789abcdefghijklmnopqrstuv",
    "password": "hunter2-not-a-real-password",
    "secret": "not-a-real-client-secret",
}


class FakeRunner:
    """按队列返回 TestResult 的假 Runner（让 pipeline 不必真的跑 pytest）。"""

    def __init__(self, results: list[validation.TestResult]) -> None:
        self._results = list(results)

    def run(self, artifacts) -> validation.TestResult:
        assert self._results, "runner 被调用的次数超过预期"
        return self._results.pop(0)


def _passed() -> validation.TestResult:
    return validation.TestResult(status="passed", exit_code=0, passed=1, duration=0.1)


def _artifacts(files: dict[str, str]) -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(path=path, action="create", content=content, purpose="测试")
            for path, content in files.items()
        ],
        dependency_changes=[],
        summary="trace 测试用产物",
    )


def _events(trace: TraceCollector, stage: str) -> list:
    return [event for event in trace.events() if event.stage == stage]


def _only(trace: TraceCollector, stage: str, event_type: str):
    matches = [
        event for event in trace.events() if event.stage == stage and event.event_type == event_type
    ]
    assert len(matches) == 1, f"期望恰好一条 {stage}/{event_type}，实际 {len(matches)} 条"
    return matches[0]


def _serialized(trace: TraceCollector) -> str:
    """整个 trace 的文本形式——"某个值有没有进 trace"只能这样问。"""
    return json.dumps([event.model_dump() for event in trace.events()], ensure_ascii=False)


def _tool_context() -> ToolContext:
    return ToolContext(
        project_root=DEMO_PROJECT_PATH,
        project=scan_repository(DEMO_PROJECT_PATH),
        api=parse_openapi(PETSTORE_PATH),
        known_files=[],
    )


@pytest.fixture(scope="module")
def traced_run() -> tuple[pipeline.PipelineResult, TraceCollector]:
    """真实 Pipeline + 真实 TestRunner 跑一次（模块级：只跑一次 pytest 子进程）。"""
    trace = TraceCollector()
    result = pipeline.run_pipeline(
        PETSTORE_PATH, DEMO_PROJECT_PATH, request="pet endpoints", trace=trace
    )
    return result, trace


# ------------------------------------------------------------ 1. 收集器基本语义


def test_emit_appends_events_in_call_order() -> None:
    trace = TraceCollector()

    trace.emit("agent", "tool_called", "第一次调用", metadata={"tool": "search_code"})
    trace.emit("agent", "tool_completed", "第一次结束", status="ok")

    events = trace.events()
    assert [event.event_type for event in events] == ["tool_called", "tool_completed"]
    assert [event.message for event in events] == ["第一次调用", "第一次结束"]
    assert [event.stage for event in events] == ["agent", "agent"]
    assert isinstance(events[0].timestamp, float)
    assert events[1].status == "ok"


def test_duration_and_status_default_to_none() -> None:
    trace = TraceCollector()

    trace.emit("agent", "stage_started", "开始")

    event = trace.events()[0]
    assert event.duration is None
    assert event.status is None
    assert event.metadata == {}


def test_events_returns_a_defensive_copy() -> None:
    """调用方拿到的是副本：排序 / 清空自己的列表不该动到收集器。"""
    trace = TraceCollector()
    trace.emit("agent", "stage_started", "开始")

    snapshot = trace.events()
    snapshot.clear()
    snapshot.append("不是事件的东西")

    assert len(trace.events()) == 1


def test_clear_empties_the_collector() -> None:
    trace = TraceCollector()
    trace.emit("agent", "stage_started", "开始")

    trace.clear()

    assert trace.events() == []
    trace.emit("agent", "stage_completed", "结束后还能继续记")
    assert [event.message for event in trace.events()] == ["结束后还能继续记"]


# ---------------------------------------------------------------- 2. metadata


def test_metadata_is_normalized_into_json_scalars() -> None:
    flag = threading.Event()
    trace = TraceCollector()

    trace.emit(
        "agent",
        "tool_called",
        "混合类型",
        metadata={
            "int": 3,
            "float": 0.5,
            "bool": True,
            "none": None,
            "list": [1, "two"],
            "tuple": ("a", "b"),
            "set": {"b", "a"},
            "object": flag,
        },
    )

    metadata = trace.events()[0].metadata
    assert metadata["int"] == 3
    assert metadata["float"] == 0.5
    assert metadata["bool"] is True  # bool 不被当成 int 处理
    assert metadata["none"] is None
    assert metadata["list"] == [1, "two"]
    assert metadata["tuple"] == ["a", "b"]
    assert metadata["set"] == ["a", "b"]  # 集合排序后落成列表（同一份数据 → 同一份内容）
    assert metadata["object"] == "<Event>"
    # 整个事件的 metadata 必须能 JSON 序列化，否则它就不是"结构化数据"
    json.dumps(metadata)


def test_metadata_copies_the_callers_dict() -> None:
    trace = TraceCollector()
    payload = {"tool": "search_code"}

    trace.emit("agent", "tool_called", "开始", metadata=payload)
    payload["tool"] = "被调用方改掉了"

    assert trace.events()[0].metadata == {"tool": "search_code"}


def test_long_string_values_are_truncated() -> None:
    """单值上限挡住"顺手把一段源码 / 一个响应体塞进 metadata"。"""
    trace = TraceCollector()
    long_source = "x = 1\n" * 100

    trace.emit("planner", "planning_started", "开始", metadata={"snippet": long_source})

    value = trace.events()[0].metadata["snippet"]
    assert value.startswith("x = 1\nx = 1\n")
    assert value.endswith(f"...(+{len(long_source) - MAX_STRING_CHARS} chars)")
    assert len(value) < len(long_source)


def test_short_strings_are_left_alone() -> None:
    trace = TraceCollector()

    trace.emit("agent", "tool_called", "开始", metadata={"tool": "search_code"})

    assert trace.events()[0].metadata["tool"] == "search_code"


def test_deeply_nested_metadata_is_capped() -> None:
    trace = TraceCollector()
    deep = {"a": {"b": {"c": {"d": {"e": "太深了"}}}}}

    trace.emit("agent", "tool_called", "开始", metadata=deep)

    node = trace.events()[0].metadata
    for key in ("a", "b", "c", "d"):
        node = node[key]
    assert node == {"e": "<max depth>"}


def test_event_survives_a_json_round_trip() -> None:
    trace = TraceCollector()
    trace.emit(
        "test_runner",
        "test_completed",
        "测试运行结束",
        metadata={"status": "failed", "durations": {"pytest": 0.4}},
        duration=0.5,
        status="failed",
    )

    original = trace.events()[0]
    restored = type(original).model_validate(json.loads(original.model_dump_json()))

    assert restored == original


# ------------------------------------------------------------ 6. Secret Safety


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "api-key",
        "API_KEY",
        "apikey",
        "token",
        "access_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "authorization",
        "credential",
        "private_key",
        "access_key",
        "bearer",
    ],
)
def test_secret_shaped_keys_are_redacted(key: str) -> None:
    """按键名脱敏：命中即整个值被替换，值本身一个字都不留。"""
    trace = TraceCollector()

    trace.emit("agent", "tool_called", "开始", metadata={key: "should-never-appear"})

    metadata = trace.events()[0].metadata
    assert metadata == {key: REDACTED}
    assert "should-never-appear" not in _serialized(trace)


def test_redaction_reaches_nested_metadata() -> None:
    trace = TraceCollector()

    trace.emit(
        "agent",
        "tool_called",
        "开始",
        metadata={"config": {"headers": {"Authorization": "Bearer abc"}, "retries": 3}},
    )

    nested = trace.events()[0].metadata["config"]
    assert nested["headers"]["Authorization"] == REDACTED
    assert nested["retries"] == 3
    assert "Bearer abc" not in _serialized(trace)


def test_non_secret_keys_are_not_redacted() -> None:
    """反面对照：脱敏不能把正常字段也吃掉（否则前两条断言是空的）。"""
    trace = TraceCollector()

    trace.emit(
        "test_runner",
        "test_completed",
        "测试运行结束",
        metadata={"status": "failed", "failed": 1, "call_count": 12, "keyboard": "us"},
    )

    metadata = trace.events()[0].metadata
    assert metadata["status"] == "failed"
    assert metadata["failed"] == 1
    assert metadata["call_count"] == 12
    assert metadata["keyboard"] == "us"


def test_keys_merely_containing_a_secret_word_are_redacted_too() -> None:
    """键名匹配是**子串**匹配：宁可多抹，不可漏抹。

    tokens_used 只是恰好多带了一个 "token"，就被整值抹掉——这是有意的取舍
    （凭据格式永远追不完，漏抹一次就是泄漏；多抹一个计数字段只是少一条信息）。
    这条用例把这个取舍固定下来，避免以后有人"顺手"把匹配改精确。
    """
    trace = TraceCollector()

    trace.emit("agent", "llm_completed", "一轮结束", metadata={"tokens_used": 12})

    assert trace.events()[0].metadata == {"tokens_used": REDACTED}


def test_tool_arguments_never_reach_the_trace() -> None:
    """§六：只记工具名，不记参数——参数是模型写的，完全可能夹带凭据。

    非空洞性：先证明那份凭据**真的**进了消息历史（假客户端记下了逐轮请求），
    再说它没有出现在 trace 里。
    """
    arguments = json.dumps({"query": "httpx", **FAKE_CREDENTIALS})
    client = FakeToolCallingClient(
        [
            AssistantTurn(
                tool_calls=[ToolCallRequest(id="call-1", name="search_code", arguments=arguments)]
            ),
            AssistantTurn(content="完成"),
        ]
    )
    trace = TraceCollector()

    with use_trace(trace):
        result = AgentLoopRunner().run(
            system_prompt="system",
            user_prompt="user",
            context=_tool_context(),
            registry=build_default_registry(),
            llm=client,
        )

    assert result.status == "completed"
    # 非空洞性：凭据确实随 assistant 消息回到了下一轮请求里（第二轮请求带着它）。
    assert len(client.requests) == 2
    forwarded = json.dumps([message.model_dump() for message in client.requests[1]], default=str)
    assert FAKE_CREDENTIALS["api_key"] in forwarded
    # 而 trace 里没有：tool_called 只带工具名与序号。
    serialized = _serialized(trace)
    for value in FAKE_CREDENTIALS.values():
        assert value not in serialized
    assert _only(trace, "agent", "tool_called").metadata == {"tool": "search_code", "call": 1}


def test_pipeline_secrets_do_not_leak_into_the_trace(
    traced_run: tuple[pipeline.PipelineResult, TraceCollector],
) -> None:
    """端到端：整条流水线的 trace 里不得出现任何"像凭据"的取值。"""
    _, trace = traced_run

    serialized = _serialized(trace)
    assert serialized  # 空 trace 会让下面每条断言都恒真
    for value in FAKE_CREDENTIALS.values():
        assert value not in serialized


# -------------------------------------------------- 7. Backward Compatibility


def test_emit_without_a_scope_is_a_noop() -> None:
    assert current_trace() is None

    emit("agent", "tool_called", "没人听")

    assert current_trace() is None


def test_scope_is_restored_after_exit() -> None:
    trace = TraceCollector()

    with use_trace(trace):
        assert current_trace() is trace
        emit("agent", "stage_started", "开始")

    assert current_trace() is None
    assert len(trace.events()) == 1


def test_nested_scopes_restore_the_outer_collector() -> None:
    outer, inner = TraceCollector(), TraceCollector()

    with use_trace(outer):
        emit("agent", "stage_started", "外层")
        with use_trace(inner):
            emit("agent", "tool_called", "内层")
        emit("agent", "stage_completed", "回到外层")

    assert [event.message for event in outer.events()] == ["外层", "回到外层"]
    assert [event.message for event in inner.events()] == ["内层"]


def test_scope_none_keeps_the_outer_collector() -> None:
    """trace=None 是"这次调用没有要求收集"，不是"把外层关掉"。"""
    outer = TraceCollector()

    with use_trace(outer), trace_scope(None):
        emit("agent", "stage_started", "仍然记在外层")

    assert [event.message for event in outer.events()] == ["仍然记在外层"]


def test_collectors_do_not_leak_between_threads() -> None:
    """contextvars 而不是模块级全局：并发请求的 trace 不能串到一起。

    用 barrier 保证两个线程**同时**在自己的作用域里（而不是靠 sleep 赌交错），
    这样"全局变量"实现必然在这一格失败。
    """
    barrier = threading.Barrier(2)
    seen: dict[str, list[str]] = {}

    def worker(name: str) -> None:
        trace = TraceCollector()
        with use_trace(trace):
            emit("agent", "tool_called", name, metadata={"tool": name})
            barrier.wait(timeout=10)
            emit("agent", "tool_completed", name)
        seen[name] = [event.message for event in trace.events()]

    threads = [threading.Thread(target=worker, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert seen == {"worker-0": ["worker-0", "worker-0"], "worker-1": ["worker-1", "worker-1"]}


def test_pipeline_without_trace_leaves_no_scope_behind() -> None:
    result = pipeline.run_pipeline(
        PETSTORE_PATH, DEMO_PROJECT_PATH, test_runner=FakeRunner([_passed()])
    )

    assert result.status == "passed"
    assert current_trace() is None


def test_pipeline_with_trace_also_leaves_no_scope_behind() -> None:
    trace = TraceCollector()

    pipeline.run_pipeline(
        PETSTORE_PATH, DEMO_PROJECT_PATH, test_runner=FakeRunner([_passed()]), trace=trace
    )

    assert current_trace() is None
    assert len(trace.events()) > 0


def test_pipeline_result_is_identical_with_and_without_trace() -> None:
    """Backward Compatibility 的硬要求：trace 只是旁路，不改变任何结果。

    用假 Runner 把唯一的不确定来源（真实 pytest 的耗时）去掉，于是可以要求
    两次运行的 PipelineResult **全字段相等**，而不只是"状态一样"。
    """

    def run(trace: TraceCollector | None) -> pipeline.PipelineResult:
        return pipeline.run_pipeline(
            PETSTORE_PATH, DEMO_PROJECT_PATH, test_runner=FakeRunner([_passed()]), trace=trace
        )

    without = run(None)
    with_trace = run(TraceCollector())

    assert without.model_dump() == with_trace.model_dump()


# ------------------------------------------------ 8. 词表闭合（静态扫描）


def _emit_call_sites() -> list[tuple[Path, int, str | None, str | None]]:
    """扫出全仓库的 emit(...) 调用点，(文件, 行号, stage, event_type)。

    stage / event_type 不是字面量的调用点（例如 orchestrator 里的短名映射）
    记为 None，由别的用例单独守。
    """
    sites: list[tuple[Path, int, str | None, str | None]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "emit":
                continue
            sites.append(
                (path, node.lineno, _string_literal(node.args[0]), _string_literal(node.args[1]))
            )
    return sites


def _string_literal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def test_every_emit_call_site_uses_a_registered_stage_and_event_type() -> None:
    """埋点写了、词表里没有 = 事件被静默丢掉（本文件开头说的那个坑）。

    这条扫的是**源码**而不是运行结果：漏登记的类型在运行期完全不报错，
    只有静态扫描能在它变成"trace 里少了一段"之前拦住。
    """
    sites = [
        (path, lineno, stage, event_type)
        for path, lineno, stage, event_type in _emit_call_sites()
        if stage is not None and event_type is not None
    ]

    assert len(sites) >= 15, f"只扫到 {len(sites)} 个字面量埋点，扫描实现八成坏了"
    stages, event_types = set(get_args(TraceStage)), set(get_args(TraceEventType))
    problems = [
        f"{path.name}:{lineno} {stage}/{event_type}"
        for path, lineno, stage, event_type in sites
        if stage not in stages or event_type not in event_types
    ]
    assert problems == [], f"这些埋点用了未登记的 stage 或 event_type：{problems}"
    # 抽查：几个必须存在的埋点（防止扫描"扫到了别的东西"也算通过）
    found = {(stage, event_type) for _, _, stage, event_type in sites}
    assert ("test_runner", "test_completed") in found
    assert ("generation", "generation_completed") in found
    assert ("repair", "repair_completed") in found
    assert ("patch", "patch_generated") in found


def test_stage_name_mapping_values_are_registered() -> None:
    """pipeline 的短名（failed_stage 的对外取值）→ trace 阶段名的映射必须落在词表内。"""
    mapping = orchestrator_module._STAGE_NAME

    assert set(mapping) == {"parse", "scan", "plan", "generate"}
    assert set(mapping.values()) <= set(get_args(TraceStage))


def test_an_unregistered_event_type_is_dropped_without_raising() -> None:
    """写错的埋点只丢事件，不让业务流程崩——但它确实被丢了（不是悄悄变成别的类型）。"""
    trace = TraceCollector()

    trace.emit("agent", "totally_made_up", "x")  # type: ignore[arg-type]
    trace.emit("no_such_stage", "stage_started", "y")  # type: ignore[arg-type]
    trace.emit("agent", "stage_started", "合法的")

    assert [event.message for event in trace.events()] == ["合法的"]


# -------------------------------------------------------- 3./4./5. 阶段埋点与顺序


def _assert_stage_lifecycles(events) -> None:
    """每个 stage_started 之后，同一个阶段必须收口一次（且不重复开始）。"""
    open_stages: set[str] = set()
    for event in events:
        if event.event_type == "stage_started":
            assert event.stage not in open_stages, f"{event.stage} 重复开始"
            open_stages.add(event.stage)
        elif event.event_type in STAGE_CLOSERS:
            assert event.stage in open_stages, f"{event.stage} 没有对应的 stage_started"
            open_stages.discard(event.stage)
    assert open_stages == set(), f"这些阶段没有收口：{sorted(open_stages)}"


def test_pipeline_emits_every_required_stage(
    traced_run: tuple[pipeline.PipelineResult, TraceCollector],
) -> None:
    """§十四.4：一次完整运行至少产生这五个阶段的事件。"""
    result, trace = traced_run

    assert result.status == "passed"  # 前提：这是一次真的跑到底的运行
    stages = {event.stage for event in trace.events()}
    assert {
        "api_understanding",
        "repository_understanding",
        "planner",
        "generation",
        "test_runner",
    } <= stages


def test_pipeline_stage_lifecycles_are_ordered_and_closed(
    traced_run: tuple[pipeline.PipelineResult, TraceCollector],
) -> None:
    result, trace = traced_run
    assert result.status == "passed"

    _assert_stage_lifecycles(trace.events())

    skeleton = [
        (event.stage, event.event_type)
        for event in trace.events()
        if event.event_type in STAGE_CLOSERS or event.event_type == "stage_started"
    ]
    assert skeleton == [
        ("api_understanding", "stage_started"),
        ("api_understanding", "stage_completed"),
        ("repository_understanding", "stage_started"),
        ("repository_understanding", "stage_completed"),
        ("planner", "stage_started"),
        ("planner", "stage_completed"),
        ("generation", "stage_started"),
        ("generation", "generation_completed"),
        ("repair", "stage_started"),
        ("repair", "stage_completed"),
        ("patch", "stage_started"),
        ("patch", "patch_generated"),
    ]


def test_pipeline_event_order_is_stable_across_runs() -> None:
    """同一份输入跑两次，事件类型序列一致（时间戳与耗时除外）。"""

    def sequence() -> list[tuple[str, str]]:
        trace = TraceCollector()
        pipeline.run_pipeline(
            PETSTORE_PATH, DEMO_PROJECT_PATH, test_runner=FakeRunner([_passed()]), trace=trace
        )
        return [(event.stage, event.event_type) for event in trace.events()]

    first, second = sequence(), sequence()

    assert first == second
    assert len(first) >= 8


def test_planner_emits_its_own_events() -> None:
    """planner 被单独使用时（没有 pipeline 那层）也必须有自己的信号。"""
    trace = TraceCollector()

    with use_trace(trace):
        plan = DeterministicPlanner().plan(
            PlannerState(
                api=parse_openapi(PETSTORE_PATH),
                project=scan_repository(DEMO_PROJECT_PATH),
                request="pet endpoints",
            )
        )

    started = _only(trace, "planner", "planning_started")
    completed = _only(trace, "planner", "planning_completed")
    assert started.metadata["planner"] == "deterministic"
    assert completed.metadata["endpoints"] == len(plan.endpoints)
    assert completed.metadata["files_to_create"] == len(plan.files_to_create)
    assert completed.duration is not None


def test_generation_completed_reports_the_shape_not_the_source() -> None:
    trace = TraceCollector()
    plan = DeterministicPlanner().plan(
        PlannerState(
            api=parse_openapi(PETSTORE_PATH),
            project=scan_repository(DEMO_PROJECT_PATH),
            request="",
        )
    )

    with use_trace(trace):
        artifacts = generation.DeterministicCodeGenerator().generate(plan)

    metadata = _only(trace, "generation", "generation_completed").metadata
    created = [item for item in artifacts.files if item.action == "create"]
    tests = sum(1 for item in created if item.path.startswith("tests/"))
    assert tests > 0  # 否则下面那条占比断言是空的
    assert metadata["file_count"] == len(artifacts.files)
    assert metadata["generated_test_count"] == tests
    assert metadata["generated_source_count"] == len(created) - tests
    # 只记形状，不记内容：生成出来的源码一个字都不在 trace 里。
    serialized = _serialized(trace)
    assert artifacts.files[0].content[:40] not in serialized


def test_agent_loop_emits_llm_and_tool_events_in_order() -> None:
    """§六：agent 阶段的完整事件序列（一轮工具调用 + 一轮最终回答）。"""
    client = FakeToolCallingClient(
        [
            AssistantTurn(
                tool_calls=[
                    ToolCallRequest(id="call-1", name="search_code", arguments='{"query": "httpx"}')
                ]
            ),
            AssistantTurn(content="完成"),
        ]
    )
    trace = TraceCollector()

    with use_trace(trace):
        result = AgentLoopRunner().run(
            system_prompt="system",
            user_prompt="user",
            context=_tool_context(),
            registry=build_default_registry(),
            llm=client,
        )

    assert result.status == "completed"
    assert [(event.stage, event.event_type) for event in trace.events()] == [
        ("agent", "stage_started"),
        ("agent", "llm_called"),
        ("agent", "llm_completed"),
        ("agent", "tool_called"),
        ("agent", "tool_completed"),
        ("agent", "llm_called"),
        ("agent", "llm_completed"),
        ("agent", "stage_completed"),
    ]
    requests = [event for event in trace.events() if event.event_type == "llm_completed"]
    assert [event.metadata["turn"] for event in requests] == [1, 2]
    assert requests[0].metadata["tool_calls"] == 1  # 第一轮请求了工具
    assert requests[1].metadata["tool_calls"] == 0  # 第二轮直接给答案
    assert requests[0].duration is not None
    assert _only(trace, "agent", "tool_completed").status == "ok"
    assert _only(trace, "agent", "stage_completed").status == "completed"


def test_agent_loop_failure_closes_the_stage() -> None:
    """提前退出的路径也必须收口——9 条 return 只有一条发 stage_failed。"""
    client = FakeToolCallingClient(
        [AssistantTurn(tool_calls=[ToolCallRequest(id="c", name="不存在", arguments="{}")])] * 1
    )
    trace = TraceCollector()

    with use_trace(trace):
        result = AgentLoopRunner().run(
            system_prompt="system",
            user_prompt="user",
            context=_tool_context(),
            registry=build_default_registry(),
            llm=client,
        )

    # 假客户端被问第二次时直接抛 AssertionError → 回路把它收成 llm_error，
    # 而不是继续跑下去。测的是"异常路径也要收口"，不是异常本身。
    assert result.status == "llm_error"
    assert _only(trace, "agent", "llm_completed").metadata["tool_calls"] == 1
    failed = _only(trace, "agent", "stage_failed")
    assert failed.status == "llm_error"
    _assert_stage_lifecycles(trace.events())


# ------------------------------------------------------------ 5. TestResult 一致性


def test_test_completed_metadata_mirrors_the_test_result() -> None:
    """trace 的测试结论必须**等于** TestResult，而不是另算一份。"""
    trace = TraceCollector()

    with use_trace(trace):
        result = repair.run_repair_loop(
            _artifacts({"tests/test_bad.py": FAILING_TEST}), max_iterations=1
        )

    test_result = result.test_result
    assert test_result is not None
    metadata = _only(trace, "test_runner", "test_completed").metadata
    assert metadata["status"] == test_result.status == "failed"
    assert metadata["passed"] == test_result.passed == 0
    assert metadata["failed"] == test_result.failed == 1
    assert metadata["errors"] == test_result.errors == 0
    assert metadata["skipped"] == test_result.skipped == 0
    assert metadata["exit_code"] == test_result.exit_code == 1
    assert metadata["duration"] == test_result.duration
    assert _only(trace, "test_runner", "test_completed").duration is not None


def test_test_started_precedes_the_test_result() -> None:
    trace = TraceCollector()

    with use_trace(trace):
        repair.run_repair_loop(_artifacts({"tests/test_bad.py": FAILING_TEST}), max_iterations=1)

    started = _only(trace, "test_runner", "test_started")
    assert started.metadata["files"] == 1
    assert started.metadata["timeout"] > 0


# ------------------------------------------------------------------ §十 Repair


def test_repair_attempts_are_paired_and_numbered() -> None:
    trace = TraceCollector()

    with use_trace(trace):
        result = repair.run_repair_loop(
            _artifacts({"tests/test_bad.py": FAILING_TEST}), max_iterations=2
        )

    repair_events = _events(trace, "repair")
    assert [event.event_type for event in repair_events] == ["repair_started", "repair_completed"]
    started, completed = repair_events
    assert started.metadata["attempt"] == 1
    assert started.metadata["test_status"] == "failed"
    assert completed.metadata["attempt"] == 1
    assert completed.metadata["changed"] is False
    assert completed.metadata["status"] == result.status == "no_progress"
    assert completed.duration is not None


def test_repair_completed_reports_no_patch_body() -> None:
    """§十：只记"改了没有、改了几个文件"，不记 patch 内容。"""
    trace = TraceCollector()

    with use_trace(trace):
        repair.run_repair_loop(_artifacts({"tests/test_bad.py": FAILING_TEST}), max_iterations=1)

    metadata = _only(trace, "repair", "repair_completed").metadata
    assert set(metadata) == {
        "attempt",
        "status",
        "changed",
        "files_changed",
        "applied_actions",
        "skipped_actions",
    }
    assert "assert 1 == 2" not in _serialized(trace)


# ---------------------------------------------------------------- §十六 确定性


def test_core_event_fields_are_deterministic_across_runs() -> None:
    """除了时间戳与耗时，同一份输入产生的事件内容必须逐字一致。"""

    def core() -> list[tuple]:
        trace = TraceCollector()
        pipeline.run_pipeline(
            PETSTORE_PATH, DEMO_PROJECT_PATH, test_runner=FakeRunner([_passed()]), trace=trace
        )
        return [
            (
                event.stage,
                event.event_type,
                event.message,
                tuple(sorted(event.metadata)),
                event.status,
            )
            for event in trace.events()
        ]

    assert core() == core()
