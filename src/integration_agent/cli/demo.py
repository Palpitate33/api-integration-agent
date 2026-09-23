"""CLI Demo 的运行与"可展示视图"构建（不含渲染）。

三段分工：

    run_demo()      装配 Pipeline + TraceCollector，跑一次**真实**集成
    build_report()  把 Trace 事件 + 最终状态整理成 DemoReport（纯函数）
    render.py       把 DemoReport 画成终端文本

为什么中间要有 DemoReport 这一层
--------------------------------
CLI 是**展示层**，它不该自己重新推断"现在到哪个阶段了""测试通过没有""Repair 干了
什么"——那些问题的答案已经在 Trace 里了。所以这里把事件读成一个结构化的视图对象，
渲染层再照着画：

    Pipeline → TraceCollector → build_report() → render_report() → stdout

唯一的例外是最终状态（SUCCESS / FAILED）：按约定它直接取自 ``PipelineResult.status``，
不由 CLI 按事件猜。

关于 Demo 数据
--------------
固定复用项目里已有的那对 fixture（examples/openapi/petstore.yaml + examples/demo_project），
常量直接从 api_server.demo 引用而不是复制一份：那对路径在那边已经被定义为"唯一的
固定 Demo 组合"，再写一遍就是第二个事实来源。``--sabotage`` 用的 SabotagedGenerator
同理（scripts/e2e_demo.py 也是这么复用的）。

**本模块不接受任何用户指定的 spec / repo 路径**：SabotagedGenerator 的注入逻辑只对
那对固定组合做过校验，接受外部路径等于把"demo 注入"变成一个可以指向任意文件的开关。
这也是 Demo 可重复的前提——输入固定，输出才固定。

可重复性
--------
同一份 fixture 连跑两次，输出必须逐行相同（duration 允许不同）。因此视图里不放
临时路径、时间戳、随机 ID；仓库 root 与 spec 的 base_url 本来就没有进 trace
（见 trace.py 的埋点说明），本模块也不额外引入。
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from integration_agent.api_server.demo import (
    DEMO_PROJECT,
    DEMO_PROJECT_PATH,
    DEMO_SPEC,
    DEMO_SPEC_PATH,
    SabotagedGenerator,
)
from integration_agent.pipeline import PipelineResult, run_pipeline
from integration_agent.trace import TraceCollector, TraceEvent

# 默认的集成请求：与 scripts/e2e_demo.py 保持一致，便于两处演示互相印证。
DEFAULT_REQUEST = "集成 Petstore 的查询与创建接口"

# Demo 展示的 8 个阶段。顺序就是流水线的执行顺序；agent 排在 planner 之前，
# 因为 ToolUsingPlanner 用的是同一条 Agent Loop（回路先跑完，计划才出来）。
DEMO_STAGES: tuple[tuple[str, str], ...] = (
    ("api_understanding", "API Understanding"),
    ("repository_understanding", "Repository Understanding"),
    ("agent", "Agent"),
    ("planner", "Integration Planner"),
    ("generation", "Code Generator"),
    ("test_runner", "Test Runner"),
    ("repair", "Repair Loop"),
    ("patch", "Final Patch"),
)

# 阶段收口事件：出现其中任意一个，这个阶段就算跑完了。generation / test_runner /
# patch 三个阶段用的是各自的完成信号（见 trace.py 的说明），不额外发 stage_completed。
STAGE_CLOSERS = frozenset(
    {"stage_completed", "stage_failed", "generation_completed", "test_completed", "patch_generated"}
)

# 三个展示状态 + 一个"开始了但没收到结果"（异常路径）。
STAGE_COMPLETED = "completed"
STAGE_FAILED = "failed"
STAGE_SKIPPED = "skipped"
STAGE_INCOMPLETE = "incomplete"

# 退出码：passed=0 是唯一"成功"；tests_failed=1 表示流程走完但集成没通过
# （这是真实结论，不是 CLI 出错）；error=2 表示流程自身异常。
EXIT_CODES = {"passed": 0, "tests_failed": 1, "error": 2}


@dataclass(frozen=True)
class ToolCallView:
    """一次工具调用：只留名字与成败，参数与结果原文一概不留。"""

    name: str
    ok: bool | None  # None = 回路发出了调用但没等到结果（中途退出）


@dataclass(frozen=True)
class StageView:
    """一个阶段的一行结论 + 若干细节行。"""

    index: int
    title: str
    status: str
    headline: str
    details: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TestView:
    """Test Runner 的结论——字段与 TestResult 一一对应，不做二次推算。"""

    status: str
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    exit_code: int | None


@dataclass(frozen=True)
class RepairAttemptView:
    """一次修复尝试：只留"改了没有、改了几个文件"，不留 patch 内容。"""

    attempt: int
    status: str
    changed: bool
    files_changed: int


@dataclass(frozen=True)
class PatchView:
    """最终产物概览。deleted 恒为 0：PatchFile.action 结构上只有 create / modify。"""

    files_changed: int
    added: int
    modified: int
    deleted: int


@dataclass(frozen=True)
class DemoReport:
    """一次 Demo 运行的全部可展示内容（渲染层的唯一输入）。"""

    spec_label: str
    project_label: str
    request: str
    mode: str
    stages: list[StageView]
    tools: list[ToolCallView]
    test: TestView | None
    repairs: list[RepairAttemptView]
    patch: PatchView | None
    warnings: list[str]
    status: str
    failed_stage: str | None


@dataclass(frozen=True)
class DemoOutcome:
    """run_demo() 的返回：报告 + 原始结果（测试要拿它们做断言）。"""

    report: DemoReport
    result: PipelineResult
    trace: TraceCollector

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.result.status, 2)


def run_demo(
    *,
    request: str = DEFAULT_REQUEST,
    max_iterations: int = 3,
    sabotage: bool = False,
) -> DemoOutcome:
    """跑一次真实的端到端集成，并把过程收集成 trace。

    sabotage=True 时换用项目已有的 SabotagedGenerator（把生成产物里的一个断言改错），
    让测试**真的**失败，从而真的走到 Repair Loop——注入的是失败，不是结论：
    后面出现的一切仍然由 Pipeline 与 TestRunner 真实产生。
    """
    trace = TraceCollector()
    extra: dict[str, Any] = {}
    if sabotage:
        extra["code_generator"] = SabotagedGenerator()
    result = run_pipeline(
        DEMO_SPEC_PATH,
        DEMO_PROJECT_PATH,
        request=request,
        max_iterations=max_iterations,
        trace=trace,
        **extra,
    )
    report = build_report(
        trace.events(),
        status=result.status,
        failed_stage=result.failed_stage,
        warnings=result.warnings,
        request=request,
        mode="sabotage（注入确定性失败）" if sabotage else "deterministic（离线）",
    )
    return DemoOutcome(report=report, result=result, trace=trace)


def build_report(
    events: Sequence[TraceEvent],
    *,
    status: str,
    failed_stage: str | None = None,
    warnings: Iterable[str] = (),
    request: str = "",
    mode: str = "",
    spec_label: str | None = None,
    project_label: str | None = None,
) -> DemoReport:
    """把一次运行的 trace 事件整理成可展示视图（纯函数，不跑任何东西）。

    status / failed_stage / warnings 三个参数来自 PipelineResult——它们是 pipeline
    的结论，不是从事件里推出来的。
    """
    calls = list(events)
    return DemoReport(
        spec_label=spec_label if spec_label is not None else f"examples/{DEMO_SPEC}",
        project_label=project_label if project_label is not None else f"examples/{DEMO_PROJECT}",
        request=request,
        mode=mode,
        stages=_stage_views(calls),
        tools=_tool_views(calls),
        test=_test_view(calls),
        repairs=_repair_views(calls),
        patch=_patch_view(calls),
        warnings=list(warnings),
        status=status,
        failed_stage=failed_stage,
    )


# ------------------------------------------------------------------ 阶段视图


def _stage_views(events: Sequence[TraceEvent]) -> list[StageView]:
    views: list[StageView] = []
    for index, (key, title) in enumerate(DEMO_STAGES, start=1):
        mine = [event for event in events if event.stage == key]
        views.append(_stage_view(index, title, key, mine, events))
    return views


def _stage_view(
    index: int,
    title: str,
    key: str,
    mine: Sequence[TraceEvent],
    all_events: Sequence[TraceEvent],
) -> StageView:
    if not mine:
        # 这次运行根本没走到这个阶段。如实说"没用上"，不画一个空的 ✓。
        return StageView(index, title, STAGE_SKIPPED, "not used in this run")
    types = {event.event_type for event in mine}
    if "stage_failed" in types:
        status = STAGE_FAILED
    elif types & STAGE_CLOSERS:
        status = STAGE_COMPLETED
    else:
        status = STAGE_INCOMPLETE  # 只发了 stage_started
    # 两处按阶段的修正——都是"阶段跑完了，但它的结论不是成功"：
    #   test_runner：这一格的标记表示**测试结论**，不是"测试进程有没有跑起来"。
    #                只有全绿才画 ✓；status=error（测试根本没跑起来）、failed、
    #                no_tests 一律 ✗。具体是什么状态由下面那块 Test Runner 说明。
    #   repair：没有任何一次修复尝试 = 这轮没有活要干，照实标成 skipped，
    #           而不是画一个 ✓ 再说一句 "skipped: tests passed"。
    if key == "test_runner":
        completed = _first(mine, "test_completed")
        if completed is not None:
            passed = _metadata(completed).get("status") == "passed"
            status = STAGE_COMPLETED if passed else STAGE_FAILED
    elif key == "repair" and _first(mine, "repair_started") is None:
        status = STAGE_SKIPPED
    headline, details = _SUMMARISERS[key](mine, all_events)
    return StageView(index, title, status, headline, details)


def _metadata(event: TraceEvent) -> dict[str, Any]:
    return dict(event.metadata or {})


def _first(events: Sequence[TraceEvent], event_type: str) -> TraceEvent | None:
    for event in events:
        if event.event_type == event_type:
            return event
    return None


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _summarise_api(mine, _all) -> tuple[str, list[str]]:
    completed = _first(mine, "stage_completed")
    failed = _first(mine, "stage_failed")
    if completed is None:
        return ("API 解析失败" if failed else "无结论"), _failed_details(failed)
    return _plural(int(_metadata(completed).get("endpoints", 0)), "endpoint"), []


def _summarise_repository(mine, _all) -> tuple[str, list[str]]:
    completed = _first(mine, "stage_completed")
    failed = _first(mine, "stage_failed")
    if completed is None:
        return ("仓库扫描失败" if failed else "无结论"), _failed_details(failed)
    data = _metadata(completed)
    kind = "Python project" if data.get("is_python_project") else "not a Python project"
    facts = [
        _plural(int(data.get("python_files", 0)), "Python file"),
        _plural(int(data.get("source_dirs", 0)), "source dir"),
        _plural(int(data.get("dependencies", 0)), "dependency"),
    ]
    return f"{kind} · " + " · ".join(facts), []


def _summarise_agent(mine, _all) -> tuple[str, list[str]]:
    completed = _first(mine, "stage_completed")
    failed = _first(mine, "stage_failed")
    if completed is not None:
        calls = int(_metadata(completed).get("tool_calls", 0))
        return _plural(calls, "tool call"), []
    if failed is not None:
        return f"回路失败：{failed.status or 'unknown'}", _failed_details(failed)
    return "回路未完成", []


def _summarise_planner(mine, _all) -> tuple[str, list[str]]:
    # planning_completed 是规划器自己发的（更细）；没有它时退回 pipeline 的 stage_completed。
    completed = _first(mine, "planning_completed") or _first(mine, "stage_completed")
    failed = _first(mine, "stage_failed")
    if completed is None:
        return ("规划失败" if failed else "无结论"), _failed_details(failed)
    data = _metadata(completed)
    details: list[str] = []
    if data.get("planner"):
        details.append(f"planner: {data['planner']}")
    if "files_to_create" in data:
        details.append(
            f"{_plural(int(data['files_to_create']), 'file')} to create · "
            f"{_plural(int(data.get('files_to_modify', 0)), 'file')} to modify"
        )
    if data.get("agent_status"):
        # Agent 回路的自述（turns / tool_calls / 状态）：计划看着正常、回路其实一路
        # 降级，是这类 Planner 最需要被看见的情况。
        details.append(
            f"agent loop: {data['agent_status']} · {data.get('turns', 0)} turns · "
            f"{data.get('tool_calls', 0)} tool calls"
        )
    return _plural(int(data.get("endpoints", 0)), "endpoint") + " planned", details


def _summarise_generation(mine, _all) -> tuple[str, list[str]]:
    completed = _first(mine, "generation_completed")
    failed = _first(mine, "stage_failed")
    if completed is None:
        return ("生成失败" if failed else "无结论"), _failed_details(failed)
    data = _metadata(completed)
    details = [
        f"{data.get('generated_source_count', 0)} source · "
        f"{data.get('generated_test_count', 0)} test files · "
        f"{data.get('modified_count', 0)} modified"
    ]
    if data.get("dependency_changes"):
        details.append(_plural(int(data["dependency_changes"]), "dependency change"))
    return _plural(int(data.get("file_count", 0)), "file") + " generated", details


def _summarise_test_runner(mine, _all) -> tuple[str, list[str]]:
    completed = _first(mine, "test_completed")
    if completed is None:
        return "no test result recorded", []
    data = _metadata(completed)
    headline = (
        f"{data.get('passed', 0)} passed / {data.get('failed', 0)} failed"
        f" / {data.get('errors', 0)} errors"
    )
    return headline, []


def _summarise_repair(mine, all_events) -> tuple[str, list[str]]:
    started = _first(mine, "repair_started")
    if started is None:
        # 没有修复尝试。原因是测试直接过了，还是有别的状况——照实说。
        test = _first(all_events, "test_completed")
        if test is not None and _metadata(test).get("status") == "passed":
            return "skipped: tests passed", []
        return "no repair attempt recorded", []
    attempts = _repair_views(all_events)
    if not attempts:
        # 发起了修复但没有收到结果（回路中途退出）：不凭空补一个结论。
        return "repair started, no result recorded", []
    if len(attempts) == 1:
        # 单次尝试：结论就是这一行，细节在下面的 Repair Loop 块里，这里不再重复一遍。
        return f"attempt {attempts[0].attempt}", []
    return f"{len(attempts)} attempts", [
        f"attempt {item.attempt}: {item.status} · {_plural(item.files_changed, 'file')} changed"
        for item in attempts
    ]


def _summarise_patch(mine, _all) -> tuple[str, list[str]]:
    completed = _first(mine, "patch_generated")
    failed = _first(mine, "stage_failed")
    if completed is None:
        return ("Patch 生成失败" if failed else "无结论"), _failed_details(failed)
    return _plural(int(_metadata(completed).get("files_changed", 0)), "file") + " changed", []


def _failed_details(failed: TraceEvent | None) -> list[str]:
    """阶段失败的细节：只报"哪个阶段失败"，异常原文留在服务端日志。"""
    if failed is None:
        return []
    stage = _metadata(failed).get("failed_stage")
    return [f"stage failed: {stage}"] if stage else []


_SUMMARISERS = {
    "api_understanding": _summarise_api,
    "repository_understanding": _summarise_repository,
    "agent": _summarise_agent,
    "planner": _summarise_planner,
    "generation": _summarise_generation,
    "test_runner": _summarise_test_runner,
    "repair": _summarise_repair,
    "patch": _summarise_patch,
}


# ------------------------------------------------------------- 详情块视图


def _tool_views(events: Sequence[TraceEvent]) -> list[ToolCallView]:
    """按 tool_called 的顺序列出工具名，成败取自同一次调用的 tool_completed。

    以调用（而不是以结果）为准：回路中途退出时，最后那次调用没有结果，但它确实
    发生过——按结果列会把它悄悄抹掉。
    """
    outcomes = {
        event.metadata.get("call"): bool(event.metadata.get("ok"))
        for event in events
        if event.event_type == "tool_completed"
    }
    views: list[ToolCallView] = []
    for event in events:
        if event.event_type != "tool_called":
            continue
        views.append(
            ToolCallView(
                name=str(event.metadata.get("tool", "unknown")),
                ok=outcomes.get(event.metadata.get("call")),
            )
        )
    return views


def _test_view(events: Sequence[TraceEvent]) -> TestView | None:
    completed = _first(events, "test_completed")
    if completed is None:
        return None
    data = _metadata(completed)
    return TestView(
        status=str(data.get("status", "unknown")),
        passed=int(data.get("passed", 0)),
        failed=int(data.get("failed", 0)),
        errors=int(data.get("errors", 0)),
        skipped=int(data.get("skipped", 0)),
        duration=float(data.get("duration", 0.0)),
        exit_code=data.get("exit_code"),
    )


def _repair_views(events: Sequence[TraceEvent]) -> list[RepairAttemptView]:
    return [
        RepairAttemptView(
            attempt=int(event.metadata.get("attempt", 0)),
            status=str(event.metadata.get("status", "unknown")),
            changed=bool(event.metadata.get("changed")),
            files_changed=int(event.metadata.get("files_changed", 0)),
        )
        for event in events
        if event.event_type == "repair_completed"
    ]


def _patch_view(events: Sequence[TraceEvent]) -> PatchView | None:
    generated = _first(events, "patch_generated")
    if generated is None:
        return None
    data = _metadata(generated)
    return PatchView(
        files_changed=int(data.get("files_changed", 0)),
        added=int(data.get("added", 0)),
        modified=int(data.get("modified", 0)),
        deleted=int(data.get("deleted", 0)),
    )


__all__ = [
    "DEFAULT_REQUEST",
    "DEMO_STAGES",
    "DemoOutcome",
    "DemoReport",
    "PatchView",
    "RepairAttemptView",
    "StageView",
    "TestView",
    "ToolCallView",
    "build_report",
    "run_demo",
]
