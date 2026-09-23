"""CLI Demo 测试。

覆盖要求逐条对应：

    1. CLI 可以启动             → 子进程真实运行 python -m integration_agent demo
    2. 成功返回 exit code        → 真运行 0；失败/异常 1/2（main() 退出码映射）
    3. 输出包含 APIForge         → 真运行 + 合成报告
    4. 输出包含 API Understanding
    5. 输出包含 Test Runner
    6. 输出包含最终结果          → SUCCESS / FAILED 横幅
    7. Trace events 正确渲染     → 合成 trace：阶段编号 / 工具名 / 测试字段 / 修复尝试 / patch
    8. --no-color 无 ANSI        → 两个方向都断言（color=True 必须有 ANSI，否则第 8 条是空断言）
    9. 失败 Trace 显示 FAILED    → 合成失败 trace + 真实的 --sabotage 运行
   10. 不泄露 secret            → DEEPSEEK_API_KEY 哨兵 + 绝对路径检查
   11. 不修改真实仓库           → 运行前后对 examples/ 全量快照比对
   14. 可重复                   → 同一 fixture 连跑两次，渲染结果逐行相同（duration 除外）

做法：绝大多数用例跑在**合成 trace** 上（毫秒级，且能精确构造"失败 / 带工具调用 /
多次修复"这些真实运行里不好凑的场景），只有 3 处真的跑 pipeline（默认路径 /
--sabotage / 可重复性）——整条 pipeline 一次约 1.5s，而它本身已被
tests/test_pipeline.py 与 tests/test_trace.py 覆盖。
"""

import contextlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from integration_agent import cli
from integration_agent.cli.demo import DEMO_STAGES, build_report, run_demo
from integration_agent.cli.render import Palette, render_report
from integration_agent.pipeline import PipelineResult
from integration_agent.trace import TraceCollector, TraceEvent

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

# 哨兵：真的存在于环境里，且任何一次输出里都不该出现。
API_KEY_SENTINEL = "sk-live-SENTINEL-DO-NOT-PRINT-0123456789"


def _event(stage: str, event_type: str, **metadata) -> TraceEvent:
    return TraceEvent(
        timestamp=1_700_000_000.0,
        stage=stage,
        event_type=event_type,
        message=f"{stage}/{event_type}",
        metadata=metadata,
    )


def _trace(
    *,
    test_status: str = "passed",
    with_tools: bool = True,
    with_repair: bool = False,
) -> list[TraceEvent]:
    """合成一次运行的事件流——形状与真实埋点一致（字段名照抄 trace 输出）。"""
    events = [
        _event("api_understanding", "stage_started"),
        _event("api_understanding", "stage_completed", endpoints=3),
        _event("repository_understanding", "stage_started"),
        _event(
            "repository_understanding",
            "stage_completed",
            is_python_project=True,
            python_files=2,
            source_dirs=1,
            test_dirs=0,
            dependencies=1,
        ),
    ]
    if with_tools:
        events += [
            _event("agent", "stage_started", max_turns=8, max_tool_calls=12),
            _event("agent", "llm_called", turn=1, messages=2, tools=4),
            _event("agent", "llm_completed", turn=1, tool_calls=1, response_chars=42),
            _event("agent", "tool_called", tool="inspect_api", call=1),
            _event("agent", "tool_completed", tool="inspect_api", call=1, ok=True, chars=120),
            _event("agent", "tool_called", tool="search_code", call=2),
            _event("agent", "tool_completed", tool="search_code", call=2, ok=True, chars=80),
            _event("agent", "tool_called", tool="read_file", call=3),
            _event("agent", "tool_completed", tool="read_file", call=3, ok=False, chars=0),
            _event("agent", "stage_completed", turns=2, tool_calls=3, warnings=0),
        ]
    events += [
        _event("planner", "stage_started", planner="ToolUsingPlanner"),
        _event("planner", "planning_started", planner="tool_using", tools=4),
        _event(
            "planner",
            "planning_completed",
            planner="tool_using",
            agent_status="completed",
            turns=2,
            tool_calls=3,
            endpoints=3,
            files_to_create=4,
            files_to_modify=2,
            warnings=0,
        ),
        _event("planner", "stage_completed", endpoints=3, warnings=0),
        _event(
            "generation",
            "stage_started",
            endpoints=3,
            files_to_create=4,
            files_to_modify=2,
        ),
        _event(
            "generation",
            "generation_completed",
            file_count=9,
            generated_test_count=3,
            generated_source_count=4,
            modified_count=2,
            dependency_changes=1,
            warnings=0,
        ),
        _event("repair", "stage_started", max_iterations=3),
        _event("test_runner", "test_started", files=7, timeout=30.0),
        _event(
            "test_runner",
            "test_completed",
            status=test_status,
            passed=9 if test_status != "passed" else 10,
            failed=1 if test_status != "passed" else 0,
            errors=0,
            skipped=0,
            duration=0.42,
            exit_code=1 if test_status != "passed" else 0,
        ),
    ]
    if with_repair:
        events += [
            _event(
                "repair",
                "repair_started",
                attempt=1,
                max_iterations=3,
                test_status="failed",
                failed=1,
                errors=0,
            ),
            _event(
                "repair",
                "repair_completed",
                attempt=1,
                status="applied",
                changed=True,
                files_changed=1,
                applied_actions=1,
                skipped_actions=0,
            ),
        ]
    events += [
        _event("repair", "stage_completed", status="passed", iterations=1, warnings=0),
        _event("patch", "stage_started"),
        _event("patch", "patch_generated", files_changed=9, added=7, modified=2, deleted=0),
    ]
    return events


def _render(events, *, status="passed", color=False, unicode=True, **kwargs) -> str:
    report = build_report(events, status=status, **kwargs)
    return render_report(report, color=color, unicode=unicode)


class _FakeTty(io.StringIO):
    """冒充终端：让 CLI 的 isatty / encoding 判断走"真终端"那一支。"""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """在假终端里跑 main()，返回 (退出码, 输出)。"""
    fake = _FakeTty()
    with contextlib.redirect_stdout(fake):
        code = cli.main(argv)
    return code, fake.getvalue()


def _snapshot_dir(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ------------------------------------------------- 7. 合成 trace 的渲染（快）


def test_stages_are_numbered_one_to_eight_in_pipeline_order() -> None:
    report = build_report(_trace(), status="passed")

    assert [(stage.index, stage.title) for stage in report.stages] == [
        (index, title) for index, (_, title) in enumerate(DEMO_STAGES, start=1)
    ]
    assert [stage.title for stage in report.stages] == [
        "API Understanding",
        "Repository Understanding",
        "Agent",
        "Integration Planner",
        "Code Generator",
        "Test Runner",
        "Repair Loop",
        "Final Patch",
    ]
    # 每个阶段的结论都来自事件，不是猜的（测试全绿 → Repair Loop 没有活要干）
    assert [stage.status for stage in report.stages] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "skipped",
        "completed",
    ]


def test_stage_headlines_come_from_trace_metadata() -> None:
    report = build_report(_trace(), status="passed")
    headlines = {stage.title: stage.headline for stage in report.stages}

    assert headlines["API Understanding"] == "3 endpoints"
    assert headlines["Repository Understanding"].startswith("Python project")
    assert "2 Python files" in headlines["Repository Understanding"]
    assert headlines["Agent"] == "3 tool calls"
    assert headlines["Integration Planner"] == "3 endpoints planned"
    assert headlines["Code Generator"] == "9 files generated"
    assert headlines["Test Runner"] == "10 passed / 0 failed / 0 errors"
    assert headlines["Final Patch"] == "9 files changed"


def test_agent_tools_are_rendered_by_name_with_status() -> None:
    text = _render(_trace(), status="passed")

    assert "Agent tools" in text
    assert "inspect_api" in text
    assert "search_code" in text
    assert "read_file" in text
    # 三次调用、其中一次失败：成败来自 tool_completed 的 ok
    report = build_report(_trace(), status="passed")
    assert [(tool.name, tool.ok) for tool in report.tools] == [
        ("inspect_api", True),
        ("search_code", True),
        ("read_file", False),
    ]


def test_tool_arguments_are_never_rendered() -> None:
    """渲染层只取工具名与成败：参数是模型写的，可能夹带凭据。

    非空洞性：参数就放在 metadata 里（原始事件确实带着它），断言它没被画出来。
    """
    events = [
        _event("agent", "tool_called", tool="read_file", call=1, arguments="SENTINEL-ARGUMENTS"),
        _event(
            "agent",
            "tool_completed",
            tool="read_file",
            call=1,
            ok=True,
            arguments="SENTINEL-ARGUMENTS",
        ),
    ]

    text = _render(events, status="passed")

    assert "read_file" in text
    assert "SENTINEL-ARGUMENTS" not in text


def test_tool_call_without_result_is_marked_unknown() -> None:
    """回路中途退出：调用发生过，但没有结果——不能凭空显示成成功。"""
    events = [
        _event("agent", "tool_called", tool="search_code", call=1),
        _event("agent", "stage_failed", turns=1, tool_calls=1, warnings=0),
    ]

    report = build_report(events, status="error")

    assert [(tool.name, tool.ok) for tool in report.tools] == [("search_code", None)]


def test_test_runner_block_fields_match_the_trace() -> None:
    text = _render(_trace(test_status="failed"), status="tests_failed")
    report = build_report(_trace(test_status="failed"), status="tests_failed")

    assert report.test is not None
    assert report.test.status == "failed"
    assert (report.test.passed, report.test.failed) == (9, 1)
    assert (report.test.errors, report.test.skipped) == (0, 0)
    assert (report.test.duration, report.test.exit_code) == (0.42, 1)
    assert "Status:    FAILED" in text
    assert "Passed:    9" in text
    assert "Failed:    1" in text
    assert "Duration:  0.42s" in text


def test_repair_attempts_are_rendered() -> None:
    text = _render(_trace(test_status="failed", with_repair=True), status="tests_failed")
    report = build_report(_trace(test_status="failed", with_repair=True), status="tests_failed")

    attempts = [
        (item.attempt, item.status, item.changed, item.files_changed) for item in report.repairs
    ]
    assert attempts == [(1, "applied", True, 1)]
    assert "Repair Loop" in text
    assert "Attempt 1" in text
    assert "Files changed: 1" in text


def test_repair_is_reported_as_skipped_when_tests_pass() -> None:
    """§2：没触发 Repair 就如实写 skipped，不伪造一次修复。"""
    text = _render(_trace(), status="passed")
    report = build_report(_trace(), status="passed")
    repair = report.stages[6]

    assert repair.title == "Repair Loop"
    assert repair.status == "skipped"
    assert repair.headline == "skipped: tests passed"
    assert report.repairs == []
    assert "attempt" not in text.lower()


def test_agent_stage_is_not_used_without_an_agent_loop() -> None:
    """确定性 Planner 下没有 agent 回路：如实说没用上，不画一个空的 ✓。"""
    report = build_report(_trace(with_tools=False), status="passed")
    agent = report.stages[2]

    assert agent.status == "skipped"
    assert agent.headline == "not used in this run"
    assert report.tools == []
    assert "Agent tools" not in _render(_trace(with_tools=False), status="passed")


def test_patch_block_reports_added_modified_deleted() -> None:
    text = _render(_trace(), status="passed")
    report = build_report(_trace(), status="passed")

    assert report.patch is not None
    assert (report.patch.files_changed, report.patch.added) == (9, 7)
    assert (report.patch.modified, report.patch.deleted) == (2, 0)
    assert "Final Patch" in text
    assert "Added:         7" in text
    assert "Deleted:       0" in text


def test_failed_trace_renders_failed_banner() -> None:
    """§13.9：失败 Trace 必须画 FAILED，并且不是 SUCCESS。"""
    failed = _render(
        _trace(test_status="failed", with_repair=True),
        status="tests_failed",
        failed_stage=None,
    )
    passed = _render(_trace(), status="passed")

    assert "FAILED" in failed
    assert "SUCCESS" not in failed
    assert "status: tests_failed" in failed
    assert "SUCCESS" in passed
    assert "FAILED" not in passed


def test_failed_stage_is_shown_when_the_pipeline_errors() -> None:
    text = _render(_trace(), status="error", failed_stage="parse")

    assert "FAILED" in text
    assert "failed stage: parse" in text


# ------------------------------------------------------------ 8. 颜色与编码


def test_color_flag_adds_and_removes_ansi() -> None:
    """两个方向都断言：只测"没有 ANSI"的话，把颜色功能删掉也会通过。"""
    colored = _render(_trace(), status="passed", color=True)
    plain = _render(_trace(), status="passed", color=False)

    assert "\x1b[" in colored
    assert "\x1b" not in plain
    # 上色与否只影响控制符，不影响可读内容
    assert "API Understanding" in colored
    assert "API Understanding" in plain


def test_ascii_markers_for_narrow_encodings() -> None:
    """Windows 的 cp936 装不下 ✓ / ✗ / –，渲染层要能整体降级成 ASCII。"""
    plain = Palette(enabled=False, unicode=True)
    narrow = Palette(enabled=False, unicode=False)

    assert plain.mark("completed") == "✓"
    assert narrow.mark("completed") == "OK"
    assert narrow.mark("failed") == "FAIL"

    text = _render(
        _trace(test_status="failed", with_repair=True), status="tests_failed", unicode=False
    )
    assert "✓" not in text
    assert "✗" not in text
    assert "FAIL" in text  # 失败标记降级后仍然可读


# ------------------------------------------------------- 1./2. CLI 入口与退出码


@pytest.fixture
def fake_demo(monkeypatch):
    """把 run_demo 换成合成结果：CLI 层的行为不需要每次真跑 pipeline。

    退出码取自 PipelineResult.status，所以这里给的是一个真的（最小）PipelineResult
    而不是替身对象——否则测的就是替身的行为。
    """

    def _install(status: str = "passed", failed_stage: str | None = None):
        events = _trace(test_status="passed" if status == "passed" else "failed")
        report = build_report(events, status=status, failed_stage=failed_stage)
        outcome = cli.demo.DemoOutcome(
            report=report, result=PipelineResult(status=status), trace=TraceCollector()
        )
        monkeypatch.setattr(cli, "run_demo", lambda **kwargs: outcome)
        return outcome

    return _install


def test_cli_starts_and_returns_success_exit_code(fake_demo) -> None:
    fake_demo("passed")

    code, text = _run_cli(["demo", "--no-color"])

    assert code == 0
    assert "APIForge" in text
    assert "Autonomous API Integration Agent" in text


def test_output_contains_the_required_sections(fake_demo) -> None:
    fake_demo("passed")

    _, text = _run_cli(["demo", "--no-color"])

    for needle in ("API Understanding", "Repository Understanding", "Test Runner", "SUCCESS"):
        assert needle in text


@pytest.mark.parametrize(
    ("status", "expected"),
    [("passed", 0), ("tests_failed", 1), ("error", 2)],
)
def test_exit_code_follows_the_pipeline_status(fake_demo, status: str, expected: int) -> None:
    fake_demo(status)

    code, _ = _run_cli(["demo", "--no-color"])

    assert code == expected


def test_no_subcommand_prints_usage_and_fails(fake_demo, capsys) -> None:
    code = cli.main([])

    captured = capsys.readouterr()
    assert code == 2
    assert "usage:" in captured.out


def test_no_color_flag_beats_a_terminal(fake_demo) -> None:
    fake_demo("passed")

    _, text = _run_cli(["demo", "--no-color"])

    assert "\x1b" not in text


def test_color_is_used_on_a_terminal_without_the_flag(fake_demo) -> None:
    fake_demo("passed")

    _, text = _run_cli(["demo"])

    assert "\x1b[" in text


def test_no_color_environment_variable_is_honoured(fake_demo, monkeypatch) -> None:
    fake_demo("passed")
    monkeypatch.setenv("NO_COLOR", "1")

    _, text = _run_cli(["demo"])

    assert "\x1b" not in text


def test_non_tty_output_has_no_ansi(fake_demo) -> None:
    """管道 / 重定向 / CI：不是终端就不上色（这里用 StringIO 冒充非 TTY）。"""
    fake_demo("passed")
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        cli.main(["demo"])

    assert "\x1b" not in buffer.getvalue()


# ------------------------------------------------------------ 10. Secret 安全


def test_cli_never_prints_credentials(fake_demo, monkeypatch) -> None:
    """环境里放一个真的哨兵，跑一次 CLI，断言它一个字都没出现在输出里。"""
    fake_demo("passed")
    monkeypatch.setenv("DEEPSEEK_API_KEY", API_KEY_SENTINEL)

    _, text = _run_cli(["demo", "--no-color"])

    assert "APIForge" in text  # 不空洞：输出确实有内容
    assert API_KEY_SENTINEL not in text
    assert "sk-live" not in text


def test_output_contains_no_absolute_paths() -> None:
    """trace 里没有仓库 root / 绝对路径，渲染层也不该把它们画出来。"""
    text = _render(_trace(), status="passed", spec_label="examples/openapi/petstore.yaml")

    assert str(REPO_ROOT) not in text
    assert "site-packages" not in text
    assert "\\" not in text  # Windows 绝对路径的另一个痕迹


# ---------------------------------------------------------- 11./14. 真实运行


@pytest.fixture(scope="module")
def real_run():
    """真跑一次默认 Demo（模块级：整条 pipeline 一次约 1.5s，只跑一次）。

    同时记录运行前后的仓库快照，供"不修改真实仓库"用例使用。
    """
    before = _snapshot_dir(EXAMPLES)
    outcome = run_demo()
    after = _snapshot_dir(EXAMPLES)
    return outcome, before, after


def test_real_run_succeeds_and_covers_every_stage(real_run) -> None:
    outcome, _, _ = real_run

    assert outcome.result.status == "passed"
    assert outcome.exit_code == 0
    assert len(outcome.report.stages) == 8
    # 默认路径走的是确定性 Planner：前 6 个阶段 + patch 都跑到了，Agent 回路没被用到
    statuses = {stage.title: stage.status for stage in outcome.report.stages}
    assert statuses["API Understanding"] == "completed"
    assert statuses["Repository Understanding"] == "completed"
    assert statuses["Agent"] == "skipped"
    assert statuses["Code Generator"] == "completed"
    assert statuses["Test Runner"] == "completed"
    assert statuses["Repair Loop"] == "skipped"
    assert statuses["Final Patch"] == "completed"


def test_real_run_renders_through_the_cli(real_run) -> None:
    outcome, _, _ = real_run

    text = render_report(outcome.report, color=False)

    assert text.startswith("APIForge")
    assert "[1/8] API Understanding" in text
    assert "[8/8] Final Patch" in text
    assert text.rstrip().endswith("=" * 40)
    assert "\x1b" not in text


def test_real_run_does_not_touch_the_repository(real_run) -> None:
    """§13.11：Demo 全程只读真实仓库（测试在临时工作区里跑）。"""
    _, before, after = real_run

    assert before == after
    assert before  # 不空洞：快照里确实有文件


_DURATION_LINE = re.compile(r"Duration:\s+\d+\.\d+s")


def test_demo_is_repeatable(real_run) -> None:
    """§14：同一 fixture 连跑两次，核心输出逐行相同（duration 允许不同）。"""
    first, _, _ = real_run

    second = run_demo()

    def normalize(text: str) -> str:
        return _DURATION_LINE.sub("Duration:  <elapsed>", text)

    first_text = render_report(first.report, color=False)
    second_text = render_report(second.report, color=False)

    assert normalize(first_text) == normalize(second_text)
    # 非空洞：两次都真的产生了测试结论（否则"两次都空"也会相等）
    assert first.report.test is not None and second.report.test is not None
    assert first.report.test.passed == second.report.test.passed > 0


def test_sabotaged_run_fails_honestly() -> None:
    """真实失败路径：注入的断言真的让测试失败，Repair 真的尝试，结论如实是 FAILED。"""
    outcome = run_demo(sabotage=True)

    assert outcome.result.status == "tests_failed"
    assert outcome.exit_code == 1
    assert outcome.report.test is not None
    assert outcome.report.test.status == "failed"
    assert outcome.report.test.failed >= 1
    # 测试失败后确实走到了一次真实修复尝试（不是伪造出来的 attempt）
    assert len(outcome.report.repairs) == 1

    text = render_report(outcome.report, color=False)
    assert "FAILED" in text
    assert "SUCCESS" not in text


def test_module_entry_point_runs_the_demo() -> None:
    """§3 的入口形式真的可用：python -m integration_agent demo。"""
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
        "DEEPSEEK_API_KEY": API_KEY_SENTINEL,
    }
    completed = subprocess.run(
        [sys.executable, "-m", "integration_agent", "demo", "--no-color"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "APIForge" in completed.stdout
    assert "API Understanding" in completed.stdout
    assert "Test Runner" in completed.stdout
    assert "SUCCESS" in completed.stdout
    assert "\x1b" not in completed.stdout
    assert API_KEY_SENTINEL not in completed.stdout
