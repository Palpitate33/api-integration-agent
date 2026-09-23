"""把 DemoReport / BenchmarkSummary 画成终端文本。

纯函数：``render_report(...)`` 与 ``render_benchmark(...)``。本模块不读 Trace、
不碰 Pipeline、不推断任何运行状态——拿到什么画什么，所以它可以脱离 Pipeline 单独测
（含"模拟一次失败的运行该长什么样"）。

终端兼容
--------
ANSI 只在一处产生（本模块），并且由调用方给出的 color 开关控制：CLI 在
``--no-color`` / ``NO_COLOR`` / 非 TTY 三种情况下都会传 color=False，此时输出里
**一个转义字符都不会有**（不是"颜色被忽略"，而是根本没生成）。用标准库，
不引入 rich / colorama / Typer / Click。

unicode 开关解决的是另一个平台问题：Windows 上常见的 cp936 控制台编码装不下
✓ / ✗ / … / –，直接 print 会抛 UnicodeEncodeError——一个演示会因为一个符号而崩掉。
所以标记符号在这里成对准备，由调用方按实际编码二选一。
"""

from dataclasses import dataclass

from integration_agent.benchmark import BenchmarkSummary
from integration_agent.cli.demo import (
    STAGE_COMPLETED,
    STAGE_FAILED,
    STAGE_INCOMPLETE,
    STAGE_SKIPPED,
    DemoReport,
    StageView,
)

TITLE = "APIForge"
SUBTITLE = "Autonomous API Integration Agent"
RULE = "=" * 40
STAGE_WIDTH = 30  # 阶段名对齐宽度（最长的是 Repository Understanding）
INDENT = "      "  # 细节行缩进，与阶段标题的左边界对齐

RESET = "\x1b[0m"

# 标记符号：键是"这一刻的结论"，值随终端编码二选一。
_MARKS_UNICODE = {
    STAGE_COMPLETED: "✓",
    STAGE_FAILED: "✗",
    STAGE_INCOMPLETE: "…",
    STAGE_SKIPPED: "–",
    "ok": "✓",
    "bad": "✗",
    "unknown": "·",
}
_MARKS_ASCII = {
    STAGE_COMPLETED: "OK",
    STAGE_FAILED: "FAIL",
    STAGE_INCOMPLETE: "..",
    STAGE_SKIPPED: "--",
    "ok": "OK",
    "bad": "FAIL",
    "unknown": "-",
}


@dataclass(frozen=True)
class Palette:
    """终端上色与符号的选择：两个开关都关时，所有方法都是恒等变换。"""

    enabled: bool
    unicode: bool = True

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\x1b[{code}m{text}{RESET}"

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def mark(self, kind: str) -> str:
        """按结论取一个**已上色**的标记符。"""
        table = _MARKS_UNICODE if self.unicode else _MARKS_ASCII
        text = table.get(kind, table["unknown"])
        if kind == STAGE_COMPLETED or kind == "ok":
            return self.green(text)
        if kind == STAGE_FAILED or kind == "bad":
            return self.red(text)
        if kind == STAGE_INCOMPLETE:
            return self.yellow(text)
        return self.dim(text)


def render_report(report: DemoReport, *, color: bool = False, unicode: bool = True) -> str:
    """渲染一次 Demo 的完整输出（不含结尾换行）。"""
    paint = Palette(color, unicode)
    lines: list[str] = []
    lines.extend(_header(report, paint))
    for stage in report.stages:
        lines.append("")
        lines.extend(_stage_block(stage, paint))
    blocks = [
        _tools_block(report, paint),
        _test_block(report, paint),
        _repair_block(report, paint),
        _patch_block(report, paint),
        _warnings_block(report, paint),
    ]
    for block in blocks:
        if block:
            lines.append("")
            lines.extend(block)
    lines.append("")
    lines.extend(_verdict(report, paint))
    return "\n".join(lines)


def _header(report: DemoReport, paint: Palette) -> list[str]:
    rows = [
        paint.bold(TITLE),
        paint.dim(SUBTITLE),
        "",
        f"{paint.dim('API')}      {report.spec_label}",
        f"{paint.dim('Project')}  {report.project_label}",
    ]
    if report.request:
        rows.append(f"{paint.dim('Request')}  {report.request}")
    if report.mode:
        rows.append(f"{paint.dim('Mode')}     {report.mode}")
    return rows


def _stage_block(stage: StageView, paint: Palette) -> list[str]:
    # 阶段名先补齐宽度、后不着色：着色会插入 ANSI，先补后染才不会算错列宽。
    padded = f"{stage.title:<{STAGE_WIDTH}}"
    head = f"{paint.dim(f'[{stage.index}/8]')} {padded} {paint.mark(stage.status)}"
    lines = [head.rstrip()]
    if stage.headline:
        lines.append(INDENT + stage.headline)
    for detail in stage.details:
        lines.append(INDENT + paint.dim(detail))
    return lines


def _tools_block(report: DemoReport, paint: Palette) -> list[str]:
    """工具调用：只列工具名与成败，不列参数、不列结果原文。"""
    if not report.tools:
        return []
    lines = [paint.bold("Agent tools")]
    for tool in report.tools:
        kind = "ok" if tool.ok is True else "bad" if tool.ok is False else "unknown"
        lines.append(f"  {paint.mark(kind)} {tool.name}")
    return lines


def _test_block(report: DemoReport, paint: Palette) -> list[str]:
    test = report.test
    if test is None:
        return []
    status = test.status.upper()
    painted = paint.green(status) if test.status == "passed" else paint.red(status)
    lines = [
        paint.bold("Test Runner"),
        f"  Status:    {painted}",
        f"  Passed:    {test.passed}",
        f"  Failed:    {test.failed}",
        f"  Errors:    {test.errors}",
        f"  Skipped:   {test.skipped}",
        f"  Duration:  {test.duration:.2f}s",
    ]
    if test.exit_code is not None:
        lines.append(f"  Exit code: {test.exit_code}")
    return lines


def _repair_block(report: DemoReport, paint: Palette) -> list[str]:
    if not report.repairs:
        return []
    lines = [paint.bold("Repair Loop")]
    for attempt in report.repairs:
        applied = paint.green("changed") if attempt.changed else paint.yellow("no changes")
        lines.append(f"  Attempt {attempt.attempt}")
        lines.append(f"    Status:        {attempt.status} ({applied})")
        lines.append(f"    Files changed: {attempt.files_changed}")
    return lines


def _patch_block(report: DemoReport, paint: Palette) -> list[str]:
    patch = report.patch
    if patch is None:
        return []
    return [
        paint.bold("Final Patch"),
        f"  Files changed: {patch.files_changed}",
        f"  Added:         {patch.added}",
        f"  Modified:      {patch.modified}",
        f"  Deleted:       {patch.deleted}",
    ]


def _warnings_block(report: DemoReport, paint: Palette) -> list[str]:
    if not report.warnings:
        return []
    lines = [paint.bold("Warnings")]
    lines.extend(f"  - {warning}" for warning in report.warnings)
    return lines


def _verdict(report: DemoReport, paint: Palette) -> list[str]:
    """最终状态直接取自 PipelineResult.status，不由本模块按事件推断。"""
    lines: list[str] = []
    if report.status != "passed":
        reason = f"status: {report.status}"
        if report.failed_stage:
            reason += f" · failed stage: {report.failed_stage}"
        lines.append(paint.dim(reason))
    verdict = "SUCCESS" if report.status == "passed" else "FAILED"
    painted = paint.bold(paint.green(verdict) if report.status == "passed" else paint.red(verdict))
    lines.extend([paint.dim(RULE), painted, paint.dim(RULE)])
    return lines


def render_benchmark(summary: BenchmarkSummary, *, color: bool = False, mismatches: int = 0) -> str:
    """把一轮 Benchmark 的汇总画成终端表格。

    每一个数字都直接取自 summary（包括 pass_rate 等计算属性），本函数不重新计算
    任何指标。mismatches 是 CLI 层对比 expected_outcome 后得出的计数，只用于多画
    一行提醒，不改变任何数字。
    """
    paint = Palette(color)

    def row(label: str, value: str) -> str:
        return f"{label:<20} {value}"

    def status_paint(status: str) -> str:
        if status == "passed":
            return paint.green("PASSED")
        if status == "error":
            return paint.yellow("ERROR")
        return paint.red("FAILED")

    lines = [
        paint.bold("APIForge Benchmark"),
        paint.dim(RULE),
        "",
        row("Cases:", str(summary.total_cases)),
        row("Passed:", str(summary.passed_cases)),
        row("Failed:", str(summary.failed_cases)),
        row("Errors:", str(summary.error_cases)),
        "",
        row("Pass rate:", f"{summary.pass_rate * 100:.1f}%"),
        row("Repair trigger rate:", f"{summary.repair_trigger_rate * 100:.1f}%"),
        row("Repair recovery rate:", f"{summary.repair_recovery_rate * 100:.1f}%"),
        "",
        row("Repair triggered:", str(summary.repair_triggered)),
        row("Repair recovered:", str(summary.repair_recovered)),
        "",
        row("Avg duration:", f"{summary.average_duration:.2f}s"),
        row("Avg repair attempts:", f"{summary.average_repair_attempts:.2f}"),
        "",
        paint.bold("Cases"),
        paint.dim("-" * 40),
    ]
    width = max((len(result.case_name) for result in summary.results), default=0) + 3
    for result in summary.results:
        lines.append(f"{result.case_name:<{width}} {status_paint(result.status)}")
    if mismatches:
        lines.append("")
        lines.append(
            paint.yellow(f"注意：{mismatches} 个 case 的实际结果与 expected_outcome 不一致")
        )
    return "\n".join(lines)


__all__ = ["Palette", "render_benchmark", "render_report"]
