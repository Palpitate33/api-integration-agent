"""APIForge 的命令行入口。

    python -m integration_agent demo              # 离线确定性闭环（测试全绿）
    python -m integration_agent demo --sabotage   # 注入确定性失败 → 真的走 Repair Loop
    python -m integration_agent demo --no-color   # 关掉 ANSI 颜色
    python -m integration_agent benchmark         # 跑默认 Benchmark 套件并打印指标

退出码（可直接用于 CI / 脚本）：
    demo：
        0  passed        集成通过
        1  tests_failed  流程走完但最终测试没通过
        2  error         流程自身异常（或用法错误）
    benchmark：
        0  跑完且每个 case 的实际结果都与 expected_outcome 一致
        1  跑完但有 case 与 expected_outcome 不一致（Benchmark 本身在报警）
        2  用法错误

只用标准库 argparse：这个 CLI 的全部价值是"把已经能跑的东西讲清楚"，引入 Typer /
Click 只会给一个演示入口增加依赖。

颜色策略（三层，顺序即优先级）：
    1. 显式 --no-color
    2. 环境变量 NO_COLOR（通用约定）
    3. 非 TTY（管道 / 重定向 / CI）一律不上色
"""

import argparse
import os
import sys
from collections.abc import Sequence

from integration_agent.benchmark import DEFAULT_CASES, BenchmarkRunner
from integration_agent.cli.demo import DEFAULT_REQUEST, run_demo
from integration_agent.cli.render import render_benchmark, render_report

USAGE_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m integration_agent",
        description="APIForge — Autonomous API Integration Agent",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{demo,benchmark}")
    demo = subparsers.add_parser(
        "demo",
        help="跑一次真实的端到端集成，并按阶段打印 Execution Trace",
        description=(
            "跑一次真实的端到端集成（解析 API → 扫描仓库 → 规划 → 生成 → 测试 → "
            "修复 → 最终 patch），全过程由 Execution Trace 驱动展示。"
            "数据固定使用 examples/openapi/petstore.yaml + examples/demo_project。"
        ),
    )
    demo.add_argument(
        "--sabotage",
        action="store_true",
        help="用项目已有的注入器把一个断言改错，让测试真的失败并真的走到 Repair Loop",
    )
    demo.add_argument("--request", default=DEFAULT_REQUEST, help="集成请求（写进规划提示词）")
    demo.add_argument("--max-iterations", type=int, default=3, help="Repair Loop 的最大迭代次数")
    demo.add_argument("--no-color", action="store_true", help="不输出 ANSI 颜色")
    benchmark = subparsers.add_parser(
        "benchmark",
        help="运行离线确定性 Benchmark（真实跑 Pipeline，输出量化指标）",
        description=(
            "真实执行默认 Benchmark 套件（4 个固定 case，复用 examples/ 已有 fixture，"
            "不联网），从 Execution Trace 归纳结构化结果并打印指标。"
            "退出码 1 表示有 case 的实际结果与 expected_outcome 不一致。"
        ),
    )
    benchmark.add_argument(
        "--max-iterations", type=int, default=3, help="Repair Loop 的最大迭代次数"
    )
    benchmark.add_argument("--no-color", action="store_true", help="不输出 ANSI 颜色")
    return parser


def _use_color(no_color: bool) -> bool:
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _stdout_supports_unicode() -> bool:
    """目标终端编码能否装下 ✓ / ✗ / … / –（Windows 的 cp936 装不下）。"""
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return True
    try:
        "✓✗…–·".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        outcome = run_demo(
            request=args.request,
            max_iterations=args.max_iterations,
            sabotage=args.sabotage,
        )
        print(
            render_report(
                outcome.report,
                color=_use_color(args.no_color),
                unicode=_stdout_supports_unicode(),
            )
        )
        return outcome.exit_code
    if args.command == "benchmark":
        summary = BenchmarkRunner(max_iterations=args.max_iterations).run_default()
        expected = {case.name: case.expected_outcome for case in DEFAULT_CASES}
        mismatches = sum(
            1 for result in summary.results if expected.get(result.case_name) != result.status
        )
        print(
            render_benchmark(
                summary,
                color=_use_color(args.no_color),
                mismatches=mismatches,
            )
        )
        return 1 if mismatches else 0
    parser.print_help()
    return USAGE_EXIT_CODE


__all__ = ["build_parser", "main"]
