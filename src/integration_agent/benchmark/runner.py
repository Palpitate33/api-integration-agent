"""Benchmark 执行器：真实调用 Pipeline，把 Execution Trace 归纳成结构化结果。

执行路径（§八 要求的数据流，中间没有任何替身）：

    BenchmarkCase → run_pipeline → TraceCollector → BenchmarkResult

BenchmarkResult 的每一个数字都取自 Trace 事件（test_completed / repair_started /
generation_completed / patch_generated），本模块不写死任何一个结果；唯一不属于
事件的是 status（直接来自 PipelineResult.status 的映射）与 duration（墙钟计时）。

确定性（repeat > 1）
--------------------
同一 case 连续跑两次后，核心字段（status、各测试计数、修复次数、patch 文件数、
生成计数）必须逐项一致——不一致就是 failed。这是 deterministic_repeat 这个 case
的全部意义：暴露"同一输入产出不同结果"，而不是取平均假装稳定。duration 与
timestamp 刻意不在比较范围（它们是允许波动的物理量）。

异常策略
--------
case 路径不合法（开发者笔误）抛 ValueError；run_suite 把单个 case 的任何未预期
异常收成 status="error" 的结果并记日志，让整轮 Benchmark 跑完——Benchmark 的价值
是"看清全部"，不是"第一个坏 case 就停"。
"""

import logging
import time
from collections.abc import Iterable
from pathlib import Path

from integration_agent.agent import DeterministicPlanner
from integration_agent.api_server.demo import SabotagedGenerator
from integration_agent.api_server.paths import resolve_allowed
from integration_agent.benchmark.cases import DEFAULT_CASES
from integration_agent.benchmark.models import BenchmarkCase, BenchmarkResult, BenchmarkSummary
from integration_agent.pipeline import PipelineResult, run_pipeline
from integration_agent.trace import TraceCollector, TraceEvent

logger = logging.getLogger(__name__)

# PipelineResult.status（passed / tests_failed / error）→ BenchmarkResult.status 的映射。
_PIPELINE_TO_BENCHMARK = {"passed": "passed", "tests_failed": "failed", "error": "error"}

# 确定性对比的字段：repeat 跑两次，这些字段必须逐项一致。
_CORE_FIELDS = (
    "test_passed",
    "test_failed",
    "test_errors",
    "repair_attempts",
    "patch_files",
    "generated_source_count",
    "generated_test_count",
)


class BenchmarkRunner:
    """跑默认套件或任意 case 列表的执行器。"""

    def __init__(self, *, max_iterations: int = 3) -> None:
        self.max_iterations = max_iterations

    def run_default(self) -> BenchmarkSummary:
        """跑默认的 4 个 case。"""
        return self.run_suite(DEFAULT_CASES)

    def run_suite(self, cases: Iterable[BenchmarkCase]) -> BenchmarkSummary:
        """依次真实执行每个 case，汇总成 BenchmarkSummary。"""
        results: list[BenchmarkResult] = []
        for case in cases:
            try:
                results.append(self.run_case(case))
            except Exception:
                logger.exception("benchmark case 执行失败：%s", case.name)
                results.append(_error_result(case.name))
        return _summarize(results)

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """真实执行一个 case；repeat > 1 时连跑并做确定性对比。"""
        spec, project = self._resolve(case)
        runs = [self._run_once(case, spec, project) for _ in range(case.repeat)]
        if len(runs) == 1:
            return runs[0]
        return _merge_repeat_runs(case.name, runs)

    @staticmethod
    def _resolve(case: BenchmarkCase) -> tuple[Path, Path]:
        """按 api_server.paths 的同一套规则解析 case 路径（拒绝对路径与越界）。"""
        spec, spec_error = resolve_allowed(case.spec_path)
        project, project_error = resolve_allowed(case.project_path)
        if spec_error or project_error or spec is None or project is None:
            raise ValueError(f"case {case.name!r} 的路径不合法：{spec_error or project_error}")
        return spec, project

    def _run_once(self, case: BenchmarkCase, spec: Path, project: Path) -> BenchmarkResult:
        trace = TraceCollector()
        kwargs: dict = {}
        if case.sabotage:
            kwargs["code_generator"] = SabotagedGenerator()
        if case.max_endpoints is not None:
            kwargs["planner"] = DeterministicPlanner(max_endpoints=case.max_endpoints)
        started = time.perf_counter()
        result: PipelineResult = run_pipeline(
            spec,
            project,
            request=case.request,
            max_iterations=self.max_iterations,
            trace=trace,
            **kwargs,
        )
        return _result_from_trace(
            case.name,
            status=_PIPELINE_TO_BENCHMARK.get(result.status, "error"),
            duration=time.perf_counter() - started,
            events=trace.events(),
        )


def _first(events: Iterable[TraceEvent], event_type: str) -> TraceEvent | None:
    for event in events:
        if event.event_type == event_type:
            return event
    return None


def _int_of(event: TraceEvent | None, key: str) -> int:
    if event is None:
        return 0
    return int((event.metadata or {}).get(key, 0))


def _result_from_trace(
    case_name: str,
    *,
    status: str,
    duration: float,
    events: Iterable[TraceEvent],
) -> BenchmarkResult:
    test = _first(events, "test_completed")
    generation = _first(events, "generation_completed")
    patch = _first(events, "patch_generated")
    repairs = sum(1 for event in events if event.event_type == "repair_started")
    return BenchmarkResult(
        case_name=case_name,
        status=status,
        duration=duration,
        test_passed=_int_of(test, "passed"),
        test_failed=_int_of(test, "failed"),
        test_errors=_int_of(test, "errors"),
        repair_attempts=repairs,
        patch_files=_int_of(patch, "files_changed"),
        generated_source_count=_int_of(generation, "generated_source_count"),
        generated_test_count=_int_of(generation, "generated_test_count"),
    )


def _core(result: BenchmarkResult) -> tuple:
    return tuple(getattr(result, field) for field in _CORE_FIELDS)


def _merge_repeat_runs(case_name: str, runs: list[BenchmarkResult]) -> BenchmarkResult:
    """把 repeat 的多次运行合并成一个结果：核心字段必须一致，否则 failed。"""
    first = runs[0]
    core_equal = all(_core(first) == _core(run) for run in runs[1:])
    if any(run.status == "error" for run in runs):
        status = "error"
    elif core_equal and all(run.status == "passed" for run in runs):
        status = "passed"
    else:
        status = "failed"
    return BenchmarkResult(
        case_name=case_name,
        status=status,
        duration=sum(run.duration for run in runs) / len(runs),
        test_passed=first.test_passed,
        test_failed=first.test_failed,
        test_errors=first.test_errors,
        repair_attempts=first.repair_attempts,
        patch_files=first.patch_files,
        generated_source_count=first.generated_source_count,
        generated_test_count=first.generated_test_count,
    )


def _error_result(case_name: str) -> BenchmarkResult:
    return BenchmarkResult(
        case_name=case_name,
        status="error",
        duration=0.0,
        test_passed=0,
        test_failed=0,
        test_errors=0,
        repair_attempts=0,
        patch_files=0,
    )


def _summarize(results: list[BenchmarkResult]) -> BenchmarkSummary:
    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    errors = sum(1 for result in results if result.status == "error")
    triggered = [result for result in results if result.repair_attempts > 0]
    recovered = sum(1 for result in triggered if result.status == "passed")
    return BenchmarkSummary(
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=failed,
        error_cases=errors,
        repair_triggered=len(triggered),
        repair_recovered=recovered,
        average_duration=sum(result.duration for result in results) / len(results)
        if results
        else 0.0,
        average_repair_attempts=sum(result.repair_attempts for result in results) / len(results)
        if results
        else 0.0,
        results=results,
    )


__all__ = ["BenchmarkRunner"]
