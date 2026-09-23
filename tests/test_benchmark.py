"""Benchmark 测试。

覆盖要求逐条对应：

    1. Case 模型             → 合法 / 非法 expected_outcome / repeat=0 / 越界路径
    2. 单 Case 真实运行        → BenchmarkRunner().run_case(petstore_success)
    3. 多 Case 汇总           → total / passed / failed（默认套件 4 个 case）
    4. Repair 统计           → sabotage case：repair_triggered == 1，attempts == 1
    5. Deterministic         → 同一 case 跑两次核心字段一致；repeat case 通过
    6. 不修改真实 Repository  → 运行前后对 examples/ 全量快照比对
    7. CLI                   → 子进程真跑 python -m integration_agent benchmark
    8. 无硬编码结果           → 用 spy 数 run_pipeline 的真实调用次数 == 5

做法与 CLI Demo 测试一致：数据统计类断言跑在共享的 module fixture 上（整轮套件
5 次 pipeline 运行 ≈ 3s，只跑一次），模型 / 渲染 / 参数校验类断言用合成数据。
"""

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from pydantic import ValidationError

from integration_agent import cli
from integration_agent.benchmark import (
    DEFAULT_CASES,
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSummary,
)
from integration_agent.benchmark import runner as runner_module
from integration_agent.cli.render import render_benchmark

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

# 确定性对比要排除的字段：duration 是物理量，允许波动（§四 Case 4）。
CORE_FIELDS = (
    "status",
    "test_passed",
    "test_failed",
    "test_errors",
    "repair_attempts",
    "patch_files",
    "generated_source_count",
    "generated_test_count",
)


def _snapshot_dir(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _success_case() -> BenchmarkCase:
    return next(case for case in DEFAULT_CASES if case.name == "petstore_success")


# ------------------------------------------------- 1. Case 模型与路径校验


def test_case_model_accepts_a_valid_case() -> None:
    case = BenchmarkCase(
        name="petstore_success",
        description="正常成功",
        spec_path="openapi/petstore.yaml",
        project_path="demo_project",
        request="集成 Petstore 的查询与创建接口",
        expected_outcome="passed",
    )

    assert case.expected_outcome == "passed"
    assert case.repeat == 1
    assert case.sabotage is False


def test_case_model_rejects_invalid_expected_outcome() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(
            name="x",
            description="d",
            spec_path="a",
            project_path="b",
            expected_outcome="boom",  # type: ignore[arg-type]
        )


def test_case_model_rejects_zero_repeat() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(
            name="x",
            description="d",
            spec_path="a",
            project_path="b",
            expected_outcome="passed",
            repeat=0,
        )


def test_case_model_rejects_blank_name() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(
            name="   ",
            description="d",
            spec_path="a",
            project_path="b",
            expected_outcome="passed",
        )


def test_runner_rejects_paths_outside_examples() -> None:
    """case 路径按 examples/ 的安全规则解析：越界与绝对路径都不接受。"""
    for bad in ("../pyproject.toml", "..\\..\\pyproject.toml", "/etc/passwd"):
        case = BenchmarkCase(
            name="escape",
            description="d",
            spec_path=bad,
            project_path="demo_project",
            expected_outcome="passed",
        )
        with pytest.raises(ValueError):
            BenchmarkRunner().run_case(case)


def test_suite_turns_a_broken_case_into_an_error_result() -> None:
    """单 case 崩溃不打断整轮：收成 error 结果，Benchmark 继续。"""
    broken = BenchmarkCase(
        name="broken",
        description="d",
        spec_path="openapi/does-not-exist.yaml",
        project_path="demo_project",
        expected_outcome="passed",
    )

    summary = BenchmarkRunner().run_suite([broken])

    assert summary.total_cases == 1
    assert summary.error_cases == 1
    assert summary.passed_cases == 0
    assert summary.results[0].status == "error"


# ------------------------------------------------- 2./3./4. 真实运行与汇总


@pytest.fixture(scope="module")
def success_result() -> BenchmarkResult:
    """真实跑一次 petstore_success（pipeline 子进程 ≈ 0.6s）。"""
    return BenchmarkRunner().run_case(_success_case())


@pytest.fixture(scope="module")
def suite_run() -> tuple[BenchmarkSummary, dict, dict]:
    """真实跑完整默认套件（5 次 pipeline 运行），并记录运行前后的仓库快照。"""
    before = _snapshot_dir(EXAMPLES)
    summary = BenchmarkRunner().run_default()
    after = _snapshot_dir(EXAMPLES)
    return summary, before, after


def test_single_case_runs_the_real_pipeline(success_result) -> None:
    assert success_result.case_name == "petstore_success"
    assert success_result.status == "passed"
    assert success_result.test_passed > 0
    assert success_result.test_failed == 0
    assert success_result.test_errors == 0
    assert success_result.repair_attempts == 0  # 全绿 → 没触发 Repair
    assert success_result.patch_files > 0
    assert success_result.generated_source_count > 0
    assert success_result.generated_test_count > 0
    assert success_result.duration > 0


def test_summary_counts(suite_run) -> None:
    summary, _, _ = suite_run

    assert summary.total_cases == 4
    assert summary.passed_cases == 3
    assert summary.failed_cases == 1
    assert summary.error_cases == 0
    assert [result.case_name for result in summary.results] == [case.name for case in DEFAULT_CASES]


def test_repair_statistics(suite_run) -> None:
    summary, _, _ = suite_run

    assert summary.repair_triggered == 1
    assert summary.repair_recovered == 0
    assert summary.repair_trigger_rate == pytest.approx(0.25)
    assert summary.repair_recovery_rate == 0.0
    # 触发 Repair 的是 sabotage case：测试真的失败，修复真的尝试了一次
    sabotage = summary.results[1]
    assert sabotage.case_name == "petstore_sabotage"
    assert sabotage.status == "failed"
    assert sabotage.test_failed == 1
    assert sabotage.repair_attempts == 1


def test_rates_are_computed_from_results_not_hardcoded(suite_run) -> None:
    summary, _, _ = suite_run

    assert summary.pass_rate == pytest.approx(summary.passed_cases / summary.total_cases)
    assert summary.pass_rate == pytest.approx(0.75)
    assert summary.average_duration == pytest.approx(
        sum(result.duration for result in summary.results) / 4
    )
    assert summary.average_repair_attempts == pytest.approx(0.25)


def test_different_cases_on_the_same_fixtures_produce_different_results(suite_run) -> None:
    """同一 fixture、不同配置（sabotage / max_endpoints）→ 结果真的不同。

    这也是"结果不是写死的常量"的旁证：常量不会随 case 变。
    """
    summary, _, _ = suite_run
    success, sabotage, multi = summary.results[0], summary.results[1], summary.results[2]

    assert success.test_failed == 0
    assert sabotage.test_failed == 1
    assert success.repair_attempts == 0
    assert sabotage.repair_attempts == 1
    # max_endpoints=2 → 端点更少 → 通过测试数不同（8 vs 10）
    assert multi.test_passed != success.test_passed


# ------------------------------------------------------------ 5. Deterministic


def test_same_case_runs_twice_produces_identical_core_results(success_result) -> None:
    second = BenchmarkRunner().run_case(_success_case())

    for field in CORE_FIELDS:
        assert getattr(success_result, field) == getattr(second, field), field


def test_repeat_case_merges_runs_and_verifies_agreement(suite_run) -> None:
    summary, _, _ = suite_run

    repeat = summary.results[3]
    assert repeat.case_name == "deterministic_repeat"
    assert repeat.status == "passed"  # 两次运行核心字段一致且都通过
    assert repeat.test_passed == summary.results[0].test_passed


def test_repeat_case_fails_when_runs_disagree(monkeypatch) -> None:
    """确定性对比的核心语义：两次运行不一致 = failed，不是取平均假装稳定。"""

    class _DisagreeingRunner(BenchmarkRunner):
        def _run_once(self, case, spec, project) -> BenchmarkResult:
            patch = 7 + getattr(self, "_extra", 0)
            self._extra = 1  # 第二次运行产出不同的 patch 文件数
            return BenchmarkResult(
                case_name=case.name,
                status="passed",
                duration=0.1,
                test_passed=10,
                test_failed=0,
                test_errors=0,
                repair_attempts=0,
                patch_files=patch,
            )

    case = BenchmarkCase(
        name="flaky",
        description="d",
        spec_path="openapi/petstore.yaml",
        project_path="demo_project",
        expected_outcome="passed",
        repeat=2,
    )

    result = _DisagreeingRunner().run_case(case)

    assert result.status == "failed"


# -------------------------------------------------- 6. 不修改真实 Repository


def test_benchmark_does_not_touch_the_repository(suite_run) -> None:
    _, before, after = suite_run

    assert before == after
    assert before  # 不空洞：快照里确实有文件


# ------------------------------------------------------------ 7./8. CLI 与真实性


def test_benchmark_cli_runs_end_to_end() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "integration_agent", "benchmark", "--no-color"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "APIForge Benchmark" in completed.stdout
    assert "Pass rate:" in completed.stdout
    assert "petstore_success" in completed.stdout
    assert "petstore_sabotage" in completed.stdout
    assert "PASSED" in completed.stdout
    assert "FAILED" in completed.stdout
    assert "\x1b" not in completed.stdout


def test_results_come_from_real_pipeline_runs(monkeypatch) -> None:
    """§八：BenchmarkResult 必须来自真实执行。

    用 spy 数 run_pipeline 的调用次数：3 个单次 case + deterministic_repeat 的
    2 次 = 5 次。如果有任何结果是从常量里"取"出来的，调用次数就对不上。
    """
    real = runner_module.run_pipeline
    calls: list[str] = []

    def spy(*args, **kwargs):
        calls.append(str(args[1]))
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_module, "run_pipeline", spy)

    summary = BenchmarkRunner().run_default()

    assert summary.total_cases == 4
    assert len(calls) == 5
    assert all(call.endswith("demo_project") for call in calls)


def test_render_benchmark_shows_metrics_from_the_summary() -> None:
    """渲染层只画 summary 给的数字，可脱离 Pipeline 单独测。"""
    summary = BenchmarkSummary(
        total_cases=2,
        passed_cases=1,
        failed_cases=1,
        error_cases=0,
        repair_triggered=1,
        repair_recovered=0,
        average_duration=0.5,
        average_repair_attempts=0.5,
        results=[
            BenchmarkResult(
                case_name="petstore_success",
                status="passed",
                duration=0.5,
                test_passed=10,
                test_failed=0,
                test_errors=0,
                repair_attempts=0,
                patch_files=9,
            ),
            BenchmarkResult(
                case_name="petstore_sabotage",
                status="failed",
                duration=0.6,
                test_passed=9,
                test_failed=1,
                test_errors=0,
                repair_attempts=1,
                patch_files=9,
            ),
        ],
    )

    text = render_benchmark(summary, color=False)

    assert f"{'Cases:':<20} 2" in text
    assert "Pass rate:           50.0%" in text
    assert "Repair recovery rate: 0.0%" in text
    assert "petstore_success" in text
    assert "PASSED" in text
    assert "FAILED" in text
    assert "\x1b" not in text


def test_cli_exit_code_is_one_when_results_disagree_with_expectations(monkeypatch) -> None:
    """expected_outcome 不一致时 benchmark 子命令返回 1（Benchmark 自己在报警）。"""

    class _FakeRunner:
        def __init__(self, **kwargs) -> None:
            pass

        def run_default(self) -> BenchmarkSummary:
            return BenchmarkSummary(
                total_cases=1,
                passed_cases=0,
                failed_cases=1,
                error_cases=0,
                repair_triggered=0,
                repair_recovered=0,
                average_duration=0.1,
                average_repair_attempts=0.0,
                results=[
                    BenchmarkResult(
                        case_name="petstore_success",
                        status="failed",
                        duration=0.1,
                        test_passed=0,
                        test_failed=1,
                        test_errors=0,
                        repair_attempts=0,
                        patch_files=0,
                    )
                ],
            )

    monkeypatch.setattr(cli, "BenchmarkRunner", _FakeRunner)
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        code = cli.main(["benchmark", "--no-color"])

    assert code == 1
    assert "注意：" in buffer.getvalue()  # 输出了与 expected 不一致的提醒
