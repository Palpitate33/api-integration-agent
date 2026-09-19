"""RepairLoop 编排器单元测试。

覆盖要求场景：
    1. 初始测试直接通过            11. repair_plans 顺序正确
    2. failed → repair → pass      12. application_results 顺序正确
    3. failed → should_repair=False 13. warnings 正确聚合
    4. failed → changed=False      14. 原始 artifacts 不被修改
    5. 达到 max_iterations          15. deterministic
    6. 多轮 repair                  16. dependency injection
    7. TestRunner 异常              17. Protocol runtime check
    8. RepairPlanner 异常           18. Pydantic JSON round-trip
    9. RepairApplier 异常           19. 不执行 shell/git
    10. iteration 正确              20. 不修改真实 repository

以及：真实闭环（Runner → DeterministicPlanner → DeterministicApplier → Loop
正确终止为 no_progress）、max_iterations 构造校验。
"""

import os
import subprocess
from pathlib import Path

import pytest

from integration_agent import generation, repair, validation

# ------------------------------------------------------------------ fakes


class FakeRunner:
    """按队列顺序返回 TestResult 的假 Runner。"""

    def __init__(self, results: list[validation.TestResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run(self, artifacts: generation.GeneratedArtifacts) -> validation.TestResult:
        self.calls += 1
        assert self._results, "runner 被调用的次数超过预期"
        return self._results.pop(0)


class FakePlanner:
    """按队列返回 RepairPlan 的假 Planner；支持被 Loop 设置 iteration。"""

    def __init__(self, plans: list[repair.RepairPlan]) -> None:
        self._plans = list(plans)
        self.calls = 0
        self.iteration = 0

    def plan(
        self,
        artifacts: generation.GeneratedArtifacts,
        test_result: validation.TestResult,
    ) -> repair.RepairPlan:
        self.calls += 1
        assert self._plans, "planner 被调用的次数超过预期"
        template = self._plans.pop(0)
        return template.model_copy(update={"iteration": self.iteration})


class FakeApplier:
    """按队列返回 RepairApplicationResult 的假 Applier。"""

    def __init__(self, results: list[repair.RepairApplicationResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def apply(
        self,
        artifacts: generation.GeneratedArtifacts,
        plan: repair.RepairPlan,
    ) -> repair.RepairApplicationResult:
        self.calls += 1
        assert self._results, "applier 被调用的次数超过预期"
        return self._results.pop(0)


class RaisingRunner:
    def run(self, artifacts: generation.GeneratedArtifacts) -> validation.TestResult:
        raise RuntimeError("runner boom")


class RaisingPlanner:
    def plan(self, artifacts, test_result) -> repair.RepairPlan:
        raise ValueError("planner boom")


class RaisingApplier:
    def apply(self, artifacts, plan) -> repair.RepairApplicationResult:
        raise TypeError("applier boom")


# ------------------------------------------------------------------ helpers


def _passed() -> validation.TestResult:
    return validation.TestResult(status="passed", exit_code=0, passed=1, duration=0.1)


def _failed() -> validation.TestResult:
    return validation.TestResult(
        status="failed",
        exit_code=1,
        failed=1,
        duration=0.1,
        failure_details=[validation.FailureDetail(test_name="test_x", message="assert 1 == 2")],
    )


def _artifacts(marker: str = "v1") -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(
                path="tests/test_x.py",
                action="create",
                content=f"VALUE = {marker!r}\n",
            )
        ],
        summary="合成产物（单元测试用）",
    )


def _plan(should_repair: bool = True, *, warnings: list[str] | None = None) -> repair.RepairPlan:
    return repair.RepairPlan(
        should_repair=should_repair,
        failure_category="assertion" if should_repair else "none",
        analysis="分析",
        actions=[]
        if not should_repair
        else [
            repair.RepairAction(
                file="tests/test_x.py",
                action="modify",
                reason="断言失败",
                changes="修正断言",
                confidence=0.5,
            )
        ],
        warnings=warnings or [],
    )


def _application(
    changed: bool = True,
    artifacts: generation.GeneratedArtifacts | None = None,
    *,
    warnings: list[str] | None = None,
) -> repair.RepairApplicationResult:
    return repair.RepairApplicationResult(
        artifacts=artifacts or _artifacts("FIXED"),
        changed=changed,
        warnings=warnings or [],
    )


def _loop(
    runner,
    *,
    planner: FakePlanner | None = None,
    applier: FakeApplier | None = None,
    max_iterations: int = 3,
) -> repair.RepairLoopRunner:
    return repair.RepairLoopRunner(
        test_runner=runner,
        repair_planner=planner or FakePlanner([_plan()]),
        repair_applier=applier or FakeApplier([_application()]),
        max_iterations=max_iterations,
    )


def _content_marker(result: repair.RepairLoopResult) -> str:
    """从生成文件内容 "VALUE = 'v1'\\n" 中取出 marker。"""
    return result.artifacts.files[0].content.removeprefix("VALUE = ").strip().strip("'")


# ---------------------------------------------- 场景 1：初始测试直接通过


def test_initial_pass_returns_immediately() -> None:
    runner = FakeRunner([_passed()])
    planner = FakePlanner([_plan()])
    applier = FakeApplier([_application()])

    result = _loop(runner, planner=planner, applier=applier).run(_artifacts())

    assert result.status == "passed"
    assert result.iterations == 0
    assert result.test_result.status == "passed"
    assert result.repair_plans == []
    assert result.application_results == []
    assert planner.calls == 0  # 未调用 Planner
    assert applier.calls == 0  # 未调用 Applier
    assert result.warnings == []


# ------------------------------------------- 场景 2：failed → repair → pass


def test_failed_then_repair_then_pass() -> None:
    runner = FakeRunner([_failed(), _passed()])
    applier = FakeApplier([_application(changed=True, artifacts=_artifacts("FIXED"))])

    result = _loop(runner, applier=applier).run(_artifacts())

    assert result.status == "passed"
    assert result.iterations == 1
    assert _content_marker(result) == "FIXED"
    assert len(result.repair_plans) == 1
    assert len(result.application_results) == 1


# ------------------------------------- 场景 3：failed → should_repair=False


def test_should_repair_false_stops() -> None:
    runner = FakeRunner([_failed()])
    planner = FakePlanner([_plan(should_repair=False, warnings=["无法自动修复"])])
    applier = FakeApplier([_application()])

    result = _loop(runner, planner=planner, applier=applier).run(_artifacts())

    assert result.status == "not_repairable"
    assert result.iterations == 0
    assert len(result.repair_plans) == 1
    assert result.application_results == []
    assert applier.calls == 0
    assert "无法自动修复" in result.warnings
    assert _content_marker(result) == "v1"


# -------------------------------------------- 场景 4：failed → changed=False


def test_changed_false_stops() -> None:
    runner = FakeRunner([_failed()])
    applier = FakeApplier([_application(changed=False)])

    result = _loop(runner, applier=applier).run(_artifacts())

    assert result.status == "no_progress"
    assert result.iterations == 1
    assert len(result.application_results) == 1
    assert runner.calls == 1  # 不会无限重复同一 RepairPlan
    assert _content_marker(result) == "v1"


# ------------------------------------------------ 场景 5：max_iterations


def test_max_iterations_reached() -> None:
    runner = FakeRunner([_failed(), _failed(), _failed()])
    applier = FakeApplier([_application(), _application()])

    planner = FakePlanner([_plan(), _plan()])
    result = _loop(runner, planner=planner, applier=applier, max_iterations=2).run(_artifacts())

    assert result.status == "max_iterations"
    assert result.iterations == 2
    assert runner.calls == 3  # 初始测试 + 2 次修复后测试
    assert len(result.repair_plans) == 2
    assert len(result.application_results) == 2
    assert "Maximum repair iterations reached." in result.warnings
    assert result.test_result.status == "failed"


# ------------------------------------------------------ 场景 6：多轮 repair


def test_multiple_rounds() -> None:
    runner = FakeRunner([_failed(), _failed(), _passed()])
    applier = FakeApplier(
        [_application(artifacts=_artifacts("v2")), _application(artifacts=_artifacts("v3"))]
    )

    planner = FakePlanner([_plan(), _plan()])
    result = _loop(runner, planner=planner, applier=applier).run(_artifacts())

    assert result.status == "passed"
    assert result.iterations == 2
    assert runner.calls == 3
    assert _content_marker(result) == "v3"
    assert [plan.iteration for plan in result.repair_plans] == [0, 1]


# --------------------------------------------- 场景 7-9：组件异常处理


def test_runner_exception_becomes_error() -> None:
    result = _loop(RaisingRunner()).run(_artifacts())

    assert result.status == "error"
    assert result.error is not None and "TestRunner" in result.error
    assert "runner boom" in result.error
    assert result.iterations == 0
    assert any("Traceback" in item for item in result.warnings)
    assert result.artifacts == _artifacts()


def test_planner_exception_becomes_error() -> None:
    result = _loop(FakeRunner([_failed()]), planner=RaisingPlanner()).run(_artifacts())

    assert result.status == "error"
    assert result.error is not None and "RepairPlanner" in result.error
    assert result.test_result is not None and result.test_result.status == "failed"
    assert any("Traceback" in item for item in result.warnings)


def test_applier_exception_becomes_error() -> None:
    result = _loop(FakeRunner([_failed()]), applier=RaisingApplier()).run(_artifacts())

    assert result.status == "error"
    assert result.error is not None and "RepairApplier" in result.error
    assert len(result.repair_plans) == 1
    assert any("Traceback" in item for item in result.warnings)


# ------------------------------------------------- 场景 10：iteration 正确


def test_iteration_recorded_per_round() -> None:
    runner = FakeRunner([_failed(), _failed(), _failed()])
    planner = FakePlanner([_plan(), _plan(), _plan()])
    applier = FakeApplier([_application(), _application(), _application()])

    result = _loop(runner, planner=planner, applier=applier, max_iterations=3).run(_artifacts())

    assert [plan.iteration for plan in result.repair_plans] == [0, 1, 2]
    assert [plan.max_iterations for plan in result.repair_plans] == [3, 3, 3]


# ----------------------------------------- 场景 11-12：计划/结果顺序正确


def test_plans_and_applications_keep_order() -> None:
    runner = FakeRunner([_failed(), _failed(), _passed()])
    planner = FakePlanner([_plan(warnings=["第一轮"]), _plan(warnings=["第二轮"]), _plan()])
    applier = FakeApplier([_application(), _application()])

    result = _loop(runner, planner=planner, applier=applier).run(_artifacts())

    assert [plan.warnings for plan in result.repair_plans] == [["第一轮"], ["第二轮"]]
    assert [item.changed for item in result.application_results] == [True, True]


# -------------------------------------------------- 场景 13：warnings 聚合


def test_warnings_aggregated_across_rounds() -> None:
    runner = FakeRunner([_failed(), _failed(), _passed()])
    planner = FakePlanner([_plan(), _plan()])
    applier = FakeApplier(
        [_application(warnings=["第一轮应用警告"]), _application(warnings=["第二轮应用警告"])]
    )

    result = _loop(runner, planner=planner, applier=applier).run(_artifacts())

    assert result.warnings == ["第一轮应用警告", "第二轮应用警告"]


# ------------------------------------- 场景 14：原始 artifacts 不被修改


def test_original_artifacts_untouched() -> None:
    initial = _artifacts()
    before = initial.model_dump()

    result = _loop(FakeRunner([_failed(), _passed()])).run(initial)

    assert initial.model_dump() == before
    assert result.artifacts is not initial
    assert _content_marker(result) == "FIXED"  # 结果是修复后的新产物


# ------------------------------------------------------ 场景 15：deterministic


def test_deterministic_output() -> None:
    def run_once():
        return _loop(
            FakeRunner([_failed(), _failed(), _passed()]),
            planner=FakePlanner([_plan(), _plan()]),
            applier=FakeApplier([_application(), _application()]),
        ).run(_artifacts())

    assert run_once() == run_once()


# ------------------------------------------------- 场景 16：依赖注入


def test_dependency_injection_uses_injected_components() -> None:
    runner = FakeRunner([_failed(), _passed()])
    planner = FakePlanner([_plan()])
    applier = FakeApplier([_application()])

    _loop(runner, planner=planner, applier=applier).run(_artifacts())

    assert runner.calls == 2
    assert planner.calls == 1
    assert applier.calls == 1


# --------------------------------------------- 场景 17：Protocol runtime check


def test_protocol_conformance() -> None:
    loop = _loop(FakeRunner([_passed()]))
    assert isinstance(loop, repair.RepairLoop)


# ---------------------------------------------- 场景 18：JSON round-trip


def test_loop_result_json_roundtrip() -> None:
    result = _loop(FakeRunner([_failed(), _passed()])).run(_artifacts())

    restored = repair.RepairLoopResult.model_validate_json(result.model_dump_json())
    assert restored == result


# ----------------------------------------- 场景 19-20：无 shell/git、无写盘


def test_loop_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    def _forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("RepairLoop 不应执行任何子进程")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(os, "system", _forbidden)

    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }

    _loop(FakeRunner([_failed(), _passed()])).run(_artifacts())

    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


# -------------------------------------------------------- 构造校验


def test_ctor_rejects_invalid_max_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        _loop(FakeRunner([_passed()]), max_iterations=0)


# -------------------------------------------------------- 真实闭环


def test_real_closed_loop_terminates_as_no_progress() -> None:
    """真实链路：Runner → DeterministicPlanner → DeterministicApplier。

    当前 Applier 不会真正修改代码，因此 Loop 必须正确终止为 no_progress，
    而不是无限重复同一 RepairPlan。
    """
    artifacts = generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(
                path="tests/test_fail.py",
                action="create",
                content="def test_failure():\n    assert 1 == 2\n",
            )
        ],
        summary="真实闭环测试",
    )
    result = repair.run_repair_loop(artifacts)

    assert result.status == "no_progress"
    assert result.iterations == 1
    assert len(result.repair_plans) == 1
    assert result.repair_plans[0].failure_category == "assertion"
    assert len(result.application_results) == 1
    assert result.application_results[0].changed is False
    assert result.test_result is not None and result.test_result.status == "failed"
    assert result.artifacts == artifacts
    assert any("requires code synthesis" in item for item in result.warnings)


def test_loop_passes_test_result_to_capable_applier() -> None:
    """声明 accepts_test_result 的 Applier 会收到当前 TestResult（向后兼容扩展）。"""
    captured = {}

    class CapableApplier:
        accepts_test_result = True

        def apply(self, artifacts, plan, test_result=None):
            captured["test_result"] = test_result
            return repair.RepairApplicationResult(artifacts=artifacts, changed=False)

    runner = FakeRunner([_failed()])
    result = _loop(runner, applier=CapableApplier()).run(_artifacts())

    assert result.status == "no_progress"
    assert captured["test_result"] is not None
    assert captured["test_result"].status == "failed"
    assert captured["test_result"].failed == 1
