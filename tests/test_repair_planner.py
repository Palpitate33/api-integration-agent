"""RepairPlanner 单元测试。

覆盖要求场景：
    1. passed → no repair          9. multiple failure_details
    2. assertion failure            10. evidence preservation
    3. import failure               11. confidence 范围校验
    4. dependency failure           12. iteration / max_iterations 校验
    5. collection failure           13. Pydantic JSON round-trip
    6. timeout                      14. deterministic output
    7. environment error            15. Protocol runtime check
    8. unknown failure

以及：动作去重合并、动作数量上限、迭代上限停止修复、无副作用证明
（不修改 repo / 不创建文件 / 不执行 pytest / 不执行 shell）、
真实 Runner → Planner 链路 sanity check。
"""

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from integration_agent import generation, repair, validation

PASSING_TEST = "def test_add():\n    assert 1 + 1 == 2\n"


def _result(
    status: str = "failed",
    *,
    failed: int = 1,
    errors: int = 0,
    passed: int = 0,
    skipped: int = 0,
    exit_code: int = 1,
    details: list[validation.FailureDetail] | None = None,
    stdout: str = "",
    stderr: str = "",
    warnings: list[str] | None = None,
) -> validation.TestResult:
    return validation.TestResult(
        status=status,
        exit_code=exit_code,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        duration=0.5,
        stdout=stdout,
        stderr=stderr,
        failure_details=details or [],
        dependency_warnings=warnings or [],
    )


def _detail(
    message: str = "assert 1 == 2",
    *,
    name: str = "test_x",
    file: str = "tests/test_x.py",
    line: int = 2,
) -> validation.FailureDetail:
    return validation.FailureDetail(
        test_name=name, file=file, line=line, message=message, traceback=message
    )


def _artifacts(
    *,
    dependencies: list[generation.DependencyChange] | None = None,
) -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(path="tests/test_x.py", action="create", content=PASSING_TEST)
        ],
        dependency_changes=dependencies or [],
        summary="合成产物（单元测试用）",
    )


def _plan(result: validation.TestResult, artifacts: generation.GeneratedArtifacts | None = None):
    return repair.plan_repair(artifacts or _artifacts(), result)


# ------------------------------------------------- 场景 1：passed → no repair


def test_passed_needs_no_repair() -> None:
    plan = _plan(_result(status="passed", failed=0, passed=1, exit_code=0))
    assert plan.should_repair is False
    assert plan.failure_category == "none"
    assert plan.actions == []
    assert "无需修复" in plan.analysis


# ------------------------------------------------- 场景 2：assertion failure


def test_assertion_failure_classified() -> None:
    plan = _plan(_result(details=[_detail("assert response.status_code == 200")]))

    assert plan.should_repair is True
    assert plan.failure_category == "assertion"
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.file == "tests/test_x.py"
    assert action.action == "modify"
    assert action.target == "line 2"
    assert 0 <= action.confidence <= 1
    assert "断言" in plan.analysis


# ---------------------------------------------------- 场景 3：import failure


def test_import_failure_classified() -> None:
    detail = _detail("ModuleNotFoundError: No module named 'demo_client'")
    plan = _plan(_result(details=[detail]))

    assert plan.failure_category == "import"
    action = plan.actions[0]
    assert action.file == "tests/test_x.py"
    assert action.target == "import section"
    assert "demo_client" in action.changes
    assert "导入失败" in plan.analysis


# ------------------------------------------------- 场景 4：dependency failure


def test_dependency_failure_via_warnings() -> None:
    result = _result(
        details=[_detail("assert 1 == 2")],
        warnings=["依赖 httpx>=0.27 未安装（离线模式，不自动安装）"],
    )
    plan = _plan(result)

    assert plan.failure_category == "dependency"
    action = plan.actions[0]
    assert "依赖" in action.reason
    assert any("dependency warning" in item for item in action.evidence)


def test_dependency_failure_via_declared_module() -> None:
    detail = _detail("ModuleNotFoundError: No module named 'httpx'")
    artifacts = _artifacts(
        dependencies=[generation.DependencyChange(name="httpx", version=">=0.27", action="add")]
    )
    plan = _plan(_result(details=[detail]), artifacts)

    assert plan.failure_category == "dependency"
    assert "httpx" in plan.actions[0].changes
    assert "安装依赖" in plan.analysis


# ------------------------------------------------- 场景 5：collection failure


def test_collection_failure_classified() -> None:
    detail = _detail("fixture 'broken' not found")
    plan = _plan(_result(details=[detail], errors=1, failed=0))

    assert plan.failure_category == "collection"
    assert plan.actions[0].target == "fixture / 测试定义 / 导入"
    assert "收集阶段" in plan.analysis


def test_collection_failure_via_stdout() -> None:
    result = _result(
        status="error",
        exit_code=2,
        stdout="Interrupted: 1 error during collection",
    )
    plan = _plan(result)
    assert plan.failure_category == "collection"


# -------------------------------------------------------- 场景 6：timeout


def test_timeout_classified() -> None:
    result = _result(
        status="timeout",
        exit_code=None,
        failed=0,
        stderr="测试超过 30.0s 超时，pytest 子进程已被终止",
    )
    plan = _plan(result)

    assert plan.should_repair is True
    assert plan.failure_category == "timeout"
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.file == ""  # 无法定位具体文件
    assert action.confidence <= 0.4
    assert any("duration" in item for item in action.evidence)
    assert any("timeout limit: 30.0s" in item for item in action.evidence)
    assert any("Exact source file" in item for item in plan.warnings)
    assert "无法定位" in plan.analysis


# --------------------------------------------------- 场景 7：environment


def test_environment_error_no_repair() -> None:
    plan = _plan(_result(status="error", exit_code=5, failed=0))

    assert plan.should_repair is False
    assert plan.failure_category == "environment"
    assert plan.actions == []
    assert "exit_code=5" in plan.analysis


# ------------------------------------------------------- 场景 8：unknown


def test_unknown_failure_no_guessing() -> None:
    detail = _detail("something completely unexpected happened")
    plan = _plan(_result(details=[detail]))

    assert plan.should_repair is False
    assert plan.failure_category == "unknown"
    assert plan.actions == []
    assert any("无法可靠分类" in item for item in plan.warnings)


# ------------------------------------------- 场景 9：multiple failure_details


def test_multiple_details_priority_and_order() -> None:
    details = [
        _detail("assert 1 == 2", file="tests/test_a.py", line=2),
        _detail("ModuleNotFoundError: No module named 'missing_mod'", file="tests/test_b.py"),
    ]
    plan = _plan(_result(failed=2, details=details))

    assert plan.failure_category == "import"  # import 优先级高于 assertion
    assert [action.file for action in plan.actions] == [
        "tests/test_a.py",
        "tests/test_b.py",
    ]
    assert {action.file for action in plan.actions} == {
        "tests/test_a.py",
        "tests/test_b.py",
    }


def test_actions_deduplicated_and_evidence_merged() -> None:
    details = [
        _detail("assert 1 == 2", file="tests/test_a.py", line=2),
        _detail("assert 2 == 3", file="tests/test_a.py", line=5),
    ]
    plan = _plan(_result(failed=2, details=details))

    assert len(plan.actions) == 1
    evidence = plan.actions[0].evidence
    assert any("assert 1 == 2" in item for item in evidence)
    assert any("assert 2 == 3" in item for item in evidence)


def test_actions_bounded() -> None:
    details = [
        _detail("assert 1 == 2", file=f"tests/test_{index}.py", line=index) for index in range(6)
    ]
    plan = _plan(_result(failed=6, details=details))

    assert len(plan.actions) == 5
    assert any("仅保留前 5" in item for item in plan.warnings)


# ------------------------------------------------ 场景 10：evidence preservation


def test_evidence_preserved() -> None:
    detail = _detail("assert 1 == 2", file="tests/test_x.py", line=7)
    plan = _plan(_result(details=[detail]))

    evidence = plan.actions[0].evidence
    assert "failure message: assert 1 == 2" in evidence
    assert "test file: tests/test_x.py" in evidence
    assert "line: 7" in evidence


# --------------------------------------------- 场景 11：confidence 范围校验


@pytest.mark.parametrize("confidence", [1.5, -0.1, 2.0])
def test_confidence_out_of_range_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError):
        repair.RepairAction(
            file="tests/test_x.py",
            action="modify",
            reason="测试",
            changes="修改",
            confidence=confidence,
        )


# -------------------------------------- 场景 12：iteration / max_iterations


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iteration": -1},
        {"max_iterations": 0},
        {"iteration": 4, "max_iterations": 3},
    ],
)
def test_iteration_bounds_rejected(kwargs: dict) -> None:
    defaults = dict(
        should_repair=True,
        failure_category="assertion",
        analysis="分析",
    )
    with pytest.raises(ValidationError):
        repair.RepairPlan(**defaults, **kwargs)


def test_no_actions_when_should_repair_false() -> None:
    with pytest.raises(ValidationError):
        repair.RepairPlan(
            should_repair=False,
            failure_category="assertion",
            analysis="分析",
            actions=[
                repair.RepairAction(
                    file="tests/test_x.py",
                    action="modify",
                    reason="测试",
                    changes="修改",
                    confidence=0.5,
                )
            ],
        )


def test_planner_ctor_rejects_invalid_iteration() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        repair.DeterministicRepairPlanner(max_iterations=0)
    with pytest.raises(ValueError, match="不能大于"):
        repair.DeterministicRepairPlanner(max_iterations=3, iteration=4)


def test_iteration_cap_stops_repair() -> None:
    planner = repair.DeterministicRepairPlanner(max_iterations=3, iteration=3)
    plan = planner.plan(_artifacts(), _result(details=[_detail()]))

    assert plan.should_repair is False
    assert plan.failure_category == "assertion"  # 分类仍保留
    assert plan.actions == []
    assert any("最大迭代次数" in item for item in plan.warnings)


# ------------------------------------------- 场景 13：Pydantic JSON round-trip


def test_plan_json_roundtrip() -> None:
    plan = _plan(_result(details=[_detail()]))
    assert repair.RepairPlan.model_validate_json(plan.model_dump_json()) == plan


# --------------------------------------------------- 场景 14：deterministic


def test_deterministic_output() -> None:
    result = _result(
        failed=2,
        details=[
            _detail("assert 1 == 2", file="tests/test_a.py"),
            _detail("ModuleNotFoundError: No module named 'm'", file="tests/test_b.py"),
        ],
        warnings=["依赖 x 未安装（离线模式，不自动安装）"],
    )
    assert _plan(result) == _plan(result)


# --------------------------------------------- 场景 15：Protocol runtime check


def test_protocol_conformance() -> None:
    assert isinstance(repair.DeterministicRepairPlanner(), repair.RepairPlanner)


# --------------------------------------------------- 无副作用 / 真实链路


def test_planner_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    def _forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("RepairPlanner 不应执行任何子进程")

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

    _plan(_result(details=[_detail()]))
    _plan(_result(details=[_detail()]))

    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_real_runner_to_planner_pipeline() -> None:
    """真实链路 sanity check：Runner 产出的 TestResult 能被 Planner 正确分类。"""
    artifacts = generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(
                path="tests/test_fail.py",
                action="create",
                content="def test_failure():\n    assert 1 == 2\n",
            )
        ],
    )
    result = validation.run_tests(artifacts)
    plan = repair.plan_repair(artifacts, result)

    assert result.status == "failed"
    assert plan.should_repair is True
    assert plan.failure_category == "assertion"
    assert plan.actions[0].file == "tests/test_fail.py"
