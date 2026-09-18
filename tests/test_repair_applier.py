"""RepairApplier 单元测试。

覆盖要求场景：
    1. should_repair=False            9. warning 正确记录
    2. should_repair=True + 空 actions 10. deterministic
    3. file=""                        11. GeneratedArtifacts JSON round-trip
    4. unknown file                   12. 不执行文件写入
    5. create action                  13. 不执行 subprocess
    6. modify action                  14. 不执行 shell
    7. 多个 actions                   15. 不执行 git
    8. 原始 artifacts 不被修改

以及：modify 片段不可叠加、Protocol 一致性、安全跳过时保持原内容不变。
"""

import os
import subprocess
from pathlib import Path

from integration_agent import generation, repair, validation

PASSING_TEST = "def test_add():\n    assert 1 + 1 == 2\n"


def _artifacts() -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(path="tests/test_x.py", action="create", content=PASSING_TEST),
            generation.GeneratedFile(
                path="app/http.py",
                action="modify",
                content="from app.client import Client\n",
                purpose="复用既有模块",
            ),
        ],
        dependency_changes=[],
        summary="合成产物（单元测试用）",
    )


def _action(
    file: str = "tests/test_x.py",
    action: str = "modify",
    *,
    changes: str = "Add the missing module import",
) -> repair.RepairAction:
    return repair.RepairAction(
        file=file,
        action=action,  # type: ignore[arg-type]
        reason="测试失败证据",
        target="import section",
        changes=changes,
        confidence=0.5,
    )


def _plan(
    should_repair: bool = True,
    *,
    actions: list[repair.RepairAction] | None = None,
    warnings: list[str] | None = None,
) -> repair.RepairPlan:
    return repair.RepairPlan(
        should_repair=should_repair,
        failure_category="assertion" if should_repair else "none",
        analysis="分析",
        actions=actions or [],
        warnings=warnings or [],
    )


def _content(result: repair.RepairApplicationResult, path: str) -> str:
    return next(item.content for item in result.artifacts.files if item.path == path)


# ------------------------------------------------ 场景 1：should_repair=False


def test_should_repair_false_returns_unchanged_copy() -> None:
    artifacts = _artifacts()
    plan = _plan(should_repair=False, warnings=["无需修复"])

    result = repair.apply_repair(artifacts, plan)

    assert result.changed is False
    assert result.applied_actions == []
    assert result.skipped_actions == []
    assert result.warnings == ["无需修复"]
    assert result.artifacts == artifacts  # 内容等价
    assert result.artifacts is not artifacts  # 但是独立的新对象


# ---------------------------------------- 场景 2：should_repair=True + 空 actions


def test_empty_actions_warns_and_keeps_artifacts() -> None:
    result = repair.apply_repair(_artifacts(), _plan(actions=[]))

    assert result.changed is False
    assert result.applied_actions == []
    assert result.skipped_actions == []
    assert "No repair actions were provided." in result.warnings
    assert result.artifacts == _artifacts()


# -------------------------------------------------------- 场景 3：file==""


def test_empty_file_skipped_without_guessing() -> None:
    result = repair.apply_repair(_artifacts(), _plan(actions=[_action(file="")]))

    assert result.changed is False
    assert result.skipped_actions == [_action(file="")]
    assert any("无法定位目标文件" in item for item in result.warnings)
    assert result.artifacts == _artifacts()


# ------------------------------------------------------- 场景 4：unknown file


def test_unknown_file_skipped() -> None:
    result = repair.apply_repair(_artifacts(), _plan(actions=[_action(file="src/not_exist.py")]))

    assert result.changed is False
    assert len(result.skipped_actions) == 1
    assert any("src/not_exist.py" in item for item in result.warnings)
    assert result.artifacts == _artifacts()


# ------------------------------------------------------ 场景 5：create action


def test_create_action_skipped() -> None:
    result = repair.apply_repair(
        _artifacts(), _plan(actions=[_action(action="create", file="src/new.py")])
    )

    assert result.changed is False
    assert len(result.skipped_actions) == 1
    assert any("create 动作暂不受支持" in item for item in result.warnings)
    assert result.artifacts == _artifacts()  # 没有偷偷创建文件


# ------------------------------------------------------ 场景 6：modify action


def test_modify_action_with_natural_language_changes_skipped() -> None:
    """自然语言 changes 无法安全合成代码：跳过并保持原内容不变。"""
    artifacts = _artifacts()
    result = repair.apply_repair(artifacts, _plan(actions=[_action()]))

    assert result.changed is False
    assert result.applied_actions == []
    assert len(result.skipped_actions) == 1
    assert (
        "Repair action requires code synthesis and cannot be safely applied deterministically."
    ) in result.warnings
    assert _content(result, "tests/test_x.py") == PASSING_TEST  # 原内容不变


def test_modify_of_modify_snippet_skipped() -> None:
    result = repair.apply_repair(_artifacts(), _plan(actions=[_action(file="app/http.py")]))

    assert result.changed is False
    assert any("修改片段" in item for item in result.warnings)
    assert _content(result, "app/http.py") == "from app.client import Client\n"


# ---------------------------------------------------- 场景 7：多个 actions


def test_multiple_actions_processed_in_order() -> None:
    actions = [
        _action(file=""),
        _action(action="create", file="src/new.py"),
        _action(file="src/not_exist.py"),
        _action(),
    ]
    result = repair.apply_repair(_artifacts(), _plan(actions=actions))

    assert result.skipped_actions == actions  # 按原顺序完整记录
    assert result.applied_actions == []
    assert result.changed is False
    # 每条 action 至少对应一条 warning（plan.warnings 之外）
    assert len(result.warnings) == len(actions)


# ------------------------------------- 场景 8：原始 artifacts 不被修改


def test_original_artifacts_untouched() -> None:
    artifacts = _artifacts()
    before = artifacts.model_dump()

    result = repair.apply_repair(artifacts, _plan(actions=[_action(), _action(file="")]))
    # 修改返回结果不影响原始对象
    result.artifacts.files[0].content = "CHANGED"

    assert artifacts.model_dump() == before
    assert artifacts.files[0].content == PASSING_TEST


# -------------------------------------------------- 场景 9：warning 正确记录


def test_warnings_recorded() -> None:
    result = repair.apply_repair(
        _artifacts(),
        _plan(actions=[_action(file="")], warnings=["plan 的警告"]),
    )

    assert result.warnings == ["plan 的警告", "无法定位目标文件（file 为空），已跳过该修复动作"]


# ------------------------------------------------------ 场景 10：deterministic


def test_deterministic_output() -> None:
    actions = [_action(), _action(file=""), _action(action="create", file="src/x.py")]
    first = repair.apply_repair(_artifacts(), _plan(actions=actions))
    second = repair.apply_repair(_artifacts(), _plan(actions=actions))

    assert first == second
    assert first.model_dump() == second.model_dump()


# ---------------------------------------------- 场景 11：JSON round-trip


def test_result_json_roundtrip() -> None:
    result = repair.apply_repair(_artifacts(), _plan(actions=[_action(file="")]))

    restored = repair.RepairApplicationResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert isinstance(restored.artifacts, generation.GeneratedArtifacts)
    assert (
        generation.GeneratedArtifacts.model_validate_json(restored.artifacts.model_dump_json())
        == restored.artifacts
    )


# ---------------------------------------- 场景 12-15：无写入 / 无子进程


def test_applier_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    def _forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("RepairApplier 不应执行任何子进程")

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

    repair.apply_repair(_artifacts(), _plan(actions=[_action()]))

    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


# ------------------------------------------------------- Protocol 一致性


def test_protocol_conformance() -> None:
    assert isinstance(repair.DeterministicRepairApplier(), repair.RepairApplier)


# ------------------------------------------------------- 三层真实链路


def test_full_pipeline_runner_planner_applier() -> None:
    """真实链路：Runner 失败 → Planner 计划 → Applier 安全跳过（内容不变）。"""
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
    applied = repair.apply_repair(artifacts, plan)

    assert result.status == "failed"
    assert plan.should_repair is True
    assert applied.changed is False  # 确定性阶段不合成代码
    assert applied.skipped_actions == plan.actions
    assert applied.artifacts == artifacts
    assert any("requires code synthesis" in item for item in applied.warnings)
