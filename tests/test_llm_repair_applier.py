"""LLMRepairApplier 单元测试。

覆盖要求场景：
    1. LLM 返回合法 JSON → 成功修改
    2. modify 已有文件（保留元数据）
    3. create 新文件
    4. 非法 JSON            5. 缺字段 JSON
    6. LLMClient 抛异常     7. 空响应
    8. 非法绝对路径         9. ../ 路径
    10. Windows 绝对路径    11. modify 文件不存在
    12. create 已存在文件    13. 超过最大修改数量
    14. 单文件内容过大       15. 总内容过大
    16. should_repair=False（不调用 LLM）
    17. 原始 artifacts 不被修改
    18. deterministic        19. warning 正确保留
    20. RepairApplicationResult JSON round-trip

以及：重复 path、Protocol 一致性、prompt 上下文受限、无副作用、
可注入现有 RepairLoop（2 参调用兼容）、
授权白名单（越权 modify / create 一律被拒且产物不变，端到端不能让 pytest 假通过）。
"""

import json
import os
import subprocess
from pathlib import Path

from integration_agent import generation, repair, validation


def _artifacts() -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(
                path="src/client.py",
                action="create",
                content="VALUE = 1\n",
                purpose="客户端",
                insertion_point=None,
                changes=["初始生成"],
            ),
            generation.GeneratedFile(
                path="tests/test_x.py", action="create", content="UNRELATED = True\n"
            ),
        ],
        summary="合成产物（单元测试用）",
    )


def _plan(
    should_repair: bool = True,
    *,
    file: str = "src/client.py",
    files: list[str] | None = None,
) -> repair.RepairPlan:
    """构造修复计划；actions 中的 file 集合就是 LLM 被授权改动的全部文件。"""
    targets = files if files is not None else [file]
    return repair.RepairPlan(
        should_repair=should_repair,
        failure_category="assertion",
        analysis="断言失败",
        actions=[]
        if not should_repair
        else [
            repair.RepairAction(
                file=target,
                action="modify",
                reason="断言失败",
                target="line 1",
                changes="修正断言",
                confidence=0.6,
                evidence=["failure message: assert 1 == 2"],
            )
            for target in targets
        ],
        warnings=["plan 的警告"],
    )


def _result() -> validation.TestResult:
    return validation.TestResult(
        status="failed",
        exit_code=1,
        failed=1,
        duration=0.1,
        failure_details=[
            validation.FailureDetail(
                test_name="test_x", file="tests/test_x.py", line=2, message="assert 1 == 2"
            )
        ],
    )


def _applier(responses: str | list[str], **kwargs) -> repair.StructuredLLMRepairApplier:
    return repair.StructuredLLMRepairApplier(repair.FakeLLMClient(responses), **kwargs)


def _modify_response(path: str = "src/client.py", content: str = "VALUE = 2\n") -> str:
    return json.dumps(
        {
            "changes": [
                {"path": path, "action": "modify", "content": content, "reason": "修正断言"}
            ],
            "summary": "已修复",
            "warnings": [],
        }
    )


def _content(result: repair.RepairApplicationResult, path: str) -> str:
    return next(item.content for item in result.artifacts.files if item.path == path)


# --------------------------------------- 场景 1：LLM 返回合法 JSON → 修改


def test_valid_json_modifies_file() -> None:
    result = _applier(_modify_response()).apply(_artifacts(), _plan(), _result())

    assert result.changed is True
    assert _content(result, "src/client.py") == "VALUE = 2\n"
    assert len(result.applied_actions) == 1
    assert result.applied_actions[0].file == "src/client.py"
    assert result.applied_actions[0].confidence == 0.6  # 复用 plan 中该文件的置信度
    assert result.skipped_actions == []


# ------------------------------------ 场景 2：modify 保留元数据并更新 changes


def test_modify_preserves_metadata() -> None:
    result = _applier(_modify_response()).apply(_artifacts(), _plan(), _result())

    target = next(item for item in result.artifacts.files if item.path == "src/client.py")
    assert target.action == "create"
    assert target.purpose == "客户端"
    assert target.changes == ["初始生成", "修正断言"]  # 追加 LLM 的修改原因
    assert _content(result, "tests/test_x.py") == "UNRELATED = True\n"  # 无关文件不动


# ------------------------------------------------- 场景 3：create 新文件


def test_create_new_file() -> None:
    response = json.dumps(
        {
            "changes": [
                {
                    "path": "src/new_module.py",
                    "action": "create",
                    "content": "def helper():\n    return True\n",
                    "reason": "补充缺失模块",
                }
            ],
            "summary": "新增模块",
            "warnings": [],
        }
    )
    # create 同样需要授权：计划里明确要求新增这个文件才允许写。
    result = _applier(response).apply(_artifacts(), _plan(files=["src/new_module.py"]), _result())

    assert result.changed is True
    new_file = next(item for item in result.artifacts.files if item.path == "src/new_module.py")
    assert new_file.action == "create"
    assert new_file.content == "def helper():\n    return True\n"
    assert "LLM 修复生成" in new_file.purpose


# -------------------------------------------- 场景 4-5：非法 / 缺字段 JSON


def test_invalid_json_rejected() -> None:
    result = _applier("not json at all").apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("非法 JSON" in item for item in result.warnings)
    assert result.artifacts == _artifacts()


def test_missing_fields_rejected() -> None:
    response = json.dumps({"changes": [{"path": "src/client.py"}]})  # 缺 action/content
    result = _applier(response).apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("Pydantic 校验" in item for item in result.warnings)
    assert result.artifacts == _artifacts()


# --------------------------------------- 场景 6-7：LLM 异常 / 空响应


def test_llm_client_exception_handled() -> None:
    class RaisingClient:
        def generate(self, prompt: str) -> str:
            raise ConnectionError("boom")

    applier = repair.StructuredLLMRepairApplier(RaisingClient())
    result = applier.apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("LLMClient 调用失败" in item for item in result.warnings)
    assert "ConnectionError" in result.warnings[-1]


def test_empty_response_handled() -> None:
    result = _applier("").apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("空响应" in item for item in result.warnings)


# ---------------------------------------- 场景 8-10：非法 path 校验


def test_absolute_unix_path_rejected() -> None:
    result = _applier(_modify_response(path="/etc/passwd")).apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert len(result.skipped_actions) == 1
    assert any("绝对路径" in item for item in result.warnings)


def test_parent_traversal_path_rejected() -> None:
    result = _applier(_modify_response(path="../evil.py")).apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("越界" in item for item in result.warnings)


def test_windows_absolute_path_rejected() -> None:
    result = _applier(_modify_response(path="C:\\evil.py")).apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("Windows 绝对路径" in item for item in result.warnings)


# --------------------------------------- 场景 11-12：未知文件 / 已存在文件


def test_modify_unknown_file_skipped_not_created() -> None:
    # 已授权但产物里没有该文件：授权过了也不能 modify 凭空造出文件。
    result = _applier(_modify_response(path="src/ghost.py")).apply(
        _artifacts(), _plan(files=["src/ghost.py"]), _result()
    )

    assert result.changed is False
    assert len(result.skipped_actions) == 1
    assert any("不存在" in item for item in result.warnings)
    assert all(item.path != "src/ghost.py" for item in result.artifacts.files)


def test_create_existing_file_skipped() -> None:
    response = json.dumps(
        {
            "changes": [
                {"path": "src/client.py", "action": "create", "content": "x\n", "reason": ""}
            ],
            "summary": "",
            "warnings": [],
        }
    )
    result = _applier(response).apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("已存在" in item for item in result.warnings)


# --------------------------------------------------- 场景 13：数量上限


def test_max_changes_enforced() -> None:
    changes = [
        {
            "path": f"src/mod_{index}.py",
            "action": "create",
            "content": f"X = {index}\n",
            "reason": "",
        }
        for index in range(6)
    ]
    applier = repair.StructuredLLMRepairApplier(
        repair.FakeLLMClient(json.dumps({"changes": changes, "summary": "", "warnings": []})),
        max_changes=5,
    )
    result = applier.apply(
        _artifacts(), _plan(files=[f"src/mod_{index}.py" for index in range(6)]), _result()
    )

    assert len(result.applied_actions) == 5
    assert len(result.skipped_actions) == 1
    assert any("数量超过上限" in item for item in result.warnings)


def test_duplicate_paths_rejected() -> None:
    response = json.dumps(
        {
            "changes": [
                {"path": "src/client.py", "action": "modify", "content": "A\n", "reason": ""},
                {"path": "src/client.py", "action": "modify", "content": "B\n", "reason": ""},
            ],
            "summary": "",
            "warnings": [],
        }
    )
    result = _applier(response).apply(_artifacts(), _plan(), _result())

    assert len(result.applied_actions) == 1
    assert any("重复修改" in item for item in result.warnings)
    assert _content(result, "src/client.py") == "A\n"


# ---------------------------------------------- 场景 14-15：内容大小上限


def test_single_file_too_large_skipped() -> None:
    applier = repair.StructuredLLMRepairApplier(
        repair.FakeLLMClient(_modify_response(content="X" * 30)),
        max_file_content_chars=10,
    )
    result = applier.apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert any("单文件上限" in item for item in result.warnings)


def test_total_content_too_large_skipped() -> None:
    changes = [
        {"path": "src/a.py", "action": "create", "content": "A" * 15, "reason": ""},
        {"path": "src/b.py", "action": "create", "content": "B" * 15, "reason": ""},
    ]
    applier = repair.StructuredLLMRepairApplier(
        repair.FakeLLMClient(json.dumps({"changes": changes, "summary": "", "warnings": []})),
        max_total_content_chars=20,
    )
    result = applier.apply(_artifacts(), _plan(files=["src/a.py", "src/b.py"]), _result())

    assert len(result.applied_actions) == 1
    assert any("总上限" in item for item in result.warnings)


# ------------------------------------------ 场景 16：should_repair=False


def test_should_repair_false_does_not_call_llm() -> None:
    client = repair.FakeLLMClient(_modify_response())
    applier = repair.StructuredLLMRepairApplier(client)
    result = applier.apply(_artifacts(), _plan(should_repair=False), _result())

    assert client.calls == 0
    assert result.changed is False
    assert any("should_repair=False" in item for item in result.warnings)
    assert result.artifacts == _artifacts()


# ------------------------------------ 场景 17：原始 artifacts 不被修改


def test_original_artifacts_untouched() -> None:
    artifacts = _artifacts()
    before = artifacts.model_dump()

    result = _applier(_modify_response()).apply(artifacts, _plan(), _result())
    result.artifacts.files[0].content = "MUTATED"

    assert artifacts.model_dump() == before
    assert artifacts.files[0].content == "VALUE = 1\n"
    assert result.artifacts is not artifacts


# -------------------------------------------------- 场景 18：deterministic


def test_deterministic_output() -> None:
    def run_once():
        return _applier(_modify_response()).apply(_artifacts(), _plan(), _result())

    assert run_once() == run_once()


# -------------------------------------------------- 场景 19：warnings 保留


def test_warnings_preserved() -> None:
    response = json.dumps(
        {
            "changes": [
                {"path": "../bad.py", "action": "create", "content": "x\n", "reason": ""},
                {"path": "src/client.py", "action": "modify", "content": "OK\n", "reason": "r"},
            ],
            "summary": "",
            "warnings": ["LLM 的警告"],
        }
    )
    result = _applier(response).apply(_artifacts(), _plan(), _result())

    assert result.warnings[0] == "plan 的警告"
    assert "LLM 的警告" in result.warnings
    assert any("越界" in item for item in result.warnings)
    assert result.changed is True  # 合法的那条仍然应用


# --------------------------------------------- 场景 20：JSON round-trip


def test_result_json_roundtrip() -> None:
    result = _applier(_modify_response()).apply(_artifacts(), _plan(), _result())

    restored = repair.RepairApplicationResult.model_validate_json(result.model_dump_json())
    assert restored == result


# ---------------------- 授权白名单：只能改 RepairPlan.actions 列出的文件（P0-1 回归）
#
# 修复前的逃逸链：_generated_tests 把测试文件以 action="create" 写进产物 →
# TestRunner 会落地并执行它们 → LLM 在 modify 时不受任何限制，可以把断言改成
# assert True → 真实 pytest 由 failed 变 passed → Pipeline 返回 status="passed"。
# 因此授权检查必须对 modify 与 create 同等生效，且与置信度、prompt 提示无关。


def _gutting_response(path: str = "tests/test_x.py") -> str:
    """把测试文件改成恒真（真实攻击里 LLM 会这样"修复"失败）。"""
    return json.dumps(
        {
            "changes": [
                {
                    "path": path,
                    "action": "modify",
                    "content": "def test_everything_is_fine():\n    assert True\n",
                    "reason": "让测试通过",
                }
            ],
            "summary": "已修复",
            "warnings": [],
        }
    )


def test_unauthorized_test_file_modify_rejected() -> None:
    """计划只授权 src/client.py：改动 tests/test_x.py 必须被拒，产物一字节不变。"""
    artifacts = _artifacts()
    before = artifacts.model_dump()

    result = _applier(_gutting_response()).apply(artifacts, _plan(), _result())

    assert result.changed is False
    assert result.applied_actions == []
    assert len(result.skipped_actions) == 1
    assert result.skipped_actions[0].file == "tests/test_x.py"
    assert any("授权" in item for item in result.warnings)
    # 原对象与返回值都不能留下被拒 change 的痕迹
    assert artifacts.model_dump() == before
    assert result.artifacts.model_dump() == before
    assert _content(result, "tests/test_x.py") == "UNRELATED = True\n"


def test_unauthorized_create_rejected() -> None:
    """create 不豁免：新建未授权文件同样被拒（否则可塞进 conftest.py 等改变测试行为）。"""
    artifacts = _artifacts()
    before = artifacts.model_dump()
    response = json.dumps(
        {
            "changes": [
                {
                    "path": "tests/conftest.py",
                    "action": "create",
                    "content": "COLLECT_IGNORE = True\n",
                    "reason": "",
                }
            ],
            "summary": "",
            "warnings": [],
        }
    )

    result = _applier(response).apply(artifacts, _plan(), _result())

    assert result.changed is False
    assert result.applied_actions == []
    assert result.skipped_actions[0].file == "tests/conftest.py"
    assert artifacts.model_dump() == before
    assert all(item.path != "tests/conftest.py" for item in result.artifacts.files)


def test_authorized_source_file_still_modifiable() -> None:
    """授权文件照常可改：授权检查没有误伤正常 Repair 能力。"""
    result = _applier(_modify_response()).apply(_artifacts(), _plan(), _result())

    assert result.changed is True
    assert [item.file for item in result.applied_actions] == ["src/client.py"]
    assert result.skipped_actions == []
    assert _content(result, "src/client.py") == "VALUE = 2\n"


def _multi_artifacts() -> generation.GeneratedArtifacts:
    """三份产物：src/client.py 与 src/models.py 可被授权，tests/test_x.py 不被授权。"""
    artifacts = _artifacts()
    artifacts.files.append(
        generation.GeneratedFile(path="src/models.py", action="create", content="MODEL = 1\n")
    )
    return artifacts


def test_unauthorized_change_does_not_block_authorized_ones() -> None:
    """混合批次：两份授权 change 照常应用，未授权的测试文件只跳过它自己，不抛异常。"""
    response = json.dumps(
        {
            "changes": [
                {
                    "path": "src/client.py",
                    "action": "modify",
                    "content": "VALUE = 2\n",
                    "reason": "修正断言",
                },
                {
                    "path": "tests/test_x.py",
                    "action": "modify",
                    "content": "assert True\n",
                    "reason": "",
                },
                {
                    "path": "src/models.py",
                    "action": "modify",
                    "content": "MODEL = 2\n",
                    "reason": "",
                },
            ],
            "summary": "",
            "warnings": [],
        }
    )

    result = _applier(response).apply(
        _multi_artifacts(), _plan(files=["src/client.py", "src/models.py"]), _result()
    )

    assert result.changed is True
    assert [item.file for item in result.applied_actions] == ["src/client.py", "src/models.py"]
    assert [item.file for item in result.skipped_actions] == ["tests/test_x.py"]
    assert _content(result, "src/client.py") == "VALUE = 2\n"
    assert _content(result, "src/models.py") == "MODEL = 2\n"
    assert _content(result, "tests/test_x.py") == "UNRELATED = True\n"


def test_unauthorized_source_file_create_rejected() -> None:
    """非 tests/ 路径同样受管：未授权的源码文件不能靠 create 混进产物。"""
    response = json.dumps(
        {
            "changes": [
                {
                    "path": "src/other_module.py",
                    "action": "create",
                    "content": "EXTRA = 1\n",
                    "reason": "顺手加的",
                }
            ],
            "summary": "",
            "warnings": [],
        }
    )

    result = _applier(response).apply(_artifacts(), _plan(), _result())

    assert result.changed is False
    assert result.skipped_actions[0].file == "src/other_module.py"
    assert all(item.path != "src/other_module.py" for item in result.artifacts.files)


def test_authorized_tests_change_still_applied() -> None:
    """不是"tests/ 一律禁止"：计划授权了测试文件时，改动必须照常生效。"""
    result = _applier(_gutting_response()).apply(
        _artifacts(), _plan(files=["tests/test_x.py"]), _result()
    )

    assert result.changed is True
    assert result.skipped_actions == []
    assert _content(result, "tests/test_x.py") == (
        "def test_everything_is_fine():\n    assert True\n"
    )


# --------------------------- P0-1 端到端：被拒的改动不能让真实 pytest 假通过


def _e2e_artifacts() -> generation.GeneratedArtifacts:
    """一份真实可跑的产物：helper.value() 返回 1，测试期望 2 → pytest 必然 failed。"""
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(
                path="helper.py", action="create", content="def value():\n    return 1\n"
            ),
            generation.GeneratedFile(
                path="tests/test_helper.py",
                action="create",
                content=(
                    "from helper import value\n\n\ndef test_value():\n    assert value() == 2\n"
                ),
            ),
        ],
        summary="合成产物（端到端回归用）",
    )


def _e2e_plan(file: str) -> repair.RepairPlan:
    return repair.RepairPlan(
        should_repair=True,
        failure_category="assertion",
        analysis="断言失败",
        actions=[
            repair.RepairAction(
                file=file,
                action="modify",
                reason="断言失败",
                target="line 5",
                changes="修正实现或断言",
                confidence=0.6,
                evidence=["failure message: assert 1 == 2"],
            )
        ],
        warnings=[],
    )


def test_unauthorized_test_file_cannot_fake_a_pass() -> None:
    """端到端：只授权 helper.py 时，改测试文件被拒 → 重跑真实 pytest 仍然 failed。"""
    result = _applier(_gutting_response("tests/test_helper.py")).apply(
        _e2e_artifacts(), _e2e_plan("helper.py"), _result()
    )

    assert result.applied_actions == []
    assert result.changed is False

    rerun = validation.DeterministicTestRunner().run(result.artifacts)
    assert rerun.status == "failed"
    assert rerun.failed == 1


def test_authorized_test_change_does_turn_it_green() -> None:
    """对照实验：同一改动被计划授权后就真的能让 pytest 通过——所以这道门是必需的。

    本用例证明 P0-1 不是理论问题：没有授权检查时，LLM 只要改测试文件即可
    把 failed 变成 passed，从而让 Pipeline 返回 status="passed"。
    """
    result = _applier(_gutting_response("tests/test_helper.py")).apply(
        _e2e_artifacts(), _e2e_plan("tests/test_helper.py"), _result()
    )

    assert result.changed is True
    rerun = validation.DeterministicTestRunner().run(result.artifacts)
    assert rerun.status == "passed"


# ------------------------------- Repair Loop：非法 change 不能让循环崩溃


class _QueueRunner:
    """按顺序返回预设测试结果，不真正执行 pytest。"""

    def __init__(self, results: list[validation.TestResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run(self, artifacts: generation.GeneratedArtifacts) -> validation.TestResult:
        self.calls += 1
        return self._results.pop(0)


class _OnePlanPlanner:
    """无论输入怎样都返回同一份计划。"""

    def __init__(self, plan: repair.RepairPlan) -> None:
        self._plan = plan

    def plan(self, artifacts, test_result) -> repair.RepairPlan:  # noqa: ANN001
        return self._plan


def test_repair_loop_survives_all_unauthorized_changes() -> None:
    """全部 change 越权 → 循环以 no_progress 正常收敛，绝不崩溃、绝不误报 passed。"""
    runner = _QueueRunner([_result()])
    loop = repair.RepairLoopRunner(
        test_runner=runner,
        repair_planner=_OnePlanPlanner(_plan()),
        repair_applier=_applier(_gutting_response()),
        max_iterations=3,
    )

    result = loop.run(_artifacts())

    assert result.status == "no_progress"
    assert runner.calls == 1  # changed=False 时不再重复同一计划
    assert any("授权" in item for item in result.application_results[0].warnings)
    assert _content(result.application_results[0], "tests/test_x.py") == "UNRELATED = True\n"


def test_repair_loop_applies_authorized_and_skips_unauthorized() -> None:
    """越权 change 与合法 change 同批：循环照常收敛到 passed，测试文件保持原样。"""
    runner = _QueueRunner(
        [_result(), validation.TestResult(status="passed", exit_code=0, passed=1, duration=0.1)]
    )
    response = json.dumps(
        {
            "changes": [
                {"path": "tests/test_x.py", "action": "modify", "content": "assert True\n"},
                {"path": "src/client.py", "action": "modify", "content": "VALUE = 9\n"},
            ],
            "summary": "",
            "warnings": [],
        }
    )
    loop = repair.RepairLoopRunner(
        test_runner=runner,
        repair_planner=_OnePlanPlanner(_plan()),
        repair_applier=_applier(response),
        max_iterations=3,
    )

    result = loop.run(_artifacts())

    assert result.status == "passed"
    assert result.iterations == 1
    application = result.application_results[0]
    assert [item.file for item in application.applied_actions] == ["src/client.py"]
    assert [item.file for item in application.skipped_actions] == ["tests/test_x.py"]
    assert _content(application, "src/client.py") == "VALUE = 9\n"
    assert _content(application, "tests/test_x.py") == "UNRELATED = True\n"


# ------------------------------------------------------- Protocol 一致性


def test_protocol_conformance() -> None:
    assert isinstance(_applier(_modify_response()), repair.LLMRepairApplier)
    assert isinstance(repair.FakeLLMClient(""), repair.LLMClient)


# ------------------------------------------------------ prompt 上下文受限


def test_prompt_is_bounded() -> None:
    client = repair.FakeLLMClient(_modify_response())
    _applier(_modify_response())  # 占位，避免误用
    applier = repair.StructuredLLMRepairApplier(client)
    applier.apply(_artifacts(), _plan(), _result())

    prompt = client.prompts[0]
    assert "failure_category: assertion" in prompt
    assert "assert 1 == 2" in prompt  # 失败信息
    assert "VALUE = 1" in prompt  # 目标文件内容
    assert "UNRELATED = True" not in prompt  # 与计划无关的文件不进 prompt


# ----------------------------------------------------------- 无副作用


def test_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    def _forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("LLMRepairApplier 不应执行任何子进程")

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

    _applier(_modify_response()).apply(_artifacts(), _plan(), _result())

    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


# ----------------------------------------------- 可注入现有 RepairLoop


def test_injectable_into_repair_loop() -> None:
    """LLMRepairApplier 可直接注入 RepairLoop（Loop 只传 (artifacts, plan)）。"""
    failed = _result()
    passed = validation.TestResult(status="passed", exit_code=0, passed=1, duration=0.1)

    class QueueRunner:
        def __init__(self) -> None:
            self._results = [failed, passed]
            self.calls = 0

        def run(self, artifacts):
            self.calls += 1
            return self._results.pop(0)

    class OnePlanPlanner:
        def plan(self, artifacts, test_result):
            return _plan()

    loop = repair.RepairLoopRunner(
        test_runner=QueueRunner(),
        repair_planner=OnePlanPlanner(),
        repair_applier=_applier(_modify_response(content="VALUE = 9\n")),
        max_iterations=3,
    )
    result = loop.run(_artifacts())

    assert result.status == "passed"
    assert result.iterations == 1
    assert _content(result.application_results[0], "src/client.py") == "VALUE = 9\n"
