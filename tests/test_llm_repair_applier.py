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
可注入现有 RepairLoop（2 参调用兼容）。
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


def _plan(should_repair: bool = True, *, file: str = "src/client.py") -> repair.RepairPlan:
    return repair.RepairPlan(
        should_repair=should_repair,
        failure_category="assertion",
        analysis="断言失败",
        actions=[]
        if not should_repair
        else [
            repair.RepairAction(
                file=file,
                action="modify",
                reason="断言失败",
                target="line 1",
                changes="修正断言",
                confidence=0.6,
                evidence=["failure message: assert 1 == 2"],
            )
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
    result = _applier(response).apply(_artifacts(), _plan(), _result())

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
    result = _applier(_modify_response(path="src/ghost.py")).apply(_artifacts(), _plan(), _result())

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
    result = applier.apply(_artifacts(), _plan(), _result())

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
    result = applier.apply(_artifacts(), _plan(), _result())

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
