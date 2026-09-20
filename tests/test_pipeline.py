"""Pipeline 编排器单元测试。

覆盖要求场景：
    1. 真实 fixture 的完整确定性 Pipeline（→ tests_failed / no_progress）
    2. Fake TestRunner：第一次失败、第二次通过 → passed
    3. Fake RepairApplier：产生 changed=True
    4. FakeLLMClient + StructuredLLMRepairApplier 装配
    5. parse 异常 / 6. scan 异常 / 7. plan 异常 / 8. generate 异常
    9. Pipeline Protocol runtime check
    10. PipelineResult JSON round-trip
    11. 输入 OpenAPI 文件不会被修改
    12. repo fixture 不会被修改
    13. 相同输入下结果保持 deterministic（fake 组件，全字段相等）
"""

import json
from pathlib import Path

import pytest

from integration_agent import generation, pipeline, repair, validation

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PETSTORE_PATH = EXAMPLES / "openapi" / "petstore.yaml"
DEMO_PROJECT_PATH = EXAMPLES / "demo_project"


def _failed(file: str = "tests/test_x.py") -> validation.TestResult:
    return validation.TestResult(
        status="failed",
        exit_code=1,
        failed=1,
        duration=0.1,
        failure_details=[
            validation.FailureDetail(test_name="test_x", file=file, line=2, message="assert 1 == 2")
        ],
    )


def _passed() -> validation.TestResult:
    return validation.TestResult(status="passed", exit_code=0, passed=1, duration=0.1)


def _assert_no_internal_details(result: pipeline.PipelineResult) -> None:
    """stage 异常返回给客户端的内容里不得有 traceback / 绝对路径 / 异常细节。

    PipelineResult 是 FastAPI 的 response_model，会原样序列化给浏览器，
    因此这些断言等价于"客户端看不到内部实现信息"。
    """
    client_visible = [result.error or "", *result.warnings]
    joined = "\n".join(client_visible)
    assert "Traceback" not in joined
    assert 'File "' not in joined
    assert str(EXAMPLES) not in joined  # 本地绝对路径
    assert "site-packages" not in joined


class FakeRunner:
    """按队列返回 TestResult 的假 Runner。"""

    def __init__(self, results: list[validation.TestResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run(self, artifacts) -> validation.TestResult:
        self.calls += 1
        assert self._results, "runner 被调用的次数超过预期"
        return self._results.pop(0)


class FakeApplier:
    """返回固定 RepairApplicationResult 的假 Applier。"""

    def __init__(self, result: repair.RepairApplicationResult) -> None:
        self._result = result
        self.calls = 0

    def apply(self, artifacts, plan) -> repair.RepairApplicationResult:
        self.calls += 1
        return self._result


class RaisingGenerator:
    def generate(self, plan) -> generation.GeneratedArtifacts:
        raise RuntimeError("gen boom")


def _snapshot_file(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _snapshot_dir(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def real_pipeline_run():
    """真实确定性 Pipeline 运行一次，供多个测试共享（含前后快照）。"""
    before_spec = _snapshot_file(PETSTORE_PATH)
    before_repo = _snapshot_dir(DEMO_PROJECT_PATH)
    result = pipeline.run_pipeline(PETSTORE_PATH, DEMO_PROJECT_PATH)
    after_spec = _snapshot_file(PETSTORE_PATH)
    after_repo = _snapshot_dir(DEMO_PROJECT_PATH)
    return result, before_spec, after_spec, before_repo, after_repo


# -------------------------------------- 场景 1：真实 fixture 确定性 Pipeline


def test_real_deterministic_pipeline(real_pipeline_run) -> None:
    result, *_ = real_pipeline_run

    # 环境提供 httpx（dev 依赖 + fixture 声明）后：生成测试正常执行并全部通过，
    # 初始测试即 passed，无需进入修复。
    assert result.status == "passed"
    assert result.failed_stage is None
    assert result.api is not None and result.api.name == "Demo Petstore API"
    assert result.project is not None and result.project.name == "demo-project"
    assert result.plan is not None
    assert result.artifacts is not None
    assert result.artifacts.files
    assert result.initial_test_result is None  # 不重复跑测试
    assert result.repair_loop_result is not None
    assert result.repair_loop_result.status == "passed"
    assert result.repair_loop_result.iterations == 0
    assert result.repair_loop_result.test_result.passed == 10


# --------------------------------------- 场景 2-3：Fake Runner / Applier


def test_fake_runner_pass_after_repair() -> None:
    runner = FakeRunner([_failed(), _passed()])
    fixed = generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(path="tests/test_x.py", action="create", content="# FIXED\n")
        ],
        summary="fake",
    )
    applier = FakeApplier(repair.RepairApplicationResult(artifacts=fixed, changed=True))

    result = pipeline.run_pipeline(
        PETSTORE_PATH, DEMO_PROJECT_PATH, test_runner=runner, repair_applier=applier
    )

    assert result.status == "passed"
    assert runner.calls == 2
    assert applier.calls == 1
    assert result.repair_loop_result.status == "passed"
    assert result.artifacts == fixed  # 修复后的产物被保留


# --------------------------------- 场景 4：FakeLLMClient + LLM Applier 装配


def test_llm_applier_injection() -> None:
    runner = FakeRunner([_failed(file="demo_project/demo_petstore_client.py"), _passed()])
    llm_response = json.dumps(
        {
            "changes": [
                {
                    "path": "demo_project/demo_petstore_client.py",
                    "action": "modify",
                    "content": "# LLM FIXED\n",
                    "reason": "修复断言",
                }
            ],
            "summary": "已修复",
            "warnings": [],
        }
    )
    llm_applier = repair.StructuredLLMRepairApplier(repair.FakeLLMClient(llm_response))

    result = pipeline.run_pipeline(
        PETSTORE_PATH, DEMO_PROJECT_PATH, test_runner=runner, repair_applier=llm_applier
    )

    assert result.status == "passed"
    client_file = next(
        item
        for item in result.artifacts.files
        if item.path == "demo_project/demo_petstore_client.py"
    )
    assert client_file.content == "# LLM FIXED\n"


# ------------------------------------------ 场景 5-8：前置阶段异常


def test_parse_error(tmp_path: Path) -> None:
    result = pipeline.run_pipeline(tmp_path / "missing.yaml", DEMO_PROJECT_PATH)

    assert result.status == "error"
    assert result.failed_stage == "parse"
    assert result.api is None
    assert result.project is None
    assert result.error == "Integration failed during parse."
    assert any("parse 阶段失败" in item for item in result.warnings)
    _assert_no_internal_details(result)


def test_scan_error(tmp_path: Path) -> None:
    result = pipeline.run_pipeline(PETSTORE_PATH, tmp_path / "no_such_repo")

    assert result.status == "error"
    assert result.failed_stage == "scan"
    assert result.api is not None  # 上游结果保留
    assert result.project is None
    assert result.error == "Integration failed during scan."
    _assert_no_internal_details(result)


def test_plan_error(tmp_path: Path) -> None:
    spec = tmp_path / "empty.yaml"
    spec.write_text(
        'openapi: 3.0.3\ninfo: {title: Empty, version: "1"}\npaths: {}\n',
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "repo"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )

    result = pipeline.run_pipeline(spec, repo)

    assert result.status == "error"
    assert result.failed_stage == "plan"
    assert result.api is not None and result.project is not None
    assert result.error == "Integration failed during plan."
    _assert_no_internal_details(result)


def test_generate_error() -> None:
    result = pipeline.run_pipeline(
        PETSTORE_PATH, DEMO_PROJECT_PATH, code_generator=RaisingGenerator()
    )

    assert result.status == "error"
    assert result.failed_stage == "generate"
    assert result.plan is not None  # 上游结果保留
    assert result.artifacts is None
    # 异常原文（"gen boom"）不得出现在客户端可见字段中
    assert result.error == "Integration failed during generate."
    _assert_no_internal_details(result)


# --------------------------------------- 场景 9-10：Protocol / round-trip


def test_protocol_conformance() -> None:
    assert isinstance(pipeline.IntegrationPipeline(), pipeline.Pipeline)


def test_result_json_roundtrip() -> None:
    result = pipeline.run_pipeline(
        PETSTORE_PATH,
        DEMO_PROJECT_PATH,
        test_runner=FakeRunner([_failed(), _passed()]),
        repair_applier=FakeApplier(
            repair.RepairApplicationResult(
                artifacts=generation.GeneratedArtifacts(files=[], summary="fake"),
                changed=True,
            )
        ),
    )
    restored = pipeline.PipelineResult.model_validate_json(result.model_dump_json())
    assert restored == result


# ----------------------------------------- 场景 11-12：输入不被修改


def test_spec_file_not_modified(real_pipeline_run) -> None:
    _, before, after, _, _ = real_pipeline_run
    assert before == after


def test_repo_fixture_not_modified(real_pipeline_run) -> None:
    _, _, _, before, after = real_pipeline_run
    assert before == after


# ---------------------------------------------- 场景 13：deterministic


def test_deterministic_output() -> None:
    def run_once():
        return pipeline.run_pipeline(
            PETSTORE_PATH,
            DEMO_PROJECT_PATH,
            test_runner=FakeRunner([_failed(), _passed()]),
            repair_applier=FakeApplier(
                repair.RepairApplicationResult(
                    artifacts=generation.GeneratedArtifacts(files=[], summary="fake"),
                    changed=True,
                )
            ),
        )

    assert run_once() == run_once()


# --------------------------- 场景 14-18：RepairLoop.status → PipelineResult.status
#
# invariant：RepairLoopResult.status == "error" ⇒ PipelineResult.status == "error"。
# error 表示"流程自身坏了"（TestRunner / RepairPlanner / RepairApplier 抛异常），
# tests_failed 表示"测试确实跑了并且没通过"。二者混同会丢掉"该重试还是该改代码"
# 的判断依据，也会让不可信的测试结论被当成真实结论。

# failed_stage 的合法取值（parse/scan/plan/generate/repair）：repair 由本组测试引入。
LEGAL_STAGES = {"parse", "scan", "plan", "generate", "repair"}


class RaisingApplier:
    """apply 时抛异常的假 Applier：制造真实的 RepairLoopResult(status="error")。"""

    def apply(self, artifacts, plan) -> repair.RepairApplicationResult:
        raise RuntimeError("applier boom: C:/secret/path.py")


class OnePlanPlanner:
    """固定返回同一份 RepairPlan 的假 Planner。"""

    def __init__(self, plan: repair.RepairPlan) -> None:
        self._plan = plan
        self.calls = 0

    def plan(self, artifacts, test_result) -> repair.RepairPlan:
        self.calls += 1
        return self._plan


def _not_repairable_plan() -> repair.RepairPlan:
    """should_repair=False 的计划（actions 必须为空，由模型校验）。"""
    return repair.RepairPlan(
        should_repair=False,
        failure_category="assertion",
        analysis="断言失败",
        actions=[],
    )


def _run_with_applier(applier) -> pipeline.PipelineResult:
    return pipeline.run_pipeline(
        PETSTORE_PATH,
        DEMO_PROJECT_PATH,
        test_runner=FakeRunner([_failed()]),
        repair_applier=applier,
    )


# ---------------------------------------------------------------- Test 1 + Test 4


def test_repair_loop_error_maps_to_pipeline_error() -> None:
    """RepairLoop 内部异常必须原样保留为 error，不得被压缩成 tests_failed。"""
    result = _run_with_applier(RaisingApplier())

    # 前置条件：RepairLoop 自己确实返回了 error（不是我们构造的假终态）
    assert result.repair_loop_result is not None
    assert result.repair_loop_result.status == "error"

    assert result.status == "error"
    assert result.status != "tests_failed"
    # 产物仍然保留，便于事后排查
    assert result.artifacts is not None


def test_repair_loop_error_sets_legal_repair_stage() -> None:
    """failed_stage 必须定位到 repair，且取值在项目既有的 stage 命名体系内。"""
    result = _run_with_applier(RaisingApplier())

    assert result.failed_stage == "repair"
    assert result.failed_stage in LEGAL_STAGES


def test_repair_error_distinguishable_from_test_failure() -> None:
    """同一个失败输入，两种成因必须给出不同的 status（这正是本修复的目的）。"""
    loop_error = _run_with_applier(RaisingApplier())
    test_failed = _run_with_applier(
        FakeApplier(
            repair.RepairApplicationResult(
                artifacts=generation.GeneratedArtifacts(files=[], summary="fake"),
                changed=False,  # 有计划但应用层无产出 → no_progress
            )
        )
    )

    assert loop_error.status == "error"
    assert test_failed.status == "tests_failed"
    assert loop_error.status != test_failed.status
    assert loop_error.failed_stage == "repair"
    assert test_failed.failed_stage is None


# ------------------------------------------------------------------------- Test 5


def test_repair_error_details_preserved_without_leaking() -> None:
    """错误信息不因映射而丢失，同时不把 traceback / 绝对路径 / 异常原文带到响应里。"""
    result = _run_with_applier(RaisingApplier())

    # 信息保留：安全文案提升到 Pipeline 层，循环的 warnings 也照常汇聚
    assert result.error == result.repair_loop_result.error
    assert result.error
    assert result.repair_loop_result.warnings
    for warning in result.repair_loop_result.warnings:
        assert warning in result.warnings
    # 嵌套结构里的错误信息仍在（可追溯是哪个组件坏的）
    assert "RepairApplier" in (result.repair_loop_result.error or "")

    # 不泄露：沿用项目既有的安全处理
    _assert_no_internal_details(result)
    joined = "\n".join([result.error or "", *result.warnings])
    assert "applier boom" not in joined  # 异常原文（可能夹带路径）
    assert "secret" not in joined


# ----------------------------------------------------------------- Test 2 + Test 3


def test_repair_loop_passed_still_maps_to_passed() -> None:
    """Test 2：Repair Loop 修好了 → passed，且没有 failed_stage / error。"""
    result = pipeline.run_pipeline(
        PETSTORE_PATH,
        DEMO_PROJECT_PATH,
        test_runner=FakeRunner([_failed(), _passed()]),
        repair_applier=FakeApplier(
            repair.RepairApplicationResult(
                artifacts=generation.GeneratedArtifacts(files=[], summary="fake"),
                changed=True,
            )
        ),
    )

    assert result.repair_loop_result.status == "passed"
    assert result.status == "passed"
    assert result.failed_stage is None
    assert result.error is None


def test_repair_loop_no_progress_still_maps_to_tests_failed() -> None:
    """Test 2：测试真的失败了且修复无产出 → 仍然 tests_failed。"""
    result = _run_with_applier(
        FakeApplier(
            repair.RepairApplicationResult(
                artifacts=generation.GeneratedArtifacts(files=[], summary="fake"),
                changed=False,
            )
        )
    )

    assert result.repair_loop_result.status == "no_progress"
    assert result.status == "tests_failed"
    assert result.failed_stage is None
    assert result.error is None


def test_repair_loop_max_iterations_still_maps_to_tests_failed() -> None:
    """Test 2：修复次数耗尽 → 仍然 tests_failed（未被本次修复波及）。"""
    result = pipeline.run_pipeline(
        PETSTORE_PATH,
        DEMO_PROJECT_PATH,
        test_runner=FakeRunner([_failed(), _failed(), _failed(), _failed()]),
        repair_applier=FakeApplier(
            repair.RepairApplicationResult(
                artifacts=generation.GeneratedArtifacts(files=[], summary="fake"),
                changed=True,
            )
        ),
    )

    assert result.repair_loop_result.status == "max_iterations"
    assert result.status == "tests_failed"
    assert result.failed_stage is None


def test_repair_loop_not_repairable_still_maps_to_tests_failed() -> None:
    """Test 2：计划判定不可修复 → 仍然 tests_failed（未被本次修复波及）。"""
    result = pipeline.run_pipeline(
        PETSTORE_PATH,
        DEMO_PROJECT_PATH,
        test_runner=FakeRunner([_failed()]),
        repair_planner=OnePlanPlanner(_not_repairable_plan()),
    )

    assert result.repair_loop_result.status == "not_repairable"
    assert result.status == "tests_failed"
    assert result.failed_stage is None
