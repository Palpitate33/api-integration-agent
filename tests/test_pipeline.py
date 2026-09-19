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
