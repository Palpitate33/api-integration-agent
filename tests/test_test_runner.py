"""Test Runner 单元测试。

覆盖要求的 10 个场景：
    1. 全部通过            6. generated source + generated test
    2. 测试失败            7. timeout（subprocess 被终止）
    3. 测试错误            8. stdout / stderr 捕获
    4. skipped 统计        9. Repository 不变（快照证明）
    5. 多测试文件          10. Deterministic（核心字段一致）

以及：Protocol 一致性、Pydantic 契约、工作区清理、依赖离线处理、
modify 片段不落地、路径越界防护、无测试可收集等边界。

注：统一通过 validation. / generation. 前缀引用；TestResult / TestRunner
若直接 import 进测试模块命名空间，会被 pytest 误当作测试类收集。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from integration_agent import generation, validation

PASSING_TEST = "def test_add():\n    assert 1 + 1 == 2\n"


def _artifacts(
    files: dict[str, str],
    *,
    dependencies: list[generation.DependencyChange] | None = None,
    extra: list[generation.GeneratedFile] | None = None,
) -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(path=path, action="create", content=content, purpose="测试")
            for path, content in files.items()
        ]
        + (extra or []),
        dependency_changes=dependencies or [],
        summary="合成产物（单元测试用）",
    )


def _core(result: validation.TestResult) -> dict:
    """确定性比较用：只取与运行环境无关的核心字段。"""
    return {
        "status": result.status,
        "exit_code": result.exit_code,
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "skipped": result.skipped,
        "failure_details": result.failure_details,
        "dependency_warnings": result.dependency_warnings,
    }


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ------------------------------------------------------- 场景 1：全部通过


def test_all_passed() -> None:
    result = validation.run_tests(_artifacts({"tests/test_add.py": PASSING_TEST}))
    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.passed == 1
    assert result.failed == 0
    assert result.errors == 0
    assert result.skipped == 0
    assert result.failure_details == []


# ---------------------------------------------------------- 场景 2：测试失败


def test_failure_reported_with_details() -> None:
    content = "def test_failure():\n    assert 1 == 2\n"
    result = validation.run_tests(_artifacts({"tests/test_fail.py": content}))

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.failed == 1
    assert result.failure_details

    detail = result.failure_details[0]
    assert detail.test_name == "test_failure"
    assert detail.file == "tests/test_fail.py"
    assert detail.line == 2
    assert "assert 1 == 2" in detail.message
    assert detail.traceback


# ---------------------------------------------------------- 场景 3：测试错误


def test_error_enters_failure_details() -> None:
    content = 'def test_error():\n    raise RuntimeError("boom")\n'
    result = validation.run_tests(_artifacts({"tests/test_err.py": content}))

    # pytest 8+ 把测试体内未处理的异常计为 failed；错误信息必须进入 failure_details
    assert result.status == "failed"
    assert result.failed == 1
    assert result.failure_details
    detail = result.failure_details[0]
    assert detail.test_name == "test_error"
    assert "RuntimeError" in detail.message


def test_fixture_error_counts_as_error() -> None:
    """fixture 异常是 pytest 真正的 "error" 类别，errors 计数与详情都要解析。"""
    content = (
        "import pytest\n\n\n"
        "@pytest.fixture\n"
        "def broken():\n"
        '    raise RuntimeError("fixture boom")\n\n\n'
        "def test_uses(broken):\n"
        "    assert True\n"
    )
    result = validation.run_tests(_artifacts({"tests/test_fix.py": content}))

    assert result.status == "failed"
    assert result.errors == 1
    assert result.failed == 0
    assert result.failure_details
    detail = result.failure_details[0]
    assert detail.test_name == "test_uses"
    assert "fixture boom" in detail.message


# ---------------------------------------------------------- 场景 4：skipped


def test_skipped_counted() -> None:
    content = (
        "import pytest\n\n\n"
        '@pytest.mark.skip(reason="not now")\n'
        "def test_skip():\n"
        "    assert False\n\n\n"
        "def test_ok():\n"
        "    assert True\n"
    )
    result = validation.run_tests(_artifacts({"tests/test_skip.py": content}))

    assert result.status == "passed"
    assert result.passed == 1
    assert result.skipped == 1


# ------------------------------------------------------ 场景 5：多测试文件


def test_multiple_test_files() -> None:
    result = validation.run_tests(
        _artifacts(
            {
                "tests/test_a.py": PASSING_TEST,
                "tests/test_b.py": "def test_sub():\n    assert 2 - 1 == 1\n",
            }
        )
    )
    assert result.status == "passed"
    assert result.passed == 2


# ------------------------------------------- 场景 6：generated source + test


def test_generated_source_importable() -> None:
    result = validation.run_tests(
        _artifacts(
            {
                "src/demo.py": "def add(a, b):\n    return a + b\n",
                "tests/test_demo.py": (
                    "from src.demo import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
                ),
            }
        )
    )
    assert result.status == "passed"
    assert result.passed == 1


# ------------------------------------------------------------ 场景 7：timeout


def test_timeout_terminates_subprocess(tmp_path: Path) -> None:
    workspace_parent = tmp_path / "ws"
    workspace_parent.mkdir()
    runner = validation.DeterministicTestRunner(timeout=3.0, workspace_parent=workspace_parent)
    content = "import time\n\n\ndef test_slow():\n    time.sleep(30)\n"
    result = runner.run(_artifacts({"tests/test_slow.py": content}))

    assert result.status == "timeout"
    assert result.exit_code is None
    assert result.passed == 0
    assert "超时" in result.stderr
    assert result.duration < 10
    assert list(workspace_parent.iterdir()) == []  # 超时后工作区同样被清理


# ----------------------------------------------------- 场景 8：stdout / stderr


def test_stdout_stderr_captured() -> None:
    content = (
        "import sys\n\n\n"
        "def test_capture():\n"
        '    print("hello-stdout")\n'
        '    sys.stderr.write("boom-stderr\\n")\n'
        "    assert 1 == 2\n"
    )
    # -s 关闭 pytest 捕获，让 print / stderr 直通子进程的 stdout / stderr 流
    runner = validation.DeterministicTestRunner(pytest_args=["-q", "-ra", "--tb=short", "-s"])
    result = runner.run(_artifacts({"tests/test_capture.py": content}))

    assert result.status == "failed"
    assert "hello-stdout" in result.stdout
    assert "boom-stderr" in result.stderr


# ------------------------------------------------ 场景 9：Repository 不变


def test_repository_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app").mkdir()
    (root / "app" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "repo"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )
    before = _snapshot(root)

    validation.run_tests(
        _artifacts(
            {"tests/test_ok.py": PASSING_TEST},
            dependencies=[
                generation.DependencyChange(name="apiforge_missing_dep_xyz", action="add")
            ],
        )
    )

    assert _snapshot(root) == before


# ------------------------------------------------------- 场景 10：Deterministic


def test_deterministic_core_fields() -> None:
    artifacts = _artifacts(
        {
            "tests/test_pass.py": PASSING_TEST,
            "tests/test_fail.py": "def test_failure():\n    assert 1 == 2\n",
            "tests/test_skip.py": (
                "import pytest\n\n\n"
                '@pytest.mark.skip(reason="later")\n'
                "def test_skip():\n"
                "    assert False\n"
            ),
        }
    )
    first = validation.run_tests(artifacts)
    second = validation.run_tests(artifacts)

    assert _core(first) == _core(second)
    assert first.status == "failed"


# ------------------------------------------------------------- 附加与边界


def test_protocol_conformance() -> None:
    assert isinstance(validation.DeterministicTestRunner(), validation.TestRunner)


def test_result_contract_roundtrip() -> None:
    result = validation.run_tests(_artifacts({"tests/test_ok.py": PASSING_TEST}))
    assert isinstance(result, validation.TestResult)
    assert isinstance(result.failure_details, list)
    assert validation.TestResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        validation.TestResult(status="unknown")  # type: ignore[arg-type]


def test_workspace_cleaned_after_run(tmp_path: Path) -> None:
    workspace_parent = tmp_path / "ws"
    workspace_parent.mkdir()
    runner = validation.DeterministicTestRunner(workspace_parent=workspace_parent)
    result = runner.run(_artifacts({"tests/test_ok.py": PASSING_TEST}))
    assert result.status == "passed"
    assert list(workspace_parent.iterdir()) == []


def test_missing_dependency_reported_offline() -> None:
    result = validation.run_tests(
        _artifacts(
            {"tests/test_ok.py": PASSING_TEST},
            dependencies=[
                generation.DependencyChange(
                    name="apiforge_missing_dep_xyz", version=">=1.0", action="add"
                )
            ],
        )
    )
    # 离线模式不安装：测试本身不依赖它所以通过，但必须如实报告
    assert result.status == "passed"
    assert any("apiforge_missing_dep_xyz" in item for item in result.dependency_warnings)


def test_installed_dependency_no_warning() -> None:
    result = validation.run_tests(
        _artifacts(
            {"tests/test_ok.py": PASSING_TEST},
            dependencies=[
                generation.DependencyChange(name="pydantic", version=">=2.0", action="add")
            ],
        )
    )
    assert result.status == "passed"
    assert result.dependency_warnings == []


def test_no_tests_collected_is_error() -> None:
    result = validation.run_tests(_artifacts({"src/mod.py": "x = 1\n"}))
    assert result.status == "error"
    assert result.exit_code == 5


def test_modify_files_are_not_materialized(tmp_path: Path) -> None:
    workspace_parent = tmp_path / "ws"
    workspace_parent.mkdir()
    runner = validation.DeterministicTestRunner(workspace_parent=workspace_parent)
    result = runner.run(
        _artifacts(
            {"tests/test_ok.py": PASSING_TEST},
            extra=[
                generation.GeneratedFile(
                    path="app/http.py",
                    action="modify",
                    content="from app.sample_client import SampleClient\n",
                    purpose="复用既有模块",
                )
            ],
        )
    )
    assert result.status == "passed"
    assert list(workspace_parent.iterdir()) == []  # modify 片段不落地


def test_path_traversal_rejected() -> None:
    result = validation.run_tests(_artifacts({"../evil.py": "x = 1\n"}))
    assert result.status == "error"
    assert "越界" in result.stderr


def test_failure_detail_model_typed() -> None:
    result = validation.run_tests(
        _artifacts({"tests/test_fail.py": "def test_f():\n    assert 0\n"})
    )
    assert result.failure_details
    assert all(isinstance(item, validation.FailureDetail) for item in result.failure_details)
