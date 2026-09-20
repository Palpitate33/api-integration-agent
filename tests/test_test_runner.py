"""Test Runner 单元测试。

覆盖要求的 10 个场景：
    1. 全部通过            6. generated source + generated test
    2. 测试失败            7. timeout（subprocess 被终止）
    3. 测试错误            8. stdout / stderr 捕获
    4. skipped 统计        9. Repository 不变（快照证明）
    5. 多测试文件          10. Deterministic（核心字段一致）

以及：Protocol 一致性、Pydantic 契约、工作区清理、依赖离线处理、
modify 片段不落地、路径越界防护、无测试可收集等边界。

再加上 subprocess 环境隔离（P1-4）：父进程环境默认不进入 pytest 子进程，
宿主凭证无法被生成代码通过 os.environ 读到。凭据相关断言一律走
**真实 TestRunner → subprocess → pytest → 生成测试** 这条链路，
而不是只检查 build_subprocess_env() 的返回值。

注：统一通过 validation. / generation. 前缀引用；TestResult / TestRunner
若直接 import 进测试模块命名空间，会被 pytest 误当作测试类收集。
本模块只使用自造的 sentinel 值，既不读取也不输出宿主的真实凭证。
"""

import os
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


# -------------------------------------------- P1-4：subprocess 环境隔离
#
# 生成产物与生成测试都是不可信代码，它们在 pytest 子进程里可以直接读 os.environ。
# 若子进程整体继承父进程环境（env = os.environ.copy()），宿主凭证就会暴露给它们。
#
# 下面用**自造 sentinel** 证明隔离生效：断言只涉及"sentinel 取值 / 变量名是否出现"，
# 全程不读取、不写入、不打印宿主的任何真实凭证。
# 生产代码里没有这些值——它们只存在于本测试模块。

SECRET_SENTINELS = {
    "DEEPSEEK_API_KEY": "test-secret-sentinel",
    "OPENAI_API_KEY": "openai-secret-sentinel",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-sentinel",
    "DATABASE_URL": "database-secret-sentinel",
}

# 子进程环境 = allowlist + 本模块显式设定的两项（不继承父进程）
_EXPLICIT_ENV = {"PYTHONPATH", "PYTHONIOENCODING"}


def _spy_test(body: str) -> str:
    """生成一个在子进程里运行的"间谍测试"文件，供 TestRunner 真实执行。"""
    return "import os\n\n\n" + body


def _leak_probe() -> str:
    """检查父进程 sentinel 是否泄漏给子进程。

    断言信息只列变量名、不含取值——万一将来隔离被破坏，测试报告也不会
    把凭证内容抄进日志 / TestResult。
    """
    return _spy_test(
        f"SENTINELS = {SECRET_SENTINELS!r}\n"
        "\n"
        "\n"
        "def test_secret_values_not_visible():\n"
        "    leaked = [n for n, v in SENTINELS.items() if os.environ.get(n) == v]\n"
        "    assert leaked == [], leaked\n"
        "\n"
        "\n"
        "def test_secret_names_absent():\n"
        "    leaked = [n for n in SENTINELS if n in os.environ]\n"
        "    assert leaked == [], leaked\n"
    )


# ------------------------------------- §十五 1-4：四个凭证变量逐个验证


@pytest.mark.parametrize("name", sorted(SECRET_SENTINELS))
def test_parent_secret_not_visible_in_subprocess(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """父进程设 sentinel → 真实跑 TestRunner → 生成测试读 os.environ → 必须 passed。

    每个参数只设置一个变量，构成单变量对照：泄漏与否直接归因到该变量。
    """
    monkeypatch.setenv(name, SECRET_SENTINELS[name])

    result = validation.run_tests(_artifacts({"tests/test_leak_probe.py": _leak_probe()}))

    assert result.status == "passed"
    assert result.passed == 2  # 两个探针都真的跑到了，不是"没收集到测试"
    assert result.failed == 0


# ------------------------ §十五 6：必要运行环境仍然存在（以实际行为为准）


def test_required_runtime_env_reaches_child() -> None:
    """pytest 自身启动所需的环境变量必须照常传给子进程。

    期望集合由**当前 allowlist ∩ 当前父进程环境**推导，因此不写死平台差异：
    Windows 需要 SystemRoot，Linux 上则没有它，两者都应通过。
    """
    present = {name.upper() for name in os.environ}
    expected = sorted(
        n.upper() for n in validation.SUBPROCESS_ENV_ALLOWLIST if n.upper() in present
    )
    assert "PATH" in expected  # 任何平台都必须有可执行文件搜索路径

    probe = _spy_test(
        f"EXPECTED = {expected!r}\n"
        "\n"
        "\n"
        "def test_allowlisted_env_visible():\n"
        "    present = {name.upper() for name in os.environ}\n"
        "    missing = [n for n in EXPECTED if n not in present]\n"
        "    assert missing == [], missing\n"
    )
    result = validation.run_tests(_artifacts({"tests/test_runtime_env.py": probe}))

    assert result.status == "passed"
    assert result.passed == 1


def test_parent_env_is_not_inherited_wholesale(monkeypatch: pytest.MonkeyPatch) -> None:
    """结构性不变量：父进程的环境变量不会进入子进程——**不管它叫什么名字**。

    这是 allowlist 相对黑名单的核心价值：断言不依赖任何具体凭证名，
    因此将来新增的凭证默认也是安全的。

    不断言"子进程环境里只有 allowlist"：pytest 自己会在子进程内设置
    PYTEST_CURRENT_TEST / PYTEST_VERSION 之类的变量，那不是继承来的。
    这里断言的是"继承"这件事本身没有发生。

    名字比较按大小写归一：Windows 会把变量名归一成大写（SystemRoot → SYSTEMROOT），
    而 Windows 的变量名查询本来就大小写不敏感，归一后比较才是等价的。
    """
    marker = "apiforge-parent-only-value"
    planted = {f"APIFORGE_PARENT_VAR_{index}": marker for index in range(3)}
    planted.update(SECRET_SENTINELS)
    planted["SOME_FUTURE_CREDENTIAL"] = marker
    for name, value in planted.items():
        monkeypatch.setenv(name, value)

    probe = _spy_test(
        f"MARKER = {marker!r}\n"
        f"NAMES = {sorted(planted)!r}\n"
        "\n"
        "\n"
        "def test_parent_names_absent():\n"
        "    present = {n.upper() for n in os.environ}\n"
        "    leaked = [n for n in NAMES if n.upper() in present]\n"
        "    assert leaked == [], leaked\n"
        "\n"
        "\n"
        "def test_no_parent_value_anywhere():\n"
        "    tainted = sum(1 for v in os.environ.values() if MARKER in v)\n"
        "    assert tainted == 0, f'父进程的值出现在子进程环境里：{tainted} 处'\n"
    )
    result = validation.run_tests(_artifacts({"tests/test_env_inherit.py": probe}))

    assert result.status == "passed"
    assert result.passed == 2


# ------------------------------------------- §十五 5：正常 pytest 语义完整


def test_normal_pytest_semantics_under_sanitized_env() -> None:
    """隔离后普通 pytest 照常工作：通过 / 失败 / skip 三类仍分别统计与定位。"""
    result = validation.run_tests(
        _artifacts(
            {
                "tests/test_ok.py": PASSING_TEST,
                "tests/test_fail.py": "def test_failure():\n    assert 1 == 2\n",
                "tests/test_skip.py": (
                    "import pytest\n\n\n"
                    '@pytest.mark.skip(reason="later")\n'
                    "def test_skip():\n"
                    "    assert False\n"
                ),
            }
        )
    )

    assert result.status == "failed"
    assert result.exit_code == 1
    assert (result.passed, result.failed, result.skipped) == (1, 1, 1)
    assert [detail.test_name for detail in result.failure_details] == ["test_failure"]
    assert result.failure_details[0].file == "tests/test_fail.py"


# ------------------------------------------------ §十五 7：Deterministic


def test_env_isolation_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一份探针连跑两次，核心结果一致——隔离是稳定生效的，不是偶发。"""
    for name, sentinel in SECRET_SENTINELS.items():
        monkeypatch.setenv(name, sentinel)
    artifacts = _artifacts({"tests/test_leak_probe.py": _leak_probe()})

    first = validation.run_tests(artifacts)
    second = validation.run_tests(artifacts)

    assert _core(first) == _core(second)
    assert first.status == "passed"


# ------------------------------------- build_subprocess_env 的单元级契约


def test_build_subprocess_env_keeps_only_allowlist(tmp_path: Path) -> None:
    base = {name: "keep" for name in validation.SUBPROCESS_ENV_ALLOWLIST}
    base.update(
        {
            **SECRET_SENTINELS,
            "GITHUB_TOKEN": "x",
            "SOME_FUTURE_CREDENTIAL": "x",
        }
    )

    env = validation.build_subprocess_env(tmp_path, base=base)

    assert set(env) == set(validation.SUBPROCESS_ENV_ALLOWLIST) | _EXPLICIT_ENV
    assert env["PYTHONPATH"] == str(tmp_path)
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_build_subprocess_env_overrides_parent_python_settings(tmp_path: Path) -> None:
    """父进程的 PYTHONPATH / PYTHONIOENCODING 不被继承（否则可绕过工作区隔离）。"""
    base = {"PATH": "/usr/bin", "PYTHONPATH": "/host/site-packages", "PYTHONIOENCODING": "cp1252"}

    env = validation.build_subprocess_env(tmp_path, base=base)

    assert env["PYTHONPATH"] == str(tmp_path)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PATH"] == "/usr/bin"


def test_build_subprocess_env_tolerates_missing_vars(tmp_path: Path) -> None:
    """父进程缺少（或平台没有）某个 allowlist 变量时不得报错。"""
    env = validation.build_subprocess_env(tmp_path, base={})

    assert set(env) == _EXPLICIT_ENV


def test_allowlist_contains_no_credential_shaped_names() -> None:
    """结构性守卫：allowlist 里不允许出现"名字就像凭证"的变量。"""
    markers = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "API")

    offenders = [
        name
        for name in validation.SUBPROCESS_ENV_ALLOWLIST
        if any(marker in name.upper() for marker in markers)
    ]

    assert offenders == []
