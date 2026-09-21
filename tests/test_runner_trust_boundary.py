"""Test Runner 的信任边界：计数只能来自 runner-owned 插件，绝不来自 stdout。

被测对象是**不可信代码**（生成的源码 + 生成的测试），它和 pytest 跑在同一个进程里：
它能 print 任何东西、能在 conftest 里 ``os._exit``、也能提前往工作区写一份假的统计
文件。所以本文件里每一条"必须判 error / failed"的断言都配了一条**非空洞性**守卫——
先证明欺骗手段真的发生了（伪造行确实在 stdout 里、退出码确实是 0、假文件确实先写
进去了），再证明判定没有被它带走。

§九 的七个场景以 (exit_code, passed, failed, errors, skipped, status) 六元组列在
SCENARIOS 表里，一张表就是报告里那张表。timeout 单独一条（它没有退出码）。

注：常量一律通过 ``test_runner`` 子模块引用——``SESSION_NOT_COMPLETED_*`` 没有从
包根导出，为测试去扩充 `validation.__all__` 属于无谓的 API 变更。
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from integration_agent import generation, validation
from integration_agent.validation import apiforge_runner_plugin
from integration_agent.validation import test_runner as runner

PASSING_TEST = "def test_add():\n    assert 1 + 1 == 2\n"
FAILING_TEST = "def test_failure():\n    assert 1 == 2\n"
SKIPPED_TEST = (
    'import pytest\n\n\n@pytest.mark.skip(reason="later")\ndef test_skipped():\n    assert False\n'
)

# 伪造的摘要行：它和 pytest 真摘要长得一模一样，所以只有"计数不从 stdout 来"才能挡住它。
FAKE_SUMMARY = "5 passed in 0.01s"
# 用 atexit 打印：真摘要是 pytest 在 sessionfinish 里写出的，而 atexit 排在它**之后**
# ——"最后一行才是真的"这个朴素假设就是这么破的。
FAKE_SUMMARY_TEST = (
    "import atexit\n"
    "\n"
    f"atexit.register(lambda: print({FAKE_SUMMARY!r}))\n"
    "\n"
    "\n"
    "def test_failure():\n"
    "    assert 1 == 2\n"
)
# 同样的伪造，但一个测试都没真正执行（唯一的用例被跳过）——退出码 0。
# 这是最危险的一格：旧的 stdout 解析 + 退出码 0 会判 passed。
FAKE_SUMMARY_WITH_SKIP = (
    "import atexit\n"
    "import pytest\n"
    "\n"
    f"atexit.register(lambda: print({FAKE_SUMMARY!r}))\n"
    "\n"
    "\n"
    '@pytest.mark.skip(reason="later")\n'
    "def test_skipped():\n"
    "    assert False\n"
)
OS_EXIT_CONFTEST = "import os\n\nos._exit(0)\n"
SYS_EXIT_CONFTEST = "import sys\n\nsys.exit(0)\n"

# 旧实现（HEAD）的 stdout 计数解析：反向找到第一条"… in 0.05s"再抓数字。
_LEGACY_SUMMARY_KEYS = {
    "passed": "passed",
    "failed": "failed",
    "error": "errors",
    "errors": "errors",
    "skipped": "skipped",
}


def _legacy_stdout_counts(stdout: str) -> dict[str, int]:
    """**已被移除的** stdout 解析的等价实现，只为让欺骗用例非空洞。

    留着它是为了回答"凭什么说 stdout 不可信"：先证明伪造的那行会被旧实现当成真摘要，
    再说新实现不这么干。没有这条，`status == "failed"` 可能只是因为伪造压根没生效。
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for line in reversed(stdout.splitlines()):
        if re.search(r"in\s+[\d.]+s\s*$", line):
            for match in re.finditer(r"(\d+)\s+(\w+)", line):
                key = _LEGACY_SUMMARY_KEYS.get(match.group(2))
                if key:
                    counts[key] = int(match.group(1))
            break
    return counts


def _artifacts(files: dict[str, str]) -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(path=path, action="create", content=content, purpose="测试")
            for path, content in files.items()
        ],
        dependency_changes=[],
        summary="合成产物（信任边界测试用）",
    )


def _core(result: validation.TestResult) -> dict:
    """确定性比较用：与运行环境无关的六个核心字段。"""
    return {
        "status": result.status,
        "exit_code": result.exit_code,
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "skipped": result.skipped,
        "failure_details": result.failure_details,
    }


def _summary_like_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if re.search(r"in\s+[\d.]+s\s*$", line)]


def _norm(path: str) -> str:
    """路径比较用：Windows 上大小写与分隔符都可能不同。"""
    return os.path.normcase(os.path.abspath(path))


# ------------------------------------------------------------ §九：七个场景


SCENARIOS: dict[str, tuple[dict[str, str], tuple]] = {
    # 场景: (工作区文件, (exit_code, passed, failed, errors, skipped, status))
    "1_normal_pass": ({"tests/test_ok.py": PASSING_TEST}, (0, 1, 0, 0, 0, "passed")),
    "2_normal_failure": ({"tests/test_bad.py": FAILING_TEST}, (1, 0, 1, 0, 0, "failed")),
    "3_all_skipped": ({"tests/test_skip.py": SKIPPED_TEST}, (0, 0, 0, 0, 1, "error")),
    "4_conftest_os_exit": (
        {"conftest.py": OS_EXIT_CONFTEST, "tests/test_ok.py": PASSING_TEST},
        (0, 0, 0, 0, 0, "error"),
    ),
    "5_conftest_sys_exit": (
        {"conftest.py": SYS_EXIT_CONFTEST, "tests/test_ok.py": PASSING_TEST},
        (0, 0, 0, 0, 0, "error"),
    ),
    "6a_fake_summary_with_real_failure": (
        {"tests/test_noise.py": FAKE_SUMMARY_TEST},
        (1, 0, 1, 0, 0, "failed"),
    ),
    "6b_fake_summary_without_executed_tests": (
        {"tests/test_noise.py": FAKE_SUMMARY_WITH_SKIP},
        (0, 0, 0, 0, 1, "error"),
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_table(name: str) -> None:
    """§九：一行一个场景，六元组逐项断言（这一张表就是报告里那张表）。"""
    files, expected = SCENARIOS[name]

    result = validation.run_tests(_artifacts(files))

    actual = (
        result.exit_code,
        result.passed,
        result.failed,
        result.errors,
        result.skipped,
        result.status,
    )
    assert actual == expected, f"{name}: {actual} != {expected}\n{result.stdout}"


def test_all_skipped_run_says_nothing_was_executed() -> None:
    """§九 Case3：全跳过时"退出码 0"不等于通过，而且要说清是"没执行"。"""
    result = validation.run_tests(_artifacts({"tests/test_skip.py": SKIPPED_TEST}))

    assert result.status == "error"
    assert (result.passed, result.failed, result.errors, result.skipped) == (0, 0, 0, 1)
    assert result.failure_details, "没有失败详情，下游无从判断这次运行什么都没验证"
    messages = " ".join(detail.message for detail in result.failure_details)
    assert runner.NO_TESTS_EXECUTED_MESSAGE in messages
    # 跳过不算执行：说明里必须点出被跳过的那 1 个（否则"0 passed"看起来像环境问题）
    assert "1 个测试被跳过" in messages


@pytest.mark.parametrize(
    ("conftest", "expected_exit"),
    [
        (OS_EXIT_CONFTEST, 0),
        (SYS_EXIT_CONFTEST, 0),
    ],
    ids=["os_exit", "sys_exit"],
)
def test_conftest_cannot_fake_a_completed_session(conftest: str, expected_exit: int) -> None:
    """§九 Case4/5：conftest 掐断进程时，退出码 0 也不代表"跑完了"。

    非空洞性：退出码**确实**是 0、且 stdout 里没有任何真摘要——退一步看就是"成功"。
    判 error 的唯一理由是：那份统计文件没有被写出来。
    """
    result = validation.run_tests(
        _artifacts({"conftest.py": conftest, "tests/test_ok.py": PASSING_TEST})
    )

    assert result.exit_code == expected_exit
    assert _summary_like_lines(result.stdout) == [], "pytest 竟然打出了摘要，本用例的前提不成立"
    assert _legacy_stdout_counts(result.stdout) == {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert result.status == "error"
    assert (result.passed, result.failed, result.errors, result.skipped) == (0, 0, 0, 0)
    assert result.failure_details[0].test_name == runner.SESSION_NOT_COMPLETED_NAME
    assert runner.SESSION_NOT_COMPLETED_MESSAGE in result.stderr


@pytest.mark.parametrize(
    "test_source", [FAKE_SUMMARY_TEST, FAKE_SUMMARY_WITH_SKIP], ids=["failure", "skipped"]
)
def test_spoofed_summary_really_reaches_stdout(test_source: str) -> None:
    """非空洞性守卫：伪造行必须真的成为 stdout 的**最后**一行摘要形态。

    它同时钉住 `_legacy_stdout_counts` 的前提——旧实现正是把最后一行当真摘要，
    所以它读到的是伪造的 ``5 passed``。
    """
    result = validation.run_tests(_artifacts({"tests/test_noise.py": test_source}))

    lines = _summary_like_lines(result.stdout)
    assert lines, f"stdout 里没有摘要形态的行，欺骗用例是空的：\n{result.stdout}"
    assert lines[-1].strip() == FAKE_SUMMARY, f"伪造行不是最后一行：{lines[-3:]}"
    assert _legacy_stdout_counts(result.stdout)["passed"] == 5, "旧实现没有被骗到，本用例没有意义"


def test_fake_summary_cannot_override_a_real_failure() -> None:
    """§九 Case6：真实失败 + 伪造 ``5 passed`` → 仍然是 failed，passed 必须是 0。"""
    result = validation.run_tests(_artifacts({"tests/test_noise.py": FAKE_SUMMARY_TEST}))

    assert result.status == "failed"
    assert (result.passed, result.failed) == (0, 1)
    assert result.exit_code == 1


def test_fake_summary_cannot_turn_a_skipped_run_into_passed() -> None:
    """§九 Case6 的致命变体：一个测试都没执行，但 stdout 宣称 5 passed。"""
    result = validation.run_tests(_artifacts({"tests/test_noise.py": FAKE_SUMMARY_WITH_SKIP}))

    assert result.exit_code == 0
    # 旧实现：退出码 0 → passed；stdout 说 5 passed → passed=5。这就是假阳性。
    assert result.status == "error"
    assert (result.passed, result.failed, result.errors, result.skipped) == (0, 0, 0, 1)


def test_failure_details_survive_the_plugin_path() -> None:
    """§十四：引入插件后，既有失败详情（名字/文件/行号/消息/traceback）不能丢。"""
    result = validation.run_tests(_artifacts({"tests/test_bad.py": FAILING_TEST}))

    assert result.status == "failed"
    (detail,) = result.failure_details
    assert detail.test_name == "test_failure"
    assert detail.file == "tests/test_bad.py"
    assert detail.line == 2
    assert detail.message and "assert 1 == 2" in detail.message
    assert detail.traceback and "assert 1 == 2" in detail.traceback


def test_timeout_is_not_downgraded_by_the_plugin_path() -> None:
    """§十五：超时保持 ``timeout``——既不是通过，也不被改判成 failed/error。"""
    sleeping = "import time\n\n\ndef test_slow():\n    time.sleep(30)\n"

    result = validation.DeterministicTestRunner(timeout=0.5).run(
        _artifacts({"tests/test_slow.py": sleeping})
    )

    assert result.status == "timeout"
    assert result.exit_code is None
    assert (result.passed, result.failed, result.errors, result.skipped) == (0, 0, 0, 0)
    assert "超时" in result.stderr


# ------------------------------------------- §十～§十三：统计文件归 runner 所有


def test_generated_conftest_cannot_prewrite_the_stats_file() -> None:
    """生成代码提前写一份**看起来完全合法**的统计，必须被插件覆盖。

    假文件的 ``exit_status`` 故意写成真实的退出码 1：这样它不会被一致性检查挡掉，
    能被挡住就只剩一种可能——session 正常结束时插件把它重写了。这一格如果失守，
    生成代码就能自己宣布通过。
    """
    spoof = json.dumps(
        {
            "passed": 99,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "collected": 1,
            "exit_status": 1,
            "failures": [],
        }
    )
    # 收集期就把假文件写下去，并**当场读回**：写不进去 / 读不回来就让 conftest 直接炸
    # （收集期错误 → 状态不可能是 failed），免得这条用例在"压根没写成功"时空洞通过。
    conftest = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        f"TARGET = os.environ.get({runner.RUNNER_STATS_ENV_VAR!r})\n"
        f"SPOOF = {spoof!r}\n"
        "assert TARGET, 'runner 没有设置统计文件路径，本用例失去意义'\n"
        "Path(TARGET).write_text(SPOOF, encoding='utf-8')\n"
        "assert Path(TARGET).read_text(encoding='utf-8') == SPOOF\n"
    )

    result = validation.run_tests(
        _artifacts({"conftest.py": conftest, "tests/test_bad.py": FAILING_TEST})
    )

    assert (result.passed, result.failed) == (0, 1)
    assert result.status == "failed"


def test_stats_file_is_runner_owned_and_lives_only_in_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§十七 + §十：统计文件由 runner 定路径、落在工作区内、随工作区一起消失。

    顺便钉住两件事：插件是**从工作区那份副本**加载的（不是宿主安装的包），以及
    父进程里同名环境变量的值不会被子进程沿用。
    """
    workspace_parent = tmp_path / "ws"
    workspace_parent.mkdir()
    observed = tmp_path / "observed.json"
    parent_target = tmp_path / "parent-stats.json"
    monkeypatch.setenv(runner.RUNNER_STATS_ENV_VAR, str(parent_target))

    conftest = (
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "import apiforge_runner_plugin\n"
        "\n"
        f"RUNNER_ENV = {runner.RUNNER_STATS_ENV_VAR!r}\n"
        f"OBSERVED = Path({str(observed)!r})\n"
        "\n"
        "OBSERVED.write_text(\n"
        "    json.dumps(\n"
        "        {\n"
        "            'stats_path': os.environ.get(RUNNER_ENV),\n"
        "            'cwd': os.getcwd(),\n"
        "            'plugin_file': apiforge_runner_plugin.__file__,\n"
        "        }\n"
        "    ),\n"
        "    encoding='utf-8',\n"
        ")\n"
    )

    result = validation.DeterministicTestRunner(workspace_parent=workspace_parent).run(
        _artifacts({"conftest.py": conftest, "tests/test_ok.py": PASSING_TEST})
    )

    assert result.status == "passed"
    data = json.loads(observed.read_text(encoding="utf-8"))
    workspace, stats_path = data["cwd"], data["stats_path"]

    # 统计文件：路径由 runner 决定（父进程那份同名变量没有被沿用），且在工作区内。
    assert stats_path and _norm(stats_path) != _norm(str(parent_target))
    assert not parent_target.exists(), "子进程用了父进程给的路径"
    assert _norm(os.path.dirname(stats_path)) == _norm(workspace)
    assert os.path.basename(stats_path) == runner.RUNNER_STATS_FILENAME
    # 插件：真的从工作区那份副本加载（`-p` 在子进程的 sys.path 上找到的就是它）。
    assert _norm(os.path.dirname(data["plugin_file"])) == _norm(workspace)
    assert os.path.basename(data["plugin_file"]) == runner.RUNNER_PLUGIN_SOURCE
    # 运行结束后：统计文件与整个工作区都不在了。
    assert not Path(stats_path).exists()
    assert not Path(workspace).exists()
    assert list(workspace_parent.iterdir()) == []


# ---------------------------------------------------------------- 常量契约


def test_stats_env_var_matches_the_plugin() -> None:
    """常量是**复制**过来的（TestRunner 不 import 插件模块），一致性只能靠这条钉住。"""
    assert runner.RUNNER_STATS_ENV_VAR == apiforge_runner_plugin.STATS_PATH_ENV_VAR


def test_plugin_module_name_and_source_file_agree() -> None:
    """``-p <module>`` 按模块名加载，文件名必须与模块名一致，否则插件永远加载不上。"""
    source = Path(runner.RUNNER_PLUGIN_SOURCE)

    assert source.stem == runner.RUNNER_PLUGIN_MODULE
    assert Path(runner.__file__ or "").with_name(runner.RUNNER_PLUGIN_SOURCE).is_file()


def test_stats_filename_cannot_escape_the_workspace() -> None:
    name = runner.RUNNER_STATS_FILENAME

    assert name not in ("", ".", "..")
    assert "/" not in name and "\\" not in name


def test_validation_package_does_not_require_pytest() -> None:
    """pytest 是 dev 依赖：生产 import 不许因为缺 pytest 而失败。

    这是"常量复制而不是 import 插件模块"这个决定的存在理由；哪天有人想把它改成
    ``from ... import apiforge_runner_plugin``，这条会先亮红。
    """
    code = (
        "import builtins\n"
        "real = builtins.__import__\n"
        "def blocked(name, *args, **kwargs):\n"
        "    if name == 'pytest' or name.startswith('pytest.'):\n"
        "        raise ImportError('pytest is a dev dependency')\n"
        "    return real(name, *args, **kwargs)\n"
        "builtins.__import__ = blocked\n"
        "from integration_agent.validation import test_runner\n"
        "print(test_runner.RUNNER_PLUGIN_MODULE)\n"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == runner.RUNNER_PLUGIN_MODULE


# ------------------------------------------------------------------ §十六


def test_deterministic_core_fields_with_stdout_noise() -> None:
    """同一份产物连跑两次：六个核心字段完全一致（含被伪造 stdout 的场景）。"""
    artifacts = _artifacts(
        {"tests/test_noise.py": FAKE_SUMMARY_TEST, "tests/test_ok.py": PASSING_TEST}
    )

    first = validation.run_tests(artifacts)
    second = validation.run_tests(artifacts)

    assert first.status == "failed"
    assert _core(first) == _core(second)
