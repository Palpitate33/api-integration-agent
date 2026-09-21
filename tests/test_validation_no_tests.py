"""P0-2 回归：退出码 0 但一个测试都没执行，绝不能算 "passed"。

根因
----
只看退出码是不够的：pytest 在**一个测试都没真正执行**时也会返回 0
（收集结果被 conftest 清空、全部用例被 skip、`--collect-only` 之类）。
原来的判据 `exit_code == 0 → "passed"` 于是给出假阳性，而 Repair Loop 的第一条
终止条件正是 `test_result.status == "passed"` —— 整个 pipeline 会交付一份
从未被验证过的代码，并且不会有人发现。

对策
----
"passed" 要求三件事同时成立：退出码为 0、**至少有测试通过**（passed > 0）、
且没有 failed / errors。skipped 不计入执行：全部跳过的运行同样什么都没验证。
不满足时归类为 "error"（不是 "failed"——这不是"代码有问题"，是"这次运行没有证据"），
并在 failure_details 里留下 NO_TESTS_EXECUTED_MESSAGE 的可追溯说明。

测试策略
--------
判据必须在**真实 pytest 输出**上成立，所以这里不直接调用 `_status_from_result` 了事，
而是覆盖多条真实的"退出码 0 / 零执行"路径：

    conftest 把全部 item 标记为 skip   → exit 0，"N skipped"
    pytest.ini 加 --collect-only       → exit 0，"N tests collected"，一条都没跑
    conftest 把 items 清空             → exit 5，"no tests ran"（走另一条分支）

再加一条反向用例（真的跑通了仍然必须 "passed"），确认修复没有矫枉过正成"永远 error"。
最后验证 Repair Loop 不会因为"0 个测试"被短路成成功。
"""

import textwrap
from pathlib import Path

import pytest

from integration_agent import generation, validation
from integration_agent.repair import run_repair_loop
from integration_agent.validation.test_runner import _status_from_result

PASSING_TEST = "def test_add():\n    assert 1 + 1 == 2\n"
FAILING_TEST = "def test_should_fail():\n    assert False\n"

# 收集到了、但一条都不执行：pytest 退出码 0
ALL_SKIP_CONFTEST = (
    "import pytest\n"
    "\n"
    "\n"
    "def pytest_collection_modifyitems(items):\n"
    "    for item in items:\n"
    "        item.add_marker(pytest.mark.skip(reason='collected but never executed'))\n"
)

# 让 pytest 只收集不执行：退出码 0，输出里只有 "N tests collected"
COLLECT_ONLY_INI = "[pytest]\naddopts = --collect-only\n"

# 把收集结果清空：pytest 退出码 5（"no tests ran"）
DESELECT_ALL_CONFTEST = "def pytest_collection_modifyitems(config, items):\n    items[:] = []\n"


def _artifacts(files: dict[str, str]) -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=[
            generation.GeneratedFile(path=path, action="create", content=content, purpose="测试")
            for path, content in files.items()
        ],
        summary="合成产物（P0-2 回归用）",
    )


def _zero_test_artifacts() -> generation.GeneratedArtifacts:
    """收集到测试、但一条都不会执行——而且里面**有一个本该失败的测试**。

    那个本该失败的测试是关键：如果它真的跑了，退出码就会是 1。退出码是 0，
    恰恰证明它没跑。所以"退出码 0"在这种情况下不是好消息，是"什么都没验证"。
    """
    return _artifacts(
        {
            "conftest.py": ALL_SKIP_CONFTEST,
            "tests/test_would_fail.py": FAILING_TEST,
        }
    )


# ====================================================== 判据本身（真值表）


@pytest.mark.parametrize(
    ("exit_code", "counts", "expected"),
    [
        # 真的跑通了才算 passed：至少要有一个测试通过
        (0, {"passed": 1, "failed": 0, "errors": 0, "skipped": 0}, "passed"),
        (0, {"passed": 3, "failed": 0, "errors": 0, "skipped": 2}, "passed"),
        # 退出码 0 但零执行 → error
        (0, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}, "error"),
        (0, {"passed": 0, "failed": 0, "errors": 0, "skipped": 5}, "error"),
        # 退出码 0 不可能同时有失败/错误，真出现了同样不能算 passed
        (0, {"passed": 2, "failed": 1, "errors": 0, "skipped": 0}, "error"),
        (0, {"passed": 2, "failed": 0, "errors": 1, "skipped": 0}, "error"),
        # 退出码 1：确实有测试失败
        (1, {"passed": 0, "failed": 1, "errors": 0, "skipped": 0}, "failed"),
        (1, {"passed": 2, "failed": 1, "errors": 0, "skipped": 0}, "failed"),
        # 2/3/4/5：中断、内部错误、用法错误、未收集到测试
        (5, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}, "error"),
        (3, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}, "error"),
    ],
)
def test_status_truth_table(exit_code: int, counts: dict[str, int], expected: str) -> None:
    assert _status_from_result(exit_code, counts) == expected


def test_no_tests_message_is_exported() -> None:
    """措辞是**运行结论**的一部分，下游要能按名字引用它而不是各自复制字符串。"""
    assert validation.NO_TESTS_EXECUTED_MESSAGE == (
        "pytest exited successfully but no tests were executed"
    )
    assert validation.NO_TESTS_FAILURE_NAME
    assert "NO_TESTS_EXECUTED_MESSAGE" in validation.__all__
    assert "NO_TESTS_FAILURE_NAME" in validation.__all__


# ============================================== 真实 pytest 输出上的端到端判据


def test_all_skipped_run_is_not_passed() -> None:
    """conftest 把全部用例标记为 skip：退出码 0，但一条都没执行。"""
    result = validation.run_tests(_zero_test_artifacts())

    assert result.exit_code == 0, "这条路径的前提就是 pytest 成功退出"
    assert result.passed == 0
    assert result.failed == 0
    assert result.skipped > 0
    assert result.status == "error", f"0 个测试执行却被判定为 {result.status!r}"


def test_all_skipped_run_explains_itself() -> None:
    """不能只给一个没有失败详情的 error——下游无从判断这次运行到底发生了什么。"""
    result = validation.run_tests(_zero_test_artifacts())

    details = result.failure_details
    messages = [detail.message for detail in details]
    names = [detail.test_name for detail in details]
    assert any(validation.NO_TESTS_EXECUTED_MESSAGE in message for message in messages), details
    assert any(name == validation.NO_TESTS_FAILURE_NAME for name in names), details


def test_collect_only_run_is_not_passed() -> None:
    """`--collect-only`：退出码 0、测试"收集到了 N 个"，但一条都没执行。

    这条路径没有任何 FAILED/ERROR 摘要行，也没有 skipped 计数，只有一句
    "N tests collected"——靠"没有失败就算通过"的判据必然被骗过去。
    """
    artifacts = _artifacts({"pytest.ini": COLLECT_ONLY_INI, "tests/test_ok.py": PASSING_TEST})
    result = validation.run_tests(artifacts)

    assert result.exit_code == 0
    assert result.passed == 0
    assert result.status == "error", f"只收集未执行却被判定为 {result.status!r}"


def test_deselect_everything_is_not_passed() -> None:
    """收集结果被清空：pytest 以 5 退出，走的是"非 0 且非 1"那条分支。"""
    artifacts = _artifacts({"conftest.py": DESELECT_ALL_CONFTEST, "tests/test_ok.py": PASSING_TEST})
    result = validation.run_tests(artifacts)

    assert result.exit_code == 5
    assert result.passed == 0
    assert result.status == "error"


def test_skipped_only_run_is_not_passed() -> None:
    """只有一个被 skip 的测试：既没有 passed 也没有 failed，同样什么都没验证。"""
    artifacts = _artifacts(
        {
            "tests/test_skipped.py": (
                "import pytest\n\n\n"
                "@pytest.mark.skip(reason='not runnable here')\n"
                "def test_skipped():\n"
                "    assert False\n"
            )
        }
    )
    result = validation.run_tests(artifacts)

    assert result.exit_code == 0
    assert result.skipped == 1
    assert result.passed == 0
    assert result.status == "error"


def test_genuinely_passing_run_is_still_passed() -> None:
    """反向用例：真的跑通了必须仍然是 "passed"。

    没有这一条，把判据改成"永远返回 error"也能让上面所有用例变绿。
    """
    result = validation.run_tests(_artifacts({"tests/test_ok.py": PASSING_TEST}))

    assert result.exit_code == 0
    assert result.passed == 1
    assert result.failed == 0
    assert result.status == "passed"
    assert result.failure_details == []


def test_genuinely_failing_run_is_still_failed() -> None:
    """同样地，真的失败了必须仍然是 "failed"。"""
    result = validation.run_tests(_artifacts({"tests/test_bad.py": FAILING_TEST}))

    assert result.exit_code == 1
    assert result.failed == 1
    assert result.status == "failed"


# ==================================================== Repair Loop 不短路


def test_repair_loop_does_not_short_circuit_on_zero_tests() -> None:
    """Repair Loop 的第一条终止条件就是 `test_result.status == "passed"`。

    修复前这里会**立即**返回 status="passed"，一次修复都不做——整条 pipeline
    宣告成功，而那个本该失败的测试从未运行过。
    """
    result = run_repair_loop(_zero_test_artifacts(), max_iterations=1)

    assert result.status != "passed", "0 个测试执行却让 Repair Loop 判定成功"
    assert result.test_result.status == "error"
    assert result.test_result.passed == 0


def test_repair_loop_still_passes_a_real_success() -> None:
    """反向用例：真跑通了仍然要短路成 passed，修复不能把正常路径也堵死。"""
    result = run_repair_loop(_artifacts({"tests/test_ok.py": PASSING_TEST}), max_iterations=1)

    assert result.status == "passed"
    assert result.iterations == 0, "首轮通过就不应该再启动修复"
    assert result.test_result.passed == 1


# ============================================================ 契约与边界


def test_zero_tests_never_reports_a_passing_status() -> None:
    """把不变量写死在 TestResult 上：status 与数量统计不能自相矛盾。"""
    cases = [
        _zero_test_artifacts(),
        _artifacts({"pytest.ini": COLLECT_ONLY_INI, "tests/test_ok.py": PASSING_TEST}),
        _artifacts({"conftest.py": DESELECT_ALL_CONFTEST, "tests/test_ok.py": PASSING_TEST}),
    ]
    for artifacts in cases:
        result = validation.run_tests(artifacts)
        if result.status == "passed":
            assert result.passed > 0, "passed 必须有真实通过的测试支撑"
            assert result.failed == 0
            assert result.errors == 0


def test_test_result_is_not_self_contradictory_after_a_fix() -> None:
    """把不变量钉在 TestResult 上：status 与数量统计不能脱节。

    注：这条不变量由 **DeterministicTestRunner**（生产路径上唯一的 TestResult
    产出方）保证，而不是模型层的 validator——`TestResult` 保持原样，模型本身
    允许构造出 `status="passed", passed=0`，只是 runner 永远不会产出它。
    加模型校验等于给所有调用方（含既有测试夹具）新增一条可能踩中的约束，
    超出本次"最小修改"的范围。
    """
    result = validation.run_tests(_zero_test_artifacts())
    assert not (result.status == "passed" and result.passed == 0)

    ok = validation.run_tests(_artifacts({"tests/test_ok.py": PASSING_TEST}))
    assert ok.status == "passed" and ok.passed > 0


def test_error_result_is_serializable_and_has_no_local_paths() -> None:
    """error 结果要能进 API 响应：不得出现临时工作区的绝对路径。"""
    result = validation.run_tests(_zero_test_artifacts())
    payload = result.model_dump_json()

    assert "apiforge-tests-" not in payload
    assert str(Path.home()) not in payload
    assert textwrap.dedent(payload)  # 非空且是合法 JSON 字符串
