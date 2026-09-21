"""Runner-owned pytest plugin：把 pytest session 的真实统计交给 TestRunner。

为什么需要它
------------
stdout 是**被测代码可以随意书写**的输出流：一个生成测试可以直接 print 一行
``5 passed in 0.01s``，也可以让 pytest 连摘要都打不出来（conftest 里 ``os._exit``
/ ``sys.exit``）。只要 TestRunner 从 stdout 解析 ``passed / failed`` 数量，
"测试到底有没有通过"这件事就变成了由被测代码自己宣布——那正是假阳性的来源。

本插件在 ``pytest_sessionfinish`` 里把 terminal reporter 的**真实统计**写成 JSON。
这个 hook 只在 pytest session 正常走到结束时才被调用：

    - 被 ``os._exit`` 掐断、或 hook 之前崩溃 → 文件根本不会出现 →
      TestRunner 判 "这次运行没有可信结论"（``status="error"``），而不是 passed；
    - 文件存在 → 计数只认这份文件，stdout 里的任何数字都不再参与判定。

归属
----
本文件是**仓库里**的单一来源，由 TestRunner 复制进临时工作区并以 ``-p`` 加载
（子进程的 PYTHONPATH 只有工作区，因此插件能否加载不能依赖宿主包是否已安装）。
结果文件的路径经环境变量传入，由 TestRunner 决定；生成产物既不能选择插件，
也不能选择结果文件的位置。

需要注意的是：生成代码与 pytest 跑在**同一个进程**里，因此这不是密码学意义上
的防伪——本插件的价值在于"正常结束才有结论"，而不是"对抗一个专门伪造本文件的
conftest"。那类对抗需要把被测代码放进独立沙箱，超出本模块职责。
"""

import json
import os
import re
from pathlib import Path

import pytest

# 结果文件路径的环境变量名。TestRunner 显式设置；缺省时本插件不写任何东西
# （写不出文件 = 没有可信统计，TestRunner 会判 error，而不是当作通过）。
STATS_PATH_ENV_VAR = "APIFORGE_RUNNER_STATS"

# terminal reporter 的 stats 键 → 结果文件的计数字段。
# xfail/xpass 不计入：它们与 TestResult 的四个计数字段没有对应关系。
_COUNT_KEYS = (
    ("passed", "passed"),
    ("failed", "failed"),
    ("error", "errors"),
    ("skipped", "skipped"),
)

_LEADING_E = re.compile(r"^E\s+")


def _terminal_stats(config: pytest.Config) -> dict:
    """terminal reporter 的原始 stats 字典；拿不到就返回空字典。"""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # pragma: no cover - 默认终端插件始终存在
        return {}
    stats = getattr(reporter, "stats", None)
    return stats if isinstance(stats, dict) else {}


def _counts(config: pytest.Config) -> dict[str, int]:
    stats = _terminal_stats(config)
    return {field: len(stats.get(key, ())) for key, field in _COUNT_KEYS}


def _one_line(text: str) -> str:
    """取 longrepr 的最后一行非空文本，并去掉 pytest 的 ``E`` 前缀。"""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return _LEADING_E.sub("", stripped)
    return ""


def _failures(config: pytest.Config) -> list[dict[str, str]]:
    """失败/错误测试的 nodeid 与一行摘要。

    **仅供展示与兜底**：状态判定只看计数，不看这里的文本。
    """
    stats = _terminal_stats(config)
    entries: list[dict[str, str]] = []
    for key in ("failed", "error"):
        for report in stats.get(key, ()):
            longrepr = getattr(report, "longreprtext", "") or ""
            entries.append(
                {
                    "nodeid": str(getattr(report, "nodeid", "")),
                    "message": _one_line(longrepr),
                }
            )
    return entries


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """session 正常结束：把真实统计写成 JSON（唯一一次写入）。"""
    path = os.environ.get(STATS_PATH_ENV_VAR)
    if not path:
        return
    payload = {
        **_counts(session.config),
        "collected": int(session.testscollected),
        "exit_status": int(exitstatus),
        "failures": _failures(session.config),
    }
    try:
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    except OSError:  # 写不进去 = 没有可信统计，交给 TestRunner 判 error
        pass
