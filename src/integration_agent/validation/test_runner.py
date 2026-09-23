"""Deterministic Test Runner：在隔离的临时工作区运行生成产物中的测试。

契约：GeneratedArtifacts → TestResult。

工作流程：
    1. 创建临时工作区（TemporaryDirectory）
    2. 把 artifacts.created_files 写入工作区；modify 片段没有原文件上下文，
       不落地，也绝不触碰真实仓库
    3. 写入工作区专用的项目元数据（依赖清单），并对依赖做离线检查
    4. 把 runner-owned 插件复制进工作区，再用 `-p` 把它注入 pytest
    5. 在工作区内运行 pytest（subprocess + timeout + stdout/stderr 捕获）
    6. 计数只从插件写出的统计文件读；stdout 仅用于失败详情与调试
    7. 返回结构化 TestResult；工作区随上下文自动清理

信任边界（计数从哪里来）：
    生成测试是**不可信代码**，而它和 pytest 跑在同一个进程里，因此 stdout
    完全由它书写：一行 `print("5 passed in 0.01s")` 就能伪造摘要，conftest 里的
    `os._exit(0)` 能让 pytest 连摘要都打不出来。所以 passed/failed/errors/skipped
    一律取自 runner-owned 插件（apiforge_runner_plugin）在 `pytest_sessionfinish`
    里写出的统计文件：**session 正常结束才有结论**。文件缺失或无法解析时，
    本次运行判定为 "error"（"没能验证" ≠ "验证通过"），stdout 里的任何数字都
    不参与状态判定。

安全边界：
    - 绝不修改真实 Repository（只写 TemporaryDirectory）。
    - 绝不联网安装依赖（默认 OfflineDependencyPreparer 只检查当前解释器可导入性）。
    - 绝不修复失败代码：TestResult(status="failed") 即终点，Repair Loop 是下一阶段。
    - 子进程只拿到最小环境（build_subprocess_env）：生成代码属于不可信代码，
      不能通过 os.environ 读到宿主进程的 API 凭证。
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from integration_agent.generation import DependencyChange, GeneratedArtifacts
from integration_agent.trace import emit
from integration_agent.validation.models import FailureDetail, TestResult

DEFAULT_TIMEOUT = 30.0
DEFAULT_PYTEST_ARGS = ("-q", "-ra", "--tb=short")

# runner-owned 插件：仓库里的单一来源 → 复制进工作区 → `-p` 注入。
# 复制而不是 `-p integration_agent...`：子进程的 PYTHONPATH 只有工作区，插件能否
# 加载不该取决于宿主包是以 editable 还是其它方式安装的（缺了它每次运行都会变成
# "没有可信统计"，那是 fail-safe 但会让功能整体失效）。
RUNNER_PLUGIN_SOURCE = "apiforge_runner_plugin.py"
RUNNER_PLUGIN_MODULE = "apiforge_runner_plugin"
RUNNER_STATS_FILENAME = ".apiforge-runner-stats.json"
# 与插件里的 STATS_PATH_ENV_VAR 必须一致；test_runner 不 import 插件模块
# （pytest 是 dev 依赖，生产 import 不能硬依赖它），一致性由测试钉住。
RUNNER_STATS_ENV_VAR = "APIFORGE_RUNNER_STATS"

# "退出码 0 但没有测试执行"的固定措辞：它是**运行结论**的一部分，所以做成常量，
# 让下游（Repair Planner 的分类、报告、测试断言）不必各自复制这段中文/英文。
NO_TESTS_EXECUTED_MESSAGE = "pytest exited successfully but no tests were executed"
NO_TESTS_FAILURE_NAME = "<no tests executed>"

# "pytest session 没有正常结束"（统计文件缺失/不可解析）的固定措辞。
SESSION_NOT_COMPLETED_MESSAGE = "pytest session did not complete; result cannot be trusted"
SESSION_NOT_COMPLETED_NAME = "<session not completed>"

# 子进程环境 allowlist：**默认拒绝，明确放行**。
#
# 为什么用 allowlist 而不是"复制 os.environ 再删掉已知 secret"：黑名单只能挡住
# 今天想得到的名字（DEEPSEEK_API_KEY / AWS_* / ...），挡不住明天新增的凭证
# （CI 注入的新 token、.env 风格的新命名）。allowlist 的结构保证"没被列出的
# 变量一律不进入子进程"，新增凭证默认是安全的。
#
# 只放行 Python / pytest 启动所必需、且**本身不携带凭证**的变量。
# 每一项都由 tests/test_test_runner.py 的隔离测试与全量 pytest 守着。
SUBPROCESS_ENV_ALLOWLIST = (
    # 可执行文件与 DLL 解析。SystemRoot 是硬需求：Windows 上缺失会让子进程
    # 直接起不来（socket 初始化失败 → OSError WinError 10106）。
    "PATH",
    "PATHEXT",
    "SystemRoot",
    "SystemDrive",
    "windir",
    "ComSpec",
    # 临时目录：pytest 与 tempfile 都要写文件
    "TEMP",
    "TMP",
    # 用户主目录：Path.home()、pytest 缓存目录、生成代码里的 ~ 展开
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    # 语言 / 编码（PYTHONIOENCODING 由本模块显式设定，不继承）
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)


@runtime_checkable
class TestRunner(Protocol):
    """Test Runner 接口：确定性实现与未来实现共用同一契约。"""

    def run(self, artifacts: GeneratedArtifacts) -> TestResult:
        """运行生成产物中的测试并返回结构化结果。"""
        ...


class DependencyPreparer(Protocol):
    """依赖准备接口：默认为离线检查，可替换为其他策略。"""

    def prepare(self, changes: list[DependencyChange]) -> list[str]:
        """返回未满足的依赖说明；不负责联网安装。"""
        ...


class OfflineDependencyPreparer:
    """离线依赖准备：只检查当前解释器能否导入，绝不联网安装、绝不写真实项目清单。"""

    def prepare(self, changes: list[DependencyChange]) -> list[str]:
        warnings: list[str] = []
        for change in changes:
            if change.action != "add":
                continue
            if importlib.util.find_spec(change.name) is None:
                warnings.append(
                    f"依赖 {change.name}{change.version or ''} 未安装（离线模式，不自动安装）"
                )
        return warnings


def run_tests(artifacts: GeneratedArtifacts) -> TestResult:
    """便捷入口：默认使用 DeterministicTestRunner。"""
    return DeterministicTestRunner().run(artifacts)


def build_subprocess_env(workspace: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """构建 pytest 子进程的最小环境：父进程环境**默认不继承**。

    生成产物与生成测试都是不可信代码，它们可以在 pytest 子进程里直接读
    ``os.environ``。若整体继承，宿主进程的 DEEPSEEK_API_KEY / OPENAI_API_KEY /
    AWS_SECRET_ACCESS_KEY / DATABASE_URL 等凭证会直接暴露给这些代码。

    因此这里只按 SUBPROCESS_ENV_ALLOWLIST 放行 Python / pytest 启动所必需、
    且本身不携带凭证的变量，其余（含一切凭证类变量）一律不进入子进程。

    base 仅用于测试注入（默认读取当前进程的 os.environ），生产路径不传。
    """
    source = os.environ if base is None else base
    env = {name: source[name] for name in SUBPROCESS_ENV_ALLOWLIST if name in source}
    # PYTHONPATH 不继承父进程：生成代码只能 import 工作区内的模块与已安装的包。
    env["PYTHONPATH"] = str(workspace)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class DeterministicTestRunner:
    """在临时工作区运行 pytest 的确定性 Runner。

    同样输入必定产出同样的核心结果（状态/数量/失败详情）；
    不包含绝对临时路径、随机 ID 等不稳定信息。
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        pytest_args: list[str] | None = None,
        workspace_parent: Path | None = None,
        dependency_preparer: DependencyPreparer | None = None,
    ) -> None:
        self.timeout = timeout
        self.pytest_args = (
            list(pytest_args) if pytest_args is not None else list(DEFAULT_PYTEST_ARGS)
        )
        self.workspace_parent = workspace_parent
        self.dependency_preparer = dependency_preparer or OfflineDependencyPreparer()

    def run(self, artifacts: GeneratedArtifacts) -> TestResult:
        started = time.perf_counter()
        emit(
            "test_runner",
            "test_started",
            "开始运行生成产物的测试",
            metadata={"files": len(artifacts.created_files), "timeout": self.timeout},
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix="apiforge-tests-", dir=self.workspace_parent
            ) as tmp:
                workspace = Path(tmp)
                dependency_warnings = self.dependency_preparer.prepare(artifacts.dependency_changes)
                self._materialize(artifacts, workspace)
                self._write_metadata(artifacts, workspace)
                # 复制放在产物落地**之后**：工作区里这个文件名由 runner 独占，
                # 生成产物无法抢占它（抢了也被覆盖）。
                _install_runner_plugin(workspace)
                result = self._run_pytest(workspace, dependency_warnings)
        except Exception as exc:  # 环境级失败（如路径越界）统一转为 error 结果
            result = TestResult(
                status="error",
                exit_code=None,
                duration=0.0,
                stderr=f"Test Runner 执行失败：{exc}",
            )
        emit(
            "test_runner",
            "test_completed",
            "测试运行结束",
            # 记的是 TestResult 的**摘要**，不是重新解析 pytest 的结论：TestResult
            # 仍然是测试结果的唯一结构化事实来源，这里只是它的镜像——两者不一致时，
            # 错的一定是这里。事件级 duration 是整个 runner 的墙钟（含工作区准备），
            # metadata 里的 duration 才是 pytest 自己测得的测试耗时。
            metadata={
                "status": result.status,
                "passed": result.passed,
                "failed": result.failed,
                "errors": result.errors,
                "skipped": result.skipped,
                "duration": result.duration,
                "exit_code": result.exit_code,
            },
            duration=time.perf_counter() - started,
            status=result.status,
        )
        return result

    @staticmethod
    def _materialize(artifacts: GeneratedArtifacts, workspace: Path) -> None:
        """只落地 create 文件；modify 片段无原文件上下文，不落地也绝不触碰真实仓库。"""
        for item in artifacts.created_files:
            target = _safe_join(workspace, item.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.content, encoding="utf-8")

    @staticmethod
    def _write_metadata(artifacts: GeneratedArtifacts, workspace: Path) -> None:
        """把依赖变更写入工作区专用的项目元数据（真实项目的清单不受影响）。"""
        deps = [
            f'"{change.name}{change.version or ""}"'
            for change in artifacts.dependency_changes
            if change.action == "add"
        ]
        content = '[project]\nname = "apiforge-test-workspace"\nversion = "0.1.0"\n'
        content += "dependencies = [" + ", ".join(deps) + "]\n"
        (workspace / "pyproject.toml").write_text(content, encoding="utf-8")

    def _run_pytest(self, workspace: Path, dependency_warnings: list[str]) -> TestResult:
        # 最小环境：不继承父进程环境，见 build_subprocess_env
        env = build_subprocess_env(workspace)
        stats_path = workspace / RUNNER_STATS_FILENAME
        env[RUNNER_STATS_ENV_VAR] = str(stats_path)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            RUNNER_PLUGIN_MODULE,
            *self.pytest_args,
        ]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return TestResult(
                status="timeout",
                exit_code=None,
                duration=round(time.monotonic() - start, 3),
                stdout=exc.stdout or "",
                stderr=f"测试超过 {self.timeout}s 超时，pytest 子进程已被终止",
                dependency_warnings=dependency_warnings,
            )
        except FileNotFoundError as exc:
            return TestResult(
                status="error",
                exit_code=None,
                duration=round(time.monotonic() - start, 3),
                stderr=f"无法启动 pytest：{exc}",
                dependency_warnings=dependency_warnings,
            )
        duration = time.monotonic() - start
        # stdout 只用来取失败详情（人看的部分），**不参与计数**：见模块级"信任边界"。
        failure_details = _parse_failure_details(
            completed.stdout, _parse_summary_failures(completed.stdout)
        )
        stats = _read_runner_stats(stats_path, exit_code=completed.returncode)
        if stats is None:
            # session 没有正常结束（os._exit / sys.exit / hook 之前崩溃），或写出的
            # 统计与进程退出码对不上。这是"没能验证"，既不是通过也不是失败。
            return TestResult(
                status="error",
                exit_code=completed.returncode,
                duration=round(duration, 3),
                stdout=completed.stdout,
                stderr=_with_note(completed.stderr, SESSION_NOT_COMPLETED_MESSAGE),
                failure_details=[
                    FailureDetail(
                        test_name=SESSION_NOT_COMPLETED_NAME,
                        message=SESSION_NOT_COMPLETED_MESSAGE,
                    ),
                    *failure_details,
                ],
                dependency_warnings=dependency_warnings,
            )
        counts = stats.counts
        status = _status_from_result(completed.returncode, counts)
        if not failure_details:
            # stdout 里读不到详情（被伪造、被截断）时用插件记录的 nodeid 兜底，
            # 免得下游拿到一个没有失败详情的 failed。
            failure_details = stats.as_details()
        if status == "error" and completed.returncode == 0:
            # 退出码成功、却一个测试都没执行：这是**假阳性**，必须留下可追溯的说明，
            # 否则下游只看到一个没有失败详情的 error，无从判断是环境问题还是这次运行
            # 根本没验证任何东西。
            failure_details = [_no_tests_detail(counts)] + failure_details
        return TestResult(
            status=status,
            exit_code=completed.returncode,
            passed=counts["passed"],
            failed=counts["failed"],
            errors=counts["errors"],
            skipped=counts["skipped"],
            duration=round(duration, 3),
            stdout=completed.stdout,
            stderr=completed.stderr,
            failure_details=failure_details,
            dependency_warnings=dependency_warnings,
        )


# ------------------------------------------- runner-owned 统计（唯一可信计数源）


@dataclass(frozen=True)
class RunnerStats:
    """runner-owned 插件写出的统计文件内容。

    这是 TestResult 计数字段的**唯一**来源：它由插件在 ``pytest_sessionfinish``
    里写出，因此只有在 pytest session 正常结束时才存在。
    """

    counts: dict[str, int]
    collected: int
    exit_status: int
    failures: tuple[tuple[str, str], ...]  # (nodeid, 一行摘要)

    def as_details(self) -> list[FailureDetail]:
        """失败/错误测试 → FailureDetail（stdout 解析不到详情时的兜底）。"""
        details: list[FailureDetail] = []
        for nodeid, message in self.failures:
            file_part, separator, name = nodeid.rpartition("::")
            details.append(
                FailureDetail(
                    test_name=name if separator else nodeid,
                    file=file_part or None,
                    message=message,
                )
            )
        return details


def _install_runner_plugin(workspace: Path) -> None:
    """把插件复制进工作区：`-p` 按模块名加载，而子进程的 sys.path 只有工作区。"""
    shutil.copyfile(
        Path(__file__).with_name(RUNNER_PLUGIN_SOURCE), workspace / RUNNER_PLUGIN_SOURCE
    )


def _int_field(payload: dict, name: str) -> int | None:
    """非负整数才接受；bool 是 int 的子类，必须显式排除。"""
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _read_runner_stats(path: Path, *, exit_code: int) -> RunnerStats | None:
    """读取 runner-owned 统计；缺失 / 格式不对 / 与退出码矛盾时返回 None。

    返回 None 的语义是"这次运行没有可信结论"，调用方据此判 error。宁可少报一次
    真实通过，也不能凭一份来路不明的文件宣布通过。exit_status 与进程退出码的
    一致性检查是廉价的异常探测：不一致说明这份统计不代表这次运行。
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    counts: dict[str, int] = {}
    for field in ("passed", "failed", "errors", "skipped"):
        value = _int_field(payload, field)
        if value is None:
            return None
        counts[field] = value
    collected = _int_field(payload, "collected")
    exit_status = _int_field(payload, "exit_status")
    if collected is None or exit_status is None or exit_status != exit_code:
        return None
    failures: list[tuple[str, str]] = []
    for entry in payload.get("failures") or ():
        if not isinstance(entry, dict):
            return None
        nodeid, message = entry.get("nodeid"), entry.get("message")
        if not isinstance(nodeid, str) or not isinstance(message, str):
            return None
        failures.append((nodeid, message))
    return RunnerStats(
        counts=counts, collected=collected, exit_status=exit_status, failures=tuple(failures)
    )


def _with_note(stderr: str, note: str) -> str:
    """在子进程 stderr 后面追加一行结论说明（子进程输出本身不丢）。"""
    return f"{stderr.rstrip()}\n{note}" if stderr.strip() else note


# --------------------------------------------- 输出解析（仅用于失败详情与调试）


def _parse_summary_failures(stdout: str) -> list[tuple[str, str, str]]:
    """stdout 中的 FAILED/ERROR 摘要行。

    **只用于失败详情与调试**：计数一律不从这里来（见模块级"信任边界"）。
    """
    summary_failures: list[tuple[str, str, str]] = []
    for line in stdout.splitlines():
        matched = re.match(r"^(FAILED|ERROR)\s+(.+?)\s+-\s+(.*)$", line)
        if matched:
            summary_failures.append((matched.group(1), matched.group(2), matched.group(3)))
    return summary_failures


_SUMMARY_ENTRY = re.compile(r"^(FAILED|ERROR|SKIPPED|PASSED|XFAILED|XPASSED|DESELECTED)\s")
_COUNT_LINE = re.compile(r"^\s*\d+\s+\w+.*\s+in\s+[\d.]+s\s*$")


def _is_tb_boundary(line: str) -> bool:
    """traceback 段的边界：分隔线、short summary 条目、最终统计行。

    统计行包含耗时（"in 0.09s"），若混入 traceback 会破坏确定性。
    """
    return bool(
        re.match(r"^[=_-]{4,}", line) or _SUMMARY_ENTRY.match(line) or _COUNT_LINE.match(line)
    )


def _normalize_path(path: str) -> str:
    """统一为 POSIX 相对路径（Windows 的 tb 输出使用反斜杠）。"""
    return path.replace("\\", "/")


def _parse_failure_details(
    stdout: str, summary_failures: list[tuple[str, str, str]]
) -> list[FailureDetail]:
    """把 FAILED/ERROR 摘要与 --tb=short 的 traceback 定位合并为结构化失败详情。"""
    tb_locations: dict[str, tuple[str, int]] = {}
    tb_sections: dict[str, str] = {}
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        header = re.match(r"^_{4,}\s+(.+?)\s+_{4,}$", line)
        if not header:
            continue
        name = header.group(1).strip()
        section: list[str] = []
        end = index + 1
        while end < len(lines) and not _is_tb_boundary(lines[end]):
            section.append(lines[end])
            end += 1
        tb_sections[name] = "\n".join(section).strip()
        for section_line in section[:4]:
            located = re.match(r"^(.+\.py):(\d+): in (.+?)$", section_line)
            if located:
                tb_locations[located.group(3)] = (
                    _normalize_path(located.group(1)),
                    int(located.group(2)),
                )
                break
    details: list[FailureDetail] = []
    for _status, test_id, message in summary_failures:
        file_part, separator, name = test_id.rpartition("::")
        if not separator:  # 收集期错误（如 tests/test_x.py）没有 ::name
            name = file_part
            file_part = ""
        location = tb_locations.get(name)
        details.append(
            FailureDetail(
                test_name=name,
                file=location[0] if location else (file_part or None),
                line=location[1] if location else None,
                message=message,
                traceback=tb_sections.get(name),
            )
        )
    return details


def _no_tests_detail(counts: dict[str, int]) -> FailureDetail:
    """「退出码 0 但没执行任何测试」的结构化说明。"""
    executed = counts["passed"] + counts["failed"]
    detail = NO_TESTS_EXECUTED_MESSAGE
    if executed == 0 and counts["skipped"] > 0:
        detail += f"（{counts['skipped']} 个测试被跳过，跳过的测试不算执行）"
    return FailureDetail(test_name=NO_TESTS_FAILURE_NAME, message=detail)


def _status_from_result(exit_code: int, counts: dict[str, int]) -> str:
    """(退出码, 数量统计) → TestResult.status。

    只看退出码是不够的：pytest 在**一个测试都没真正执行**时也可能返回 0
    （收集被 conftest 清空、全部用例被跳过、`--collect-only` 之类）。
    那种情况报 "passed" 是假阳性——下游（Repair Loop）会据此直接判定集成成功，
    交付一份从未被验证过的代码。**"没能验证" 不等于 "验证通过"**。

    所以 "passed" 要求三件事同时成立：退出码为 0、**至少有测试通过**（passed > 0）、
    且没有失败与错误。skipped 不计入执行：全部跳过的运行同样什么都没验证。

    "error" 而不是 "failed"：这不是"代码有问题"，是"这次运行没有证据"。
    """
    if exit_code == 0:
        if counts["passed"] > 0 and counts["failed"] == 0 and counts["errors"] == 0:
            return "passed"
        return "error"
    if exit_code == 1:  # 存在失败/错误的测试
        return "failed"
    return "error"  # 2/3/4/5：中断、内部错误、用法错误、未收集到测试


def _safe_join(root: Path, relative: str) -> Path:
    """把生成产物路径安全解析到工作区内；拒绝 ".." 越界路径。"""
    parts = [part for part in relative.replace("\\", "/").split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"生成产物路径越界：{relative!r}")
    return root.joinpath(*parts)
