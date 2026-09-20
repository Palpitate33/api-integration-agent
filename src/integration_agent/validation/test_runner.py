"""Deterministic Test Runner：在隔离的临时工作区运行生成产物中的测试。

契约：GeneratedArtifacts → TestResult。

工作流程：
    1. 创建临时工作区（TemporaryDirectory）
    2. 把 artifacts.created_files 写入工作区；modify 片段没有原文件上下文，
       不落地，也绝不触碰真实仓库
    3. 写入工作区专用的项目元数据（依赖清单），并对依赖做离线检查
    4. 在工作区内运行 pytest（subprocess + timeout + stdout/stderr 捕获）
    5. 解析 pytest 输出（数量统计 + FAILED/ERROR 摘要 + traceback 定位）
    6. 返回结构化 TestResult；工作区随上下文自动清理

安全边界：
    - 绝不修改真实 Repository（只写 TemporaryDirectory）。
    - 绝不联网安装依赖（默认 OfflineDependencyPreparer 只检查当前解释器可导入性）。
    - 绝不修复失败代码：TestResult(status="failed") 即终点，Repair Loop 是下一阶段。
    - 子进程只拿到最小环境（build_subprocess_env）：生成代码属于不可信代码，
      不能通过 os.environ 读到宿主进程的 API 凭证。
"""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from integration_agent.generation import DependencyChange, GeneratedArtifacts
from integration_agent.validation.models import FailureDetail, TestResult

DEFAULT_TIMEOUT = 30.0
DEFAULT_PYTEST_ARGS = ("-q", "-ra", "--tb=short")

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

# pytest 摘要行中的数量词 → TestResult 字段名（"1 error" / "2 errors" 同义）
_SUMMARY_KEYS = {
    "passed": "passed",
    "failed": "failed",
    "error": "errors",
    "errors": "errors",
    "skipped": "skipped",
}


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
        try:
            with tempfile.TemporaryDirectory(
                prefix="apiforge-tests-", dir=self.workspace_parent
            ) as tmp:
                workspace = Path(tmp)
                dependency_warnings = self.dependency_preparer.prepare(artifacts.dependency_changes)
                self._materialize(artifacts, workspace)
                self._write_metadata(artifacts, workspace)
                return self._run_pytest(workspace, dependency_warnings)
        except Exception as exc:  # 环境级失败（如路径越界）统一转为 error 结果
            return TestResult(
                status="error",
                exit_code=None,
                duration=0.0,
                stderr=f"Test Runner 执行失败：{exc}",
            )

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
        command = [sys.executable, "-m", "pytest", *self.pytest_args]
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
        counts, summary_failures = _parse_summary(completed.stdout)
        failure_details = _parse_failure_details(completed.stdout, summary_failures)
        return TestResult(
            status=_status_from_exit_code(completed.returncode),
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


# ------------------------------------------------------------------ 输出解析


def _parse_summary(stdout: str) -> tuple[dict[str, int], list[tuple[str, str, str]]]:
    """解析 pytest 数量统计（如 "3 passed, 1 skipped in 0.05s"）与 FAILED/ERROR 摘要行。"""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    summary_failures: list[tuple[str, str, str]] = []
    summary_line = None
    for line in reversed(stdout.splitlines()):
        if re.search(r"in\s+[\d.]+s\s*$", line):
            summary_line = line
            break
    if summary_line:
        for match in re.finditer(r"(\d+)\s+(\w+)", summary_line):
            key = _SUMMARY_KEYS.get(match.group(2))
            if key:
                counts[key] = int(match.group(1))
    for line in stdout.splitlines():
        matched = re.match(r"^(FAILED|ERROR)\s+(.+?)\s+-\s+(.*)$", line)
        if matched:
            summary_failures.append((matched.group(1), matched.group(2), matched.group(3)))
    return counts, summary_failures


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


def _status_from_exit_code(exit_code: int | None) -> str:
    if exit_code == 0:
        return "passed"
    if exit_code == 1:  # 存在失败/错误的测试
        return "failed"
    return "error"  # 2/3/4/5：中断、内部错误、用法错误、未收集到测试


def _safe_join(root: Path, relative: str) -> Path:
    """把生成产物路径安全解析到工作区内；拒绝 ".." 越界路径。"""
    parts = [part for part in relative.replace("\\", "/").split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"生成产物路径越界：{relative!r}")
    return root.joinpath(*parts)
