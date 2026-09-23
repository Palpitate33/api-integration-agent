"""Repository Scanner 遇到不可访问目录时的行为（P1）。

背景
----
扫描一个真实仓库时，"某个子目录读不了"是**预期之内**的失败：权限不足、目录在
扫描途中被删掉、Windows 上的保留目录、容器里挂进来的只读卷…… 这类目录通常与
本次集成毫无关系，却能让整个 Repository Understanding 断掉——代价是 Pipeline
拿不到**任何**仓库上下文。

``os.walk`` 的缺省行为是 ``onerror=None``：**静默**丢弃错误。于是真正的问题不是
"崩溃"，而是"扫描结果悄悄少了几个目录，调用方却以为看到了全部"。

本文件用受控的 monkeypatch 模拟 PermissionError，而不是去构造一个真的不可读的
目录：Windows 的权限模型（``icacls``）与 POSIX 不同，真造目录会让测试随机 skip，
而 skip 掉的回归测试等于没有。这里替换 ``os.scandir``，只对一个指定路径抛错，
其余路径原样透传——被验证的仍是 scanner 自己那段处理逻辑。
"""

import errno
import logging
import os
from pathlib import Path

import pytest

from integration_agent.api import parse_openapi_text
from integration_agent.repository import RepositoryError, scan_repository
from integration_agent.repository.scanner import iter_python_files
from integration_agent.tools import InspectProjectTool, ToolContext

SCANNER_LOGGER = "integration_agent.repository.scanner"


def _make_project(root: Path) -> Path:
    """最小项目：``pkg/`` 可读，``locked/`` 留给测试去拒绝。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    locked = root / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("SECRET = 1\n", encoding="utf-8")
    return root


def _deny_scandir(monkeypatch: pytest.MonkeyPatch, target: Path, factory) -> list[str]:
    """让 ``os.scandir`` 在扫描 ``target`` 时抛出 ``factory()``。

    只拦这一个路径，其余一律透传给真正的 ``os.scandir``——测试要验证的是
    scanner 如何处理权限失败，而不是把整个文件系统换掉。
    """
    denied = target.resolve()
    real_scandir = os.scandir
    hits: list[str] = []

    def fake_scandir(path=".", *args, **kwargs):
        # os.scandir 也接受文件描述符；非路径入参一律透传
        if isinstance(path, (str, bytes, os.PathLike)):
            try:
                resolved = Path(path).resolve()
            except OSError:
                resolved = None
            if resolved == denied:
                hits.append(str(resolved))
                raise factory()
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", fake_scandir)
    return hits


def _permission_error(target: Path) -> PermissionError:
    return PermissionError(errno.EACCES, "Access is denied", str(target))


# ------------------------------------------ 不可访问的子目录：跳过并记录


def test_scan_survives_an_inaccessible_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path / "proj")
    hits = _deny_scandir(
        monkeypatch, project / "locked", lambda: _permission_error(project / "locked")
    )

    result = scan_repository(project)  # 不抛异常

    assert hits, "monkeypatch 没有生效：os.scandir 从未被拦截"
    assert result.skipped_paths == ["locked"]


def test_files_under_an_inaccessible_directory_are_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """跳过就是真的跳过：读不到的内容绝不能出现在快照里。"""
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project / "locked", lambda: _permission_error(project / "locked"))

    result = scan_repository(project)

    assert "pkg/mod.py" in result.python_files
    assert not [path for path in result.python_files if path.startswith("locked/")]


def test_scan_skips_only_the_inaccessible_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一个子目录读不了，不该连累它的兄弟目录。"""
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project / "locked", lambda: _permission_error(project / "locked"))

    result = scan_repository(project)

    assert result.is_python_project is True
    assert result.name == "demo"
    assert set(result.source_dirs) >= {"pkg"}
    assert sorted(result.python_files) == ["pkg/__init__.py", "pkg/mod.py"]


def test_skipped_directory_is_logged_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """跳过必须留下痕迹：静默变少的扫描结果比报错更难查。"""
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project / "locked", lambda: _permission_error(project / "locked"))

    with caplog.at_level(logging.WARNING, logger=SCANNER_LOGGER):
        scan_repository(project)

    assert any("locked" in record.getMessage() for record in caplog.records)


def test_healthy_scan_reports_nothing_skipped(tmp_path: Path) -> None:
    """能读全的时候 skipped_paths 必须为空——否则这个字段就失去了信号意义。"""
    project = _make_project(tmp_path / "proj")

    result = scan_repository(project)

    assert result.skipped_paths == []
    # locked/ 这次读得到，所以它的内容**应当**在结果里
    assert set(result.python_files) == {"pkg/__init__.py", "pkg/mod.py", "locked/hidden.py"}


def test_inaccessible_root_is_reported_not_silently_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """根目录本身读不了：过去会得到一份**静默为空**的"项目"。

    空结果与"这个项目没有任何 .py 文件"无法区分，Planner 会照着一份假事实做计划。
    """
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project, lambda: _permission_error(project))

    result = scan_repository(project)

    assert result.python_files == []
    assert result.skipped_paths == ["."]  # 明说了"这份快照不完整"


def test_skipped_paths_are_deduplicated_and_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一个目录会被两条代码路径碰到（遍历 + 识别源码目录），只应记一次。"""
    project = _make_project(tmp_path / "proj")
    for name in ("zzz_locked", "aaa_locked"):
        (project / name).mkdir()
    for name in ("zzz_locked", "aaa_locked"):
        _deny_scandir(monkeypatch, project / name, lambda n=name: _permission_error(project / n))

    result = scan_repository(project)

    assert result.skipped_paths == ["aaa_locked", "zzz_locked"]


# ------------------------------------- 只吞文件系统异常，别的照旧向上抛


def test_non_filesystem_errors_still_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """遍历器自身的 bug 不是"权限问题"，绝不能被当成可跳过的失败吞掉。

    这条测试是 ``except Exception: pass`` 的守门人：一旦有人把兜底放宽成
    "什么都接住"，它会立刻失败。
    """
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project / "locked", lambda: RuntimeError("遍历器坏了"))

    with pytest.raises(RuntimeError, match="遍历器坏了"):
        scan_repository(project)


def test_iter_python_files_propagates_non_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project / "locked", lambda: RuntimeError("遍历器坏了"))

    with pytest.raises(RuntimeError, match="遍历器坏了"):
        list(iter_python_files(project))


def test_iter_python_files_default_handler_still_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """不传 ``on_error`` 时也不能退回 os.walk 的静默缺省。"""
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project / "locked", lambda: _permission_error(project / "locked"))

    with caplog.at_level(logging.WARNING, logger=SCANNER_LOGGER):
        files = list(iter_python_files(project))

    assert sorted(path.name for path in files) == ["__init__.py", "mod.py"]
    assert any("locked" in record.getMessage() for record in caplog.records)


def test_custom_on_error_receives_the_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """调用方可以接管记录方式；收到的必须是原始 OSError，带得出错路径。"""
    project = _make_project(tmp_path / "proj")
    _deny_scandir(monkeypatch, project / "locked", lambda: _permission_error(project / "locked"))
    seen: list[OSError] = []

    list(iter_python_files(project, on_error=seen.append))

    assert len(seen) == 1
    assert isinstance(seen[0], PermissionError)
    assert seen[0].filename == str((project / "locked").resolve())


# ------------------------------------------------- 容忍失败 ≠ 降低路径安全


def test_validation_is_not_weakened_by_the_tolerance() -> None:
    """把"读不了就跳过"做进扫描，不等于"什么都不校验了"。"""
    with pytest.raises(RepositoryError):
        scan_repository("definitely/not/a/real/directory")


def test_a_file_path_is_still_rejected_as_a_root(tmp_path: Path) -> None:
    target = tmp_path / "a_file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(RepositoryError):
        scan_repository(target)


# ------------------------------------------- 跳过的事实要能被上层看见


def _context(project_root: Path, project) -> ToolContext:
    api = parse_openapi_text(
        """
        openapi: 3.0.3
        info: {title: T, version: "1"}
        paths: {/ping: {get: {operationId: ping, responses: {"200": {description: ok}}}}}
        """
    )
    return ToolContext(
        project_root=project_root, project=project, api=api, known_files=set(project.python_files)
    )


def test_inspect_project_surfaces_skipped_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """skipped_paths 必须是活的：模型下判断之前就该看到"这份快照不完整"。"""
    project_root = _make_project(tmp_path / "proj")
    _deny_scandir(
        monkeypatch, project_root / "locked", lambda: _permission_error(project_root / "locked")
    )
    project = scan_repository(project_root)

    result = InspectProjectTool().invoke({}, _context(project_root, project))

    assert result.ok is True
    assert "skipped_inaccessible" in result.content
    assert "locked" in result.content


def test_inspect_project_says_nothing_when_nothing_was_skipped(tmp_path: Path) -> None:
    """一切正常时不该多出这一节——常驻噪音会让真正的警告失去分量。"""
    project_root = _make_project(tmp_path / "proj")
    project = scan_repository(project_root)

    result = InspectProjectTool().invoke({}, _context(project_root, project))

    assert result.ok is True
    assert "skipped_inaccessible" not in result.content
