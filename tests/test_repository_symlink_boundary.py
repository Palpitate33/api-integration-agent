"""P1-3 回归：search_code 的每一个最终读取目标都必须过项目边界。

根因
----
`iter_python_files` 用 ``os.walk(followlinks=False)``，它只挡住"顺着 symlink
**目录**往下走"，**挡不住 symlink 文件**：名字以 ``.py`` 结尾的链接照样出现在
``filenames`` 里。而 ``link.stat()`` / ``link.read_text()`` 都会跟随链接——
`pkg/leak.py -> /etc/passwd` 于是以"项目内文件"的名义被读进来，内容直接进入
搜索结果（进而进入 LLM 上下文）。目录那条路也不是完全免疫：联接（junction）
在 Windows 上不需要特权，容易造成"以为挡住了"的错觉。

对策
----
新增 `resolve_inside_project(root, candidate)`：**先 resolve，再判包含关系**。
顺序不能反——链接只有在 resolve 之后才现出原形，直接比较链接自身的路径永远
得到"在项目内"。`search_code` 的读取循环对每个候选都走一次这个判据，与
`read_file` 工具（tools/repository_tools.py 的 `_resolve_inside_project`）
用的是同一条边界语义。

测试策略
--------
Windows 上创建**文件**符号链接需要管理员权限或开发者模式（WinError 1314），
所以：

  - 能造链接的用例照常跑（造不出链接的用 `pytest.skip`，而不是假装通过，
    沿用 test_repository_tools.py 里 `_make_escape_link` 的约定）；
  - **另有一条不依赖任何链接权限的用例**，直接把一个越界候选喂给读取循环——
    它验证的是同一条判据，因此在任何平台都能跑，也是这条修复最直接的看门人。
"""

import os
import subprocess
from pathlib import Path

import pytest

from integration_agent.api import APIInfo
from integration_agent.repository import (
    code_search,
    resolve_inside_project,
    scan_repository,
    search_code,
)
from integration_agent.tools import ReadFileTool, ToolContext

OUTSIDE_SECRET = "OUTSIDE_SECRET_MUST_NOT_LEAK"
INSIDE_MARKER = "INSIDE_FILE_MARKER"


# ------------------------------------------------------------------ 夹具


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "client.py").write_text(f"{INSIDE_MARKER} = 1\n", encoding="utf-8")
    return root


def _outside_secret(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    secret = outside / "secret.py"
    secret.write_text(f"{OUTSIDE_SECRET} = 1\n", encoding="utf-8")
    return secret


def _link_file(target: Path, link: Path) -> None:
    """造一个文件符号链接；当前平台不支持时 skip（不假装通过）。"""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台无法创建文件符号链接（Windows 需要管理员权限或开发者模式）")


def _link_dir(target: Path, link: Path) -> str:
    """造一个目录链接，返回实际使用的方式（symlink 或 junction）。"""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return "junction"
    pytest.skip("当前平台无法创建符号链接或目录联接，无法验证逃逸防护")


def _ctx(root: Path) -> ToolContext:
    return ToolContext(
        project_root=root.resolve(),
        project=scan_repository(root),
        api=APIInfo(name="Fake API", version="1.0.0", base_url="https://api.example.com"),
        known_files=set(),
    )


def _all_text(result) -> str:
    """搜索结果里的**全部**文本，用于判断内容有没有泄漏。"""
    return "\n".join(match.snippet for match in result.matches)


# ============================================ resolve_inside_project（判据本身）


def test_inside_file_resolves(tmp_path: Path) -> None:
    root = _project(tmp_path).resolve()

    resolved = resolve_inside_project(root, root / "pkg" / "client.py")

    assert resolved is not None
    assert resolved.is_relative_to(root)
    assert resolved.name == "client.py"


def test_parent_traversal_is_rejected(tmp_path: Path) -> None:
    root = _project(tmp_path).resolve()
    secret = _outside_secret(tmp_path)

    assert resolve_inside_project(root, root / ".." / "outside" / "secret.py") is None
    assert resolve_inside_project(root, secret) is None


def test_absolute_path_outside_is_rejected(tmp_path: Path) -> None:
    root = _project(tmp_path).resolve()
    secret = _outside_secret(tmp_path)

    assert resolve_inside_project(root, str(secret)) is None
    assert resolve_inside_project(root, str(secret.parent)) is None


def test_sibling_directory_sharing_a_prefix_is_not_inside(tmp_path: Path) -> None:
    """`/tmp/root` 与 `/tmp/root_evil` 共享前缀，但不是同一个目录。

    朴素实现用字符串 startswith 判断，这里就会放行一个项目外的路径。
    """
    root = _project(tmp_path).resolve()
    evil = tmp_path / "root_evil"
    evil.mkdir()
    (evil / "secret.py").write_text(f"{OUTSIDE_SECRET} = 1\n", encoding="utf-8")

    assert resolve_inside_project(root, evil / "secret.py") is None


def test_dot_segments_inside_are_still_inside(tmp_path: Path) -> None:
    root = _project(tmp_path).resolve()

    resolved = resolve_inside_project(root, root / "pkg" / ".." / "pkg" / "client.py")

    assert resolved is not None
    assert resolved == (root / "pkg" / "client.py").resolve()


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = _project(tmp_path).resolve()
    secret = _outside_secret(tmp_path)
    link = root / "pkg" / "leak.py"
    _link_file(secret, link)

    assert link.exists(), "链接本身是可解析的（所以不是「文件不存在」挡住的）"
    assert resolve_inside_project(root, link) is None


def test_chained_symlink_escape_is_rejected(tmp_path: Path) -> None:
    """多级链接：一级在项目内、二级才出界。resolve 必须跟完全程。"""
    root = _project(tmp_path).resolve()
    secret = _outside_secret(tmp_path)
    first = root / "pkg" / "first.py"
    second = root / "pkg" / "second.py"
    _link_file(secret, first)
    _link_file(first, second)

    assert resolve_inside_project(root, second) is None


# ==================================================== search_code：正常路径


def test_ordinary_project_file_is_searchable(tmp_path: Path) -> None:
    root = _project(tmp_path)

    result = search_code(root, INSIDE_MARKER)

    assert [match.path for match in result.matches] == ["pkg/client.py"]
    assert result.files_scanned >= 1


def test_symlink_inside_the_project_keeps_working(tmp_path: Path) -> None:
    """目标仍在项目内的链接按现有策略处理：照常搜索，报告链接自身的位置。"""
    root = _project(tmp_path)
    _link_file(root / "pkg" / "client.py", root / "pkg" / "alias.py")

    result = search_code(root, INSIDE_MARKER)

    paths = sorted(match.path for match in result.matches)
    assert paths == ["pkg/alias.py", "pkg/client.py"]


# ==================================================== search_code：越界拒绝


def test_symlink_file_escaping_project_is_not_read(tmp_path: Path) -> None:
    root = _project(tmp_path)
    secret = _outside_secret(tmp_path)
    _link_file(secret, root / "pkg" / "leak.py")

    result = search_code(root, OUTSIDE_SECRET)

    assert result.matches == [], "项目外的内容被当成项目内文件搜出来了"
    assert OUTSIDE_SECRET not in _all_text(result)
    assert "pkg/leak.py" not in [match.path for match in result.matches]


def test_symlink_directory_escaping_project_is_not_traversed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside"
    _outside_secret(tmp_path)
    _link_dir(outside, root / "linked_outside")

    result = search_code(root, OUTSIDE_SECRET)

    assert result.matches == []


def test_relative_symlink_escape_is_rejected(tmp_path: Path) -> None:
    """相对链接（`../../outside/secret.py`）：resolve 基于链接所在目录，同样要挡住。"""
    root = _project(tmp_path)
    secret = _outside_secret(tmp_path)
    link = root / "pkg" / "rel_leak.py"
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(secret, link.parent)
    assert not os.path.isabs(relative_target)
    try:
        link.symlink_to(relative_target)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台无法创建文件符号链接")

    assert resolve_inside_project(root.resolve(), link) is None
    assert search_code(root, OUTSIDE_SECRET).matches == []


def test_absolute_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = _project(tmp_path)
    secret = _outside_secret(tmp_path)
    link = root / "pkg" / "abs_leak.py"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(secret.resolve())
    except (OSError, NotImplementedError):
        pytest.skip("当前平台无法创建文件符号链接")

    assert resolve_inside_project(root.resolve(), link) is None
    assert search_code(root, OUTSIDE_SECRET).matches == []


def test_read_loop_rejects_an_escaping_candidate(tmp_path: Path, monkeypatch) -> None:
    """**不依赖任何链接权限的看门人**：直接让遍历产出一个越界候选。

    这是本文件里唯一在任何平台都必定执行到读取循环的用例。它模拟的正是
    "候选清单里混进了项目外路径"这一件事——只要判据被去掉或写反，项目外的
    内容立刻会出现在搜索结果里。
    """
    root = _project(tmp_path)
    secret = _outside_secret(tmp_path)
    assert secret.is_file(), "目标文件必须真实存在，否则「没搜到」可能只是文件不存在"

    def fake_iter(root_path, *, ignored_dirs=None):  # noqa: ARG001 - 签名需与真实实现一致
        yield root_path / "pkg" / "client.py"  # 项目内：应当照常被读
        yield secret  # 项目外：必须被丢弃

    monkeypatch.setattr(code_search, "iter_python_files", fake_iter)

    result = search_code(root, OUTSIDE_SECRET)

    assert result.matches == [], "读取循环没有校验候选是否在项目内"
    assert OUTSIDE_SECRET not in _all_text(result)


def test_read_loop_still_reads_inside_candidates_when_patched(tmp_path: Path, monkeypatch) -> None:
    """上一条的反向护栏：patch 掉遍历之后，项目内的文件仍然必须被读到。

    没有这一条，"循环什么都不读"也能让上一条通过。
    """
    root = _project(tmp_path)
    secret = _outside_secret(tmp_path)

    def fake_iter(root_path, *, ignored_dirs=None):  # noqa: ARG001
        yield root_path / "pkg" / "client.py"
        yield secret

    monkeypatch.setattr(code_search, "iter_python_files", fake_iter)

    result = search_code(root, INSIDE_MARKER)

    assert [match.path for match in result.matches] == ["pkg/client.py"]
    assert result.files_scanned == 1


def test_search_code_and_read_file_refuse_the_same_escape(tmp_path: Path) -> None:
    """两个入口必须**同样**拒绝：search_code 与 read_file 共用一条边界语义。

    用目录联接构造逃逸（不需要任何特权），因此这条在任何平台都会真的跑。
    要求的是两边给出同一个结论：这个路径不属于项目。
    """
    root = _project(tmp_path)
    _outside_secret(tmp_path)
    _link_dir(tmp_path / "outside", root / "linked_outside")

    context = _ctx(root)
    read_result = ReadFileTool().invoke({"path": "linked_outside/secret.py"}, context)

    assert read_result.ok is False, "read_file 读到了项目外的文件"
    assert OUTSIDE_SECRET not in read_result.content
    assert read_result.error

    search_result = search_code(root, OUTSIDE_SECRET)

    assert search_result.matches == [], "search_code 与 read_file 的边界判断不一致"
    assert OUTSIDE_SECRET not in _all_text(search_result)


def test_parent_escape_is_refused_by_both_entry_points(tmp_path: Path) -> None:
    """`..` 逃逸同样要对齐：read_file 按形状拒绝，search_code 按解析结果拒绝。"""
    root = _project(tmp_path)
    _outside_secret(tmp_path)

    read_result = ReadFileTool().invoke({"path": "../outside/secret.py"}, _ctx(root))

    assert read_result.ok is False
    assert OUTSIDE_SECRET not in read_result.content
    assert resolve_inside_project(root.resolve(), root / ".." / "outside" / "secret.py") is None


def test_every_reported_match_resolves_inside_the_root(tmp_path: Path) -> None:
    """不变量：搜索结果里出现的每一个路径，解析之后都必须落在 root 内。"""
    root = _project(tmp_path)
    secret = _outside_secret(tmp_path)
    for name in ("leak_a.py", "leak_b.py"):
        try:
            (root / "pkg" / name).symlink_to(secret)
        except (OSError, NotImplementedError):
            break

    root_resolved = root.resolve()
    result = search_code(root, "=")

    for match in result.matches:
        resolved = resolve_inside_project(root_resolved, root_resolved / match.path)
        assert resolved is not None, f"搜索结果里出现了项目外的路径：{match.path}"


# ==================================================== 路径写法（Windows/POSIX）


@pytest.mark.parametrize(
    "relative",
    [
        "pkg/client.py",
        "pkg\\client.py",  # Windows 风格分隔符
        "./pkg/client.py",
        "pkg/./client.py",
        "pkg/sub/../client.py",
    ],
)
def test_portable_forms_of_an_inside_path_resolve(tmp_path: Path, relative: str) -> None:
    """两套分隔符都要认：判据必须同时处理 POSIX 与 Windows 写法。"""
    root = _project(tmp_path).resolve()
    candidate = str(root) + os.sep + relative

    resolved = resolve_inside_project(root, candidate)

    assert resolved is not None, candidate


@pytest.mark.parametrize(
    "candidate",
    [
        "C:/Windows/System32/drivers/etc/hosts",
        "C:\\Windows\\System32",
        "//server/share/secret.py",
        "\\\\server\\share\\secret.py",
        "/etc/passwd",
        "\\Windows\\win.ini",
    ],
)
def test_absolute_forms_are_never_inside(tmp_path: Path, candidate: str) -> None:
    root = _project(tmp_path).resolve()

    assert resolve_inside_project(root, candidate) is None, candidate


# ============================================================ 模糊测试（性质）


_FUZZ_PIECES = (
    "",
    ".",
    "..",
    "...",
    "pkg",
    "pkg/..",
    "../..",
    "..\\",
    "/",
    "\\",
    "//",
    "C:",
    "C:\\",
    "C:/",
    "//server/share",
    "\\\\server\\share",
    "pkg/sub/../..",
    "....//",
    "pkg/./client.py",
    "x" * 200,
    "pkg/client.py",
    "\u2028",
)
_FUZZ_SEPARATORS = ("/", "\\", "")


def _fuzz_candidates() -> list[str]:
    return [
        f"{first}{separator}{second}"
        for first in _FUZZ_PIECES
        for second in _FUZZ_PIECES
        for separator in _FUZZ_SEPARATORS
    ]


def test_fuzz_candidates_never_resolve_outside_the_root(tmp_path: Path) -> None:
    """把几千个畸形路径灌进判据，任何一个解析到 root 之外都算失败。

    "26 万 fuzz 输入零逃逸"是这条修复必须保住的既有性质。这里用较小的组合规模
    （22 × 22 × 3 = 1452 条）覆盖同一类形状——分隔符混用、`..` 层级、盘符、
    UNC、超长段、绝对路径——因为它要能在单元测试里跑完；规模不同，判据相同。
    """
    root = _project(tmp_path).resolve()
    secret = _outside_secret(tmp_path).resolve()

    candidates = _fuzz_candidates()
    assert len(candidates) > 1000

    escaped: list[str] = []
    for candidate in candidates:
        try:
            resolved = resolve_inside_project(root, candidate)
        except (ValueError, OSError) as exc:  # 判据本身不允许把异常抛给调用方
            escaped.append(f"{candidate!r} 抛出 {type(exc).__name__}: {exc}")
            continue
        if resolved is None:
            continue
        if not resolved.is_relative_to(root) or resolved == secret:
            escaped.append(f"{candidate!r} -> {resolved}")

    assert escaped == [], escaped[:10]
