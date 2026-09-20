"""三个只读工具的行为与安全边界测试（E2 Commit 2）。

覆盖要求场景：
    1. inspect_project 正常扫描 / 相对路径 / 不泄漏绝对路径
    2. inspect_project 输出长度限制 / project_root 不存在
    3. inspect_project 不读取文件内容 / 不信任可能过期的 ctx.project
    4. search_code 正常搜索 / 大小写 / 空 query / 无匹配
    5. search_code 结果数量限制 / 输出长度限制 / 上限不可被放宽
    6. search_code 只搜项目目录内 / 非法输入与异常处理
    7. read_file 正常读取 / 相对路径 / chars 正确
    8. read_file 拒绝绝对路径 / Windows 盘符 / 盘符相对路径 / ~
    9. read_file 拒绝 .. 逃逸 / symlink（Windows 上退化为 junction）逃逸
    10. read_file 文件不存在 / 目录 / 二进制 / 凭据文件
    11. read_file 内容截断 / truncated 标记
    12. 错误信息不含 traceback、不含绝对服务器路径
    13. 三个工具都能从 ToolRegistry 取到，spec 正确且 read_only=True
    14. 工具不修改项目（只读验证）

全部在 pytest 的 tmp_path 里造真实目录，不触碰仓库本身、不调用 DeepSeek、不发网络请求。
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from integration_agent.api import APIInfo
from integration_agent.repository import ProjectStructure, scan_repository
from integration_agent.tools import (
    InspectProjectTool,
    ReadFileTool,
    SearchCodeTool,
    ToolContext,
    ToolRegistry,
    build_default_registry,
)
from integration_agent.tools.repository_tools import (
    MAX_INSPECT_CHARS,
    MAX_READ_CHARS,
    MAX_SEARCH_CHARS,
    MAX_SEARCH_RESULTS,
)

SECRET_SENTINEL = "SENTINEL_CONTENT_MUST_NOT_LEAK"
OUTSIDE_SECRET = "OUTSIDE_ROOT_SECRET"
# 默认注册表里应当有的工具。E2 Commit 3 加入了 inspect_api，这里跟着变成 4 个：
# 这份清单是"默认集合"的断言，不是"本模块实现的工具"的清单。
DEFAULT_TOOL_NAMES = ("inspect_api", "inspect_project", "read_file", "search_code")

# 安全性上限的字面期望值。其余用例引用模块常量即可读性更好，这一组则刻意写死：
# 如果只从实现里读上限，把 MAX_READ_CHARS 放宽到一千万这类改动会让所有断言
# 静默跟着放宽——测试就失去了看门的作用。改动这些数字必须是有意识的决定。
EXPECTED_LIMITS = {
    "MAX_INSPECT_CHARS": 4000,
    "MAX_SEARCH_CHARS": 6000,
    "MAX_SEARCH_RESULTS": 20,
    "MAX_READ_CHARS": 8000,
}
ACTUAL_LIMITS = {
    "MAX_INSPECT_CHARS": MAX_INSPECT_CHARS,
    "MAX_SEARCH_CHARS": MAX_SEARCH_CHARS,
    "MAX_SEARCH_RESULTS": MAX_SEARCH_RESULTS,
    "MAX_READ_CHARS": MAX_READ_CHARS,
}


# ------------------------------------------------------------------ helpers


def _project_tree(root: Path) -> None:
    """造一个最小但完整的 Python 项目。"""
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sample-project"\nversion = "0.1.0"\n'
        'dependencies = ["httpx>=0.27", "pydantic>=2.0"]\n',
        encoding="utf-8",
    )
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "client.py").write_text(
        'def fetch_user_profile(user_id):\n    """Fetch a user profile."""\n    return user_id\n',
        encoding="utf-8",
    )
    (root / "pkg" / "helpers.py").write_text(
        "def fetch_user_orders(user_id):\n    return []\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_client.py").write_text(
        "def test_fetch_user_profile():\n    assert True\n",
        encoding="utf-8",
    )


def _ctx(root: Path) -> ToolContext:
    """构造 ToolContext。

    ctx.project 只用一份与磁盘无关的占位快照。工具若偷懒直接读 ctx.project
    而不重新扫描，test_inspect_project_rescans_* 会立刻暴露出来。
    """
    project = (
        scan_repository(root)
        if root.is_dir()
        else ProjectStructure(root=str(root), name="placeholder", is_python_project=False)
    )
    return ToolContext(
        project_root=root,
        project=project,
        api=APIInfo(name="Fake API", version="1.0.0", base_url="https://api.example.com"),
        known_files=set(),
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    """记录目录下所有文件的内容，用于验证工具没有写操作。"""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_sanitized(text: str) -> None:
    """任何回给模型或客户端的文本都不得携带堆栈或本机路径。"""
    assert "Traceback" not in text
    assert 'File "' not in text
    assert "repository_tools.py" not in text


def _assert_refused(result, *, leak: str | None = None) -> None:
    """断言这是一次**明确的拒绝**，而不是撞上兜底 catch-all 的意外失败。

    只断言 ok is False 会给出虚假的安心，这一点是实测踩出来的：
    把 resolve() 之后那道包含性检查摘掉，越界路径会在随后的 relative_to()
    上抛 ValueError，被工具边界兜成"工具内部错误"——同样是 ok=False，
    只测 ok 的用例照样全绿，安全围栏没了也看不出来。
    所以这里必须把"有意拒绝"和"意外崩溃"区分开。
    """
    assert result.ok is False
    assert result.content == ""
    assert result.error
    assert "工具内部错误" not in result.error, f"安全围栏疑似失效，落到了兜底分支：{result.error}"
    _assert_sanitized(result.error)
    if leak is not None:
        assert leak not in result.content
        assert leak not in result.error


def _make_escape_link(target: Path, link: Path) -> str:
    """在 root 内造一个指向 root 之外的链接，返回实际使用的方式。

    优先 os.symlink；Windows 上创建符号链接需要管理员权限或开发者模式，
    失败时退回目录联接（junction）——联接不需要特权，且 Path.resolve() 同样会
    跟随它，因此仍能真正走到"resolve 之后发现越界"那条分支。
    两者都不可用时 skip，而不是假装通过。
    """
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


# ============================================== 场景 1-3：inspect_project


def test_safety_limits_are_pinned_to_known_values() -> None:
    """上限是安全边界的一部分，放松它必须是一次显式修改。"""
    assert ACTUAL_LIMITS == EXPECTED_LIMITS
    assert MAX_READ_CHARS == 8000


def test_inspect_project_summarizes_project(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = InspectProjectTool().invoke({}, _ctx(tmp_path))

    assert result.ok is True
    assert result.tool == "inspect_project"
    assert "sample-project" in result.content
    assert "pyproject.toml" in result.content
    assert "pkg/client.py" in result.content
    assert "httpx>=0.27" in result.content
    assert result.truncated is False
    assert result.error is None
    assert result.chars == len(result.content)


def test_inspect_project_rescans_instead_of_trusting_context(tmp_path: Path) -> None:
    """工具必须在调用这一刻看磁盘，而不是回放上层可能过期的快照。"""
    _project_tree(tmp_path)
    ctx = _ctx(tmp_path)
    ctx.project = ProjectStructure(
        root=str(tmp_path), name="stale-snapshot", is_python_project=False
    )

    result = InspectProjectTool().invoke({}, ctx)

    assert result.ok is True
    assert "sample-project" in result.content
    assert "stale-snapshot" not in result.content


def test_inspect_project_lists_relative_paths_only(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = InspectProjectTool().invoke({}, _ctx(tmp_path))

    assert "pkg/client.py" in result.content
    assert str(tmp_path) not in result.content
    assert str(tmp_path).replace("\\", "/") not in result.content


def test_inspect_project_does_not_leak_absolute_paths(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = InspectProjectTool().invoke({}, _ctx(tmp_path))

    for leaked in (str(tmp_path), str(tmp_path.resolve())):
        assert leaked not in result.content
    # 输出里不应出现任何以盘符开头的路径
    assert ":\\" not in result.content
    assert ":/" not in result.content


def test_inspect_project_truncates_large_listing(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    for index in range(200):
        name = f"pkg/module_with_a_deliberately_long_name_{index:03d}.py"
        (tmp_path / name).write_text("# filler\n", encoding="utf-8")

    result = InspectProjectTool().invoke({}, _ctx(tmp_path))

    assert result.ok is True
    assert len(result.content) <= MAX_INSPECT_CHARS
    assert result.truncated is True
    assert "已省略" in result.content


def test_inspect_project_missing_root_returns_structured_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    result = InspectProjectTool().invoke({}, _ctx(missing))

    _assert_refused(result)
    # 报错不得回吐服务器上的绝对路径
    assert str(missing) not in result.error
    assert str(tmp_path) not in result.error


def test_inspect_project_does_not_read_file_contents(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    (tmp_path / "pkg" / "secret_holder.py").write_text(
        f'TOKEN = "{SECRET_SENTINEL}"\n', encoding="utf-8"
    )

    result = InspectProjectTool().invoke({}, _ctx(tmp_path))

    assert result.ok is True
    assert SECRET_SENTINEL not in result.content
    assert "secret_holder.py" in result.content  # 只列名字，不读内容


# ================================================ 场景 4-6：search_code


def test_search_code_finds_matches(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": "fetch_user_profile"}, _ctx(tmp_path))

    assert result.ok is True
    assert "pkg/client.py" in result.content
    assert result.chars == len(result.content)


def test_search_code_is_case_insensitive_by_default(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": "FETCH_USER_PROFILE"}, _ctx(tmp_path))

    assert result.ok is True
    assert "pkg/client.py" in result.content


def test_search_code_honours_case_sensitive(tmp_path: Path) -> None:
    """区分大小写时，大写关键词应当搜不到小写的定义。"""
    _project_tree(tmp_path)
    uppercase = SearchCodeTool().invoke(
        {"query": "FETCH_USER_PROFILE", "case_sensitive": True}, _ctx(tmp_path)
    )
    lowercase = SearchCodeTool().invoke(
        {"query": "fetch_user_profile", "case_sensitive": True}, _ctx(tmp_path)
    )

    assert uppercase.ok is True
    assert "未找到匹配" in uppercase.content
    assert lowercase.ok is True
    assert "pkg/client.py" in lowercase.content


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_search_code_rejects_empty_query(tmp_path: Path, query: str) -> None:
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": query}, _ctx(tmp_path))

    assert result.ok is False
    assert result.error
    _assert_sanitized(result.error)


@pytest.mark.parametrize("query", [None, 1, ["fetch"], {"q": "fetch"}])
def test_search_code_rejects_non_string_query(tmp_path: Path, query) -> None:
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": query}, _ctx(tmp_path))

    assert result.ok is False
    assert "query" in result.error


def test_search_code_rejects_overlong_query(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": "x" * 5000}, _ctx(tmp_path))

    assert result.ok is False
    assert "过长" in result.error


def test_search_code_returns_ok_when_nothing_matches(tmp_path: Path) -> None:
    """搜不到不是错误：ok=True + 空结果，让模型自己决定换关键词。"""
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": "no_such_symbol_anywhere"}, _ctx(tmp_path))

    assert result.ok is True
    assert "未找到匹配" in result.content
    assert result.truncated is False


def _many_matching_files(root: Path, count: int = 60) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / f"pkg/mod_{index:02d}.py").write_text(
            "def needle_symbol():\n    return 1\n", encoding="utf-8"
        )


def test_search_code_marks_truncated_when_matches_exceed_cap(tmp_path: Path) -> None:
    _many_matching_files(tmp_path)
    result = SearchCodeTool().invoke({"query": "needle_symbol"}, _ctx(tmp_path))

    assert result.ok is True
    assert result.truncated is True


def test_search_code_respects_max_results(tmp_path: Path) -> None:
    _many_matching_files(tmp_path)
    narrowed = SearchCodeTool().invoke({"query": "needle_symbol", "max_results": 3}, _ctx(tmp_path))

    assert narrowed.ok is True
    # 每条命中带 path + 行号 + 2 行上下文，3 条命中不会覆盖到 4 个不同文件
    assert (
        len({line.split(":")[0] for line in narrowed.content.splitlines() if ".py:" in line}) <= 3
    )


def test_search_code_cannot_raise_the_result_cap(tmp_path: Path) -> None:
    """模型可以收紧上限，但不能突破硬上限。"""
    _many_matching_files(tmp_path)
    result = SearchCodeTool().invoke(
        {"query": "needle_symbol", "max_results": 10_000}, _ctx(tmp_path)
    )

    assert result.ok is True
    matched_files = {line.split(":")[0] for line in result.content.splitlines() if ".py:" in line}
    assert len(matched_files) <= MAX_SEARCH_RESULTS


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True])
def test_search_code_rejects_invalid_max_results(tmp_path: Path, bad) -> None:
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": "fetch", "max_results": bad}, _ctx(tmp_path))

    assert result.ok is False
    assert "max_results" in result.error


def test_search_code_rejects_invalid_case_sensitive(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = SearchCodeTool().invoke({"query": "fetch", "case_sensitive": "yes"}, _ctx(tmp_path))

    assert result.ok is False
    assert "case_sensitive" in result.error


def test_search_code_caps_total_output(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    for index in range(40):
        (tmp_path / f"pkg/mod_{index:02d}.py").write_text(
            ("needle_symbol = 1  # " + "padding " * 20 + "\n") * 3, encoding="utf-8"
        )

    result = SearchCodeTool().invoke({"query": "needle_symbol"}, _ctx(tmp_path))

    assert result.ok is True
    assert len(result.content) <= MAX_SEARCH_CHARS
    assert result.truncated is True


def test_search_code_results_stay_inside_the_project(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "inside.py").write_text("marker_symbol = 1\n", encoding="utf-8")
    (outside / "sibling.py").write_text("marker_symbol = 1\n", encoding="utf-8")

    result = SearchCodeTool().invoke({"query": "marker_symbol"}, _ctx(root))

    assert result.ok is True
    assert "inside.py" in result.content
    assert "sibling.py" not in result.content
    assert "outside" not in result.content


def test_search_code_missing_root_returns_structured_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    result = SearchCodeTool().invoke({"query": "anything"}, _ctx(missing))

    _assert_refused(result)
    assert str(missing) not in result.error


# ================================================== 场景 7-12：read_file


def test_read_file_reads_relative_path(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": "pkg/client.py"}, _ctx(tmp_path))

    assert result.ok is True
    assert "def fetch_user_profile" in result.content
    assert result.truncated is False
    assert result.error is None


def test_read_file_accepts_nested_and_forward_slash_paths(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    tool = ReadFileTool()

    assert tool.invoke({"path": "pkg/client.py"}, _ctx(tmp_path)).ok is True
    assert tool.invoke({"path": "./pkg/client.py"}, _ctx(tmp_path)).ok is True
    assert tool.invoke({"path": "pkg\\client.py"}, _ctx(tmp_path)).ok is True


def test_read_file_sets_chars_to_content_length(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": "pkg/client.py"}, _ctx(tmp_path))

    assert result.chars == len(result.content)
    assert result.chars > 0


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/",
        "//server/share/file.txt",
        "C:\\Windows\\win.ini",
        "C:/Windows/win.ini",
        "c:\\windows\\win.ini",
    ],
)
def test_read_file_rejects_absolute_paths(tmp_path: Path, path: str) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": path}, _ctx(tmp_path))

    assert result.ok is False
    assert result.content == ""
    assert "相对路径" in result.error
    _assert_sanitized(result.error)


@pytest.mark.parametrize("path", ["C:foo.txt", "C:foo/bar.txt", "D:secret.py"])
def test_read_file_rejects_drive_relative_paths(tmp_path: Path, path: str) -> None:
    """盘符相对路径不满足 is_absolute()，但同样能跳出当前目录。"""
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": path}, _ctx(tmp_path))

    assert result.ok is False
    assert "盘符" in result.error


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", "../../outside.txt", "pkg/../../outside.txt", "pkg/..", "..", "..\\out.txt"],
)
def test_read_file_rejects_parent_escape(tmp_path: Path, path: str) -> None:
    _project_tree(tmp_path)
    (tmp_path.parent / "outside.txt").write_text(OUTSIDE_SECRET, encoding="utf-8")

    result = ReadFileTool().invoke({"path": path}, _ctx(tmp_path))

    _assert_refused(result, leak=OUTSIDE_SECRET)


def test_read_file_rejects_tilde_paths(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": "~/secrets.txt"}, _ctx(tmp_path))

    _assert_refused(result)


def test_read_file_rejects_symlink_escape(tmp_path: Path) -> None:
    """符号链接（Windows 上退化为目录联接）指向 root 之外时必须被挡住。"""
    root = tmp_path / "proj"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text(OUTSIDE_SECRET, encoding="utf-8")

    link = root / "escape"
    _make_escape_link(outside, link)
    try:
        result = ReadFileTool().invoke({"path": "escape/secret.txt"}, _ctx(root))

        _assert_refused(result, leak=OUTSIDE_SECRET)
        assert str(outside) not in result.error
    finally:
        # 只摘掉链接本身，不要跟着它把目标目录也删了
        try:
            os.rmdir(link)
        except OSError:
            shutil.rmtree(link, ignore_errors=True)


def test_read_file_rejects_symlink_escaping_directory(tmp_path: Path) -> None:
    """整条路径都落在链接后面时，同样不能读到 root 外的目录内容。"""
    root = tmp_path / "proj"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "config.py").write_text(f'TOKEN = "{OUTSIDE_SECRET}"\n', encoding="utf-8")

    link = root / "escape"
    _make_escape_link(outside, link)
    try:
        result = ReadFileTool().invoke({"path": "escape/config.py"}, _ctx(root))
        _assert_refused(result, leak=OUTSIDE_SECRET)
    finally:
        try:
            os.rmdir(link)
        except OSError:
            shutil.rmtree(link, ignore_errors=True)


def test_read_file_missing_file_returns_structured_error(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": "pkg/not_here.py"}, _ctx(tmp_path))

    assert result.ok is False
    assert "不存在" in result.error
    assert "pkg/not_here.py" in result.error  # 回显的是模型自己给的相对路径
    _assert_sanitized(result.error)
    assert str(tmp_path) not in result.error


def test_read_file_rejects_directory(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": "pkg"}, _ctx(tmp_path))

    assert result.ok is False
    assert result.content == ""
    assert "目录" in result.error


@pytest.mark.parametrize("path", [".env", "pkg/.env", ".env.local", "secrets.pem", "id_rsa"])
def test_read_file_rejects_credential_files(tmp_path: Path, path: str) -> None:
    """凭据文件一律拒读，即使它确实存在且位于项目内。"""
    _project_tree(tmp_path)
    (tmp_path / path).write_text("API_KEY=sk-should-never-be-read\n", encoding="utf-8")

    result = ReadFileTool().invoke({"path": path}, _ctx(tmp_path))

    assert result.ok is False
    assert result.content == ""
    assert "凭据" in result.error
    assert "sk-should-never-be-read" not in result.content


def test_read_file_rejects_binary_file(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    (tmp_path / "pkg" / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00data")

    result = ReadFileTool().invoke({"path": "pkg/blob.bin"}, _ctx(tmp_path))

    assert result.ok is False
    assert "二进制" in result.error


def test_read_file_truncates_long_content(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    (tmp_path / "pkg" / "big.py").write_text("x" * (MAX_READ_CHARS * 3), encoding="utf-8")

    result = ReadFileTool().invoke({"path": "pkg/big.py"}, _ctx(tmp_path))

    assert result.ok is True
    assert result.truncated is True
    assert len(result.content) == MAX_READ_CHARS
    assert result.chars == len(result.content)


def test_read_file_does_not_truncate_short_content(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": "pkg/helpers.py"}, _ctx(tmp_path))

    assert result.truncated is False
    assert len(result.content) < MAX_READ_CHARS


def test_read_file_handles_undecodable_bytes(tmp_path: Path) -> None:
    """非 UTF-8 但非二进制的字节用替换字符降级，不能让工具崩掉。"""
    _project_tree(tmp_path)
    (tmp_path / "pkg" / "latin.py").write_bytes(b"# caf\xe9 comment\nvalue = 1\n")

    result = ReadFileTool().invoke({"path": "pkg/latin.py"}, _ctx(tmp_path))

    assert result.ok is True
    assert "value = 1" in result.content


@pytest.mark.parametrize("path", ["", "   ", None, 42])
def test_read_file_rejects_invalid_path_argument(tmp_path: Path, path) -> None:
    _project_tree(tmp_path)
    result = ReadFileTool().invoke({"path": path}, _ctx(tmp_path))

    assert result.ok is False
    assert result.error
    _assert_sanitized(result.error)


def test_read_file_missing_root_returns_structured_error(tmp_path: Path) -> None:
    missing = tmp_path / "gone"
    result = ReadFileTool().invoke({"path": "pkg/client.py"}, _ctx(missing))

    _assert_refused(result)
    assert str(missing) not in result.error


def test_unexpected_failure_is_sanitized_but_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """工具边界必须兜住意外异常：堆栈进服务端日志，模型只看到一句话。

    这是 AgentTool 契约（任何失败都返回 ok=False）与项目既有约定的交叉点：
    异常原文带着本机绝对路径与源码行，绝不能进 LLM 上下文。
    """
    _project_tree(tmp_path)
    original_open = Path.open
    leaky = "boom at C:\\repo\\src\\integration_agent\\tools\\repository_tools.py"

    def exploding_open(self, *args, **kwargs):
        if self.name == "client.py":
            raise ValueError(leaky)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", exploding_open)

    with caplog.at_level(logging.ERROR, logger="integration_agent.tools.repository_tools"):
        result = ReadFileTool().invoke({"path": "pkg/client.py"}, _ctx(tmp_path))

    # 回给模型的是脱敏结果
    assert result.ok is False
    assert result.content == ""
    assert "boom at" not in result.error
    _assert_sanitized(result.error)
    # 完整堆栈留在了服务端日志里，没有被丢掉
    assert "boom at" in caplog.text


# ========================================== 场景 13-14：注册表与只读性


def test_default_registry_exposes_every_default_tool() -> None:
    registry = build_default_registry()

    assert [spec.name for spec in registry.specs()] == sorted(DEFAULT_TOOL_NAMES)
    assert len(registry.specs()) == 4
    for name in DEFAULT_TOOL_NAMES:
        assert registry.get(name) is not None
    assert registry.get("no_such_tool") is None


def test_default_registry_specs_are_correct() -> None:
    specs = {spec.name: spec for spec in build_default_registry().specs()}

    assert specs["inspect_project"].parameters == {"type": "object", "properties": {}}
    assert specs["search_code"].parameters["required"] == ["query"]
    assert specs["search_code"].parameters["properties"]["query"]["type"] == "string"
    assert specs["read_file"].parameters["required"] == ["path"]
    assert specs["read_file"].parameters["properties"]["path"]["type"] == "string"
    for spec in specs.values():
        assert spec.description.strip()
        assert spec.parameters.get("type") == "object"


def test_default_registry_tools_are_read_only() -> None:
    for spec in build_default_registry().specs():
        assert spec.read_only is True, f"{spec.name} 必须是只读工具"


def test_default_registry_describe_is_stable() -> None:
    first = build_default_registry()
    second = build_default_registry()

    assert first.describe() == second.describe()
    assert first.describe() == first.describe()
    assert all(name in first.describe() for name in DEFAULT_TOOL_NAMES)


def test_registry_order_is_independent_of_registration_order() -> None:
    forward = build_default_registry()
    tools = [forward.get(name) for name in DEFAULT_TOOL_NAMES]
    assert all(tool is not None for tool in tools)

    reversed_registry = ToolRegistry(list(reversed(tools)))

    assert [spec.name for spec in reversed_registry.specs()] == [
        spec.name for spec in forward.specs()
    ]
    assert reversed_registry.describe() == forward.describe()


def test_tools_do_not_modify_the_project(tmp_path: Path) -> None:
    """三个工具跑一遍之后，项目内容必须逐字节不变。"""
    _project_tree(tmp_path)
    before = _snapshot(tmp_path)
    ctx = _ctx(tmp_path)
    registry = build_default_registry()

    for name, args in (
        ("inspect_project", {}),
        ("search_code", {"query": "fetch_user"}),
        ("read_file", {"path": "pkg/client.py"}),
    ):
        tool = registry.get(name)
        assert tool is not None
        result = tool.invoke(args, ctx)
        assert result.ok is True, f"{name} 意外失败：{result.error}"

    assert _snapshot(tmp_path) == before


def test_tools_do_not_create_new_files(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    ctx = _ctx(tmp_path)
    InspectProjectTool().invoke({}, ctx)
    SearchCodeTool().invoke({"query": "fetch"}, ctx)
    ReadFileTool().invoke({"path": "pyproject.toml"}, ctx)

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before
