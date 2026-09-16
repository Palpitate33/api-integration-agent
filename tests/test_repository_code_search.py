"""Code Search 单元测试。"""

from pathlib import Path

import pytest

from integration_agent.repository import RepositoryError, search_code

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODULE_SOURCE = """import os


def fetch_user(user_id):
    \"\"\"Fetch a user.\"\"\"
    return user_id
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "client.py").write_text(MODULE_SOURCE, encoding="utf-8")
    (package / "other.py").write_text("def unrelated():\n    return 'needle-2'\n", encoding="utf-8")
    (package / "third.py").write_text("needle = 3\n", encoding="utf-8")
    (package / "notes.txt").write_text("fetch_user\n", encoding="utf-8")
    return tmp_path


def test_finds_match_with_path_and_line(project: Path) -> None:
    result = search_code(project, "fetch_user")
    assert result.query == "fetch_user"
    assert result.truncated is False
    match = result.matches[0]
    assert match.path == "pkg/client.py"
    assert match.line_number == 4
    assert "def fetch_user(user_id):" in match.snippet


def test_snippet_includes_context_and_line_numbers(project: Path) -> None:
    result = search_code(project, "fetch_user", context_lines=1)
    snippet = result.matches[0].snippet
    assert "3: " in snippet and "4: def fetch_user(user_id):" in snippet
    assert "5: " in snippet
    assert "1: import os" not in snippet  # 上下文之外的行不出现


def test_zero_context_lines(project: Path) -> None:
    result = search_code(project, "fetch_user", context_lines=0)
    assert result.matches[0].snippet == "4: def fetch_user(user_id):"


def test_case_insensitive_by_default(project: Path) -> None:
    assert search_code(project, "FETCH_USER").matches[0].path == "pkg/client.py"


def test_case_sensitive_opt_in(project: Path) -> None:
    assert search_code(project, "FETCH_USER", case_sensitive=True).matches == []
    assert search_code(project, "fetch_user", case_sensitive=True).matches


def test_only_python_files_are_searched(project: Path) -> None:
    result = search_code(project, "fetch_user")
    assert [match.path for match in result.matches] == ["pkg/client.py"]
    assert result.files_scanned == 3  # 只读取了 .py 文件


def test_ignored_dirs_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle = 1\n", encoding="utf-8")
    for ignored in (".venv", "__pycache__", ".git"):
        target = tmp_path / ignored
        target.mkdir()
        (target / "dep.py").write_text("needle = 2\n", encoding="utf-8")
    result = search_code(tmp_path, "needle")
    assert [match.path for match in result.matches] == ["app.py"]


def test_no_match_returns_empty_result(project: Path) -> None:
    result = search_code(project, "definitely_not_present")
    assert result.matches == []
    assert result.truncated is False


def test_max_results_truncates(project: Path) -> None:
    result = search_code(project, "needle", max_results=1)
    assert len(result.matches) == 1
    assert result.truncated is True


def test_max_snippet_chars_truncates(project: Path) -> None:
    result = search_code(project, "fetch_user", context_lines=0, max_snippet_chars=10)
    assert len(result.matches[0].snippet) == 10
    assert result.truncated is True


def test_max_total_chars_stops_collecting(project: Path) -> None:
    full = search_code(project, "needle", context_lines=0)
    assert [match.path for match in full.matches] == ["pkg/other.py", "pkg/third.py"]

    budget = len(full.matches[0].snippet)
    limited = search_code(project, "needle", context_lines=0, max_total_chars=budget)
    assert [match.path for match in limited.matches] == ["pkg/other.py"]
    assert limited.truncated is True


def test_max_total_chars_too_small_returns_nothing(project: Path) -> None:
    result = search_code(project, "fetch_user", max_total_chars=1)
    assert result.matches == []
    assert result.truncated is True


def test_large_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("needle = 1\n", encoding="utf-8")
    result = search_code(tmp_path, "needle", max_file_bytes=1)
    assert result.matches == []
    assert result.files_scanned == 0


def test_empty_query_raises(project: Path) -> None:
    with pytest.raises(RepositoryError, match="关键词不能为空"):
        search_code(project, "   ")


def test_invalid_arguments_raise(project: Path) -> None:
    with pytest.raises(RepositoryError, match="context_lines"):
        search_code(project, "needle", context_lines=-1)
    with pytest.raises(RepositoryError, match="必须为正数"):
        search_code(project, "needle", max_results=0)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(RepositoryError, match="项目目录不存在"):
        search_code(tmp_path / "nope", "needle")


def test_search_demo_project() -> None:
    result = search_code(PROJECT_ROOT / "examples" / "demo_project", "get_user_profile")
    match = result.matches[0]
    assert match.path == "demo_project/service.py"
    assert match.line_number == 4
    assert "def get_user_profile(user_id: str) -> dict:" in match.snippet


def test_search_own_project_limits_output() -> None:
    result = search_code(PROJECT_ROOT, "def ", max_results=3, max_total_chars=300)
    assert len(result.matches) <= 3
    total = sum(len(match.snippet) for match in result.matches)
    assert total <= 300
    assert all(match.path.endswith(".py") for match in result.matches)
