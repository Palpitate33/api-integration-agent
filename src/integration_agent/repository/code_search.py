"""Code Search：在已有 Python 项目中按关键词搜索代码片段。

典型用法：
    from integration_agent.repository import search_code

    result = search_code("examples/demo_project", "get_user_profile")
    for match in result.matches:
        print(match.path, match.line_number)
        print(match.snippet)

设计原则：
    - 简单的行级关键词匹配（可选大小写敏感），不引入向量数据库或索引。
    - 三层长度限制（单片段 / 结果条数 / 总字符数），避免把整个仓库塞进 LLM 上下文。
    - 结果中的 truncated 标记提示调用方：应缩小关键词范围或放宽限制。
"""

from collections.abc import Collection
from pathlib import Path

from pydantic import BaseModel, Field

from integration_agent.repository.scanner import (
    DEFAULT_IGNORED_DIRS,
    RepositoryError,
    iter_python_files,
    validate_project_root,
)


class CodeMatch(BaseModel):
    """单个代码匹配。snippet 为带行号的上下文片段。"""

    path: str  # 相对项目根目录的 POSIX 路径
    line_number: int  # 命中行号（1 起）
    snippet: str  # 含上下文行的片段，每行前缀为 "行号: "


class SearchResult(BaseModel):
    """一次代码搜索的结果。"""

    query: str  # 原始关键词
    matches: list[CodeMatch] = Field(default_factory=list)
    files_scanned: int = 0  # 实际读取的 .py 文件数
    truncated: bool = False  # 是否因长度/条数限制提前停止


def search_code(
    root: str | Path,
    query: str,
    *,
    case_sensitive: bool = False,
    context_lines: int = 2,
    max_results: int = 20,
    max_snippet_chars: int = 400,
    max_total_chars: int = 6000,
    max_file_bytes: int = 2_000_000,
    ignored_dirs: Collection[str] = DEFAULT_IGNORED_DIRS,
) -> SearchResult:
    """在项目内的 .py 文件中按关键词搜索，返回受限的代码片段。

    Args:
        root: 项目根目录。
        query: 关键词，按子串匹配（非正则）。
        case_sensitive: 是否区分大小写，默认不区分。
        context_lines: 命中行上下各保留的上下文行数。
        max_results: 最多返回的匹配条数。
        max_snippet_chars: 单个片段的字符上限，超出则截断。
        max_total_chars: 所有片段的总字符上限，超出则停止收集。
        max_file_bytes: 超过该大小的文件跳过（通常是生成产物或数据文件）。
        ignored_dirs: 扫描时跳过的目录名。

    Raises:
        RepositoryError: 项目目录无效、关键词为空或参数非法。
    """
    root_path = validate_project_root(root)
    keyword = query.strip()
    if not keyword:
        raise RepositoryError("搜索关键词不能为空")
    if context_lines < 0:
        raise RepositoryError(f"context_lines 不能为负数：{context_lines}")
    if min(max_results, max_snippet_chars, max_total_chars) < 1:
        raise RepositoryError("max_results / max_snippet_chars / max_total_chars 必须为正数")

    needle = keyword if case_sensitive else keyword.casefold()
    matches: list[CodeMatch] = []
    files_scanned = 0
    total_chars = 0
    truncated = False
    stop = False

    for file_path in iter_python_files(root_path, ignored_dirs=ignored_dirs):
        if stop:
            break
        try:
            if file_path.stat().st_size > max_file_bytes:
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - 文件在扫描过程中消失或不可读
            continue
        files_scanned += 1
        relative_path = file_path.relative_to(root_path).as_posix()

        for index, line in enumerate(text.splitlines()):
            haystack = line if case_sensitive else line.casefold()
            if needle not in haystack:
                continue
            if len(matches) >= max_results:
                truncated = True
                stop = True
                break

            snippet = _build_snippet(text, index, context_lines)
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars]
                truncated = True
            if total_chars + len(snippet) > max_total_chars:
                truncated = True
                stop = True
                break

            matches.append(CodeMatch(path=relative_path, line_number=index + 1, snippet=snippet))
            total_chars += len(snippet)

    return SearchResult(
        query=keyword,
        matches=matches,
        files_scanned=files_scanned,
        truncated=truncated,
    )


def _build_snippet(text: str, index: int, context_lines: int) -> str:
    """构建以 index 行为中心、带行号前缀的片段。"""
    lines = text.splitlines()
    start = max(0, index - context_lines)
    end = min(len(lines), index + context_lines + 1)
    return "\n".join(f"{number + 1}: {lines[number].rstrip()}" for number in range(start, end))
