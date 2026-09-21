"""Repository Scanner 阶段：扫描目标 Python 项目的结构、依赖与代码风格。

主要 API：
    scan_repository(path)  —— 扫描项目，返回 ProjectStructure 结构快照
    search_code(root, kw)  —— 按关键词搜索 .py 文件，返回受限的代码片段

结构化对象：
    ProjectStructure / CodeMatch / SearchResult
"""

from integration_agent.repository.code_search import CodeMatch, SearchResult, search_code
from integration_agent.repository.scanner import (
    DEFAULT_IGNORED_DIRS,
    MANIFEST_FILES,
    ProjectStructure,
    RepositoryError,
    iter_python_files,
    resolve_inside_project,
    scan_repository,
)

__all__ = [
    "DEFAULT_IGNORED_DIRS",
    "MANIFEST_FILES",
    "CodeMatch",
    "ProjectStructure",
    "RepositoryError",
    "SearchResult",
    "iter_python_files",
    "resolve_inside_project",
    "scan_repository",
    "search_code",
]
