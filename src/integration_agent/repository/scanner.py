"""Repository Scanner：扫描已有 Python 项目的结构，为 Integration Planner 提供上下文。

典型用法：
    from integration_agent.repository import scan_repository

    project = scan_repository("examples/demo_project")
    print(project.name, project.is_python_project)
    print(project.manifest_files, project.source_dirs, project.test_dirs)
    print(project.dependencies, project.python_files)

设计原则：
    - 只做文件系统层面的扫描：不解析 AST、不引入向量数据库。
    - 输出结构化对象（相对 POSIX 路径、固定忽略规则），便于按需裁剪后再交给 LLM。
"""

import os
from collections.abc import Collection, Iterator
from pathlib import Path

from pydantic import BaseModel, Field

try:  # Python 3.11+ 自带 tomllib；3.10 下退化为不解析 pyproject.toml
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 仅 Python 3.10
    tomllib = None  # type: ignore[assignment]

# 扫描时跳过的目录名（版本控制、虚拟环境、缓存、构建产物等）
DEFAULT_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        "node_modules",
        "build",
        "dist",
        "htmlcov",
        ".eggs",
        ".idea",
        ".vscode",
    }
)

# 用于识别 Python 项目的清单文件（按优先级排序）
MANIFEST_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
)

# 常见的源码目录与测试目录候选名
SOURCE_DIR_CANDIDATES = ("src", "lib")
TEST_DIR_CANDIDATES = ("tests", "test")


class RepositoryError(ValueError):
    """仓库扫描或代码搜索失败时抛出。"""


class ProjectStructure(BaseModel):
    """一个已有 Python 项目的结构快照。所有路径均为相对 root 的 POSIX 风格字符串。"""

    root: str  # 项目根目录的绝对路径
    name: str  # 项目名（取自 pyproject.toml，缺失时回退为目录名）
    is_python_project: bool  # 存在 Python 清单文件或有 .py 文件
    manifest_files: list[str] = Field(default_factory=list)  # pyproject.toml 等
    source_dirs: list[str] = Field(default_factory=list)  # src、含 __init__.py 的包目录
    test_dirs: list[str] = Field(default_factory=list)  # tests / test
    dependencies: list[str] = Field(default_factory=list)  # pyproject + requirements 声明的依赖
    python_files: list[str] = Field(default_factory=list)  # 全部 .py 文件（已忽略无关目录）


def scan_repository(
    root: str | Path, *, ignored_dirs: Collection[str] = DEFAULT_IGNORED_DIRS
) -> ProjectStructure:
    """扫描项目目录并返回结构快照；目录不存在或不是目录时抛出 RepositoryError。"""
    path = validate_project_root(root)
    python_files = [
        file.relative_to(path).as_posix()
        for file in iter_python_files(path, ignored_dirs=ignored_dirs)
    ]
    manifest_files = [name for name in MANIFEST_FILES if (path / name).is_file()]

    name, dependencies = _read_pyproject(path / "pyproject.toml")
    dependencies = _merge_dependencies(dependencies, _read_requirements(path / "requirements.txt"))

    return ProjectStructure(
        root=str(path),
        name=name or path.name,
        is_python_project=bool(manifest_files or python_files),
        manifest_files=manifest_files,
        source_dirs=_detect_source_dirs(path, ignored_dirs=ignored_dirs),
        test_dirs=[name for name in TEST_DIR_CANDIDATES if (path / name).is_dir()],
        dependencies=dependencies,
        python_files=python_files,
    )


def iter_python_files(
    root: str | Path, *, ignored_dirs: Collection[str] = DEFAULT_IGNORED_DIRS
) -> Iterator[Path]:
    """按稳定顺序遍历项目中的 .py 文件（绝对路径），跳过忽略目录与 *.egg-info。

    ``followlinks=False`` 只挡住"顺着 symlink **目录**往下走"，**挡不住 symlink
    文件**：名字以 .py 结尾的链接照样会被列出来。所以这个函数只负责"列出候选"，
    凡是**要读内容**的调用方都必须再走一次 resolve_inside_project——
    ``link.stat()`` / ``link.read_text()`` 都会跟随链接，读到 root 外面去。
    """
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in ignored_dirs and not name.endswith(".egg-info")
        )
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def resolve_inside_project(root: str | Path, candidate: str | Path) -> Path | None:
    """候选路径 → 项目内的真实路径；解析后落在 root 之外时返回 None。

    这是"读取项目内文件"的统一边界判据，与 read_file 工具用的是同一条：
    **先 resolve，再判包含关系**。顺序不能反——symlink / junction 只有在 resolve
    之后才现出原形，直接比较链接自身的路径永远得到"在项目内"。

    root 必须已经是真实路径（``validate_project_root`` 保证这一点），
    否则两边一个 resolve 过一个没 resolve，比较没有意义。

    返回 None 而不是抛异常：调用方的语义是"这个候选不可用，跳过"，
    与"文件不可读"是同一种处理。断链、权限不足、链接成环等 OSError 同样返回 None。
    """
    try:
        resolved = Path(candidate).resolve()
    except OSError:  # 断链 / 权限不足 / 链接成环
        return None
    return resolved if resolved.is_relative_to(Path(root)) else None


def validate_project_root(root: str | Path) -> Path:
    """校验并规范化项目根目录；无效时抛出 RepositoryError。"""
    path = Path(root)
    if not path.exists():
        raise RepositoryError(f"项目目录不存在：{path}")
    if not path.is_dir():
        raise RepositoryError(f"项目路径不是目录：{path}")
    return path.resolve()


def _detect_source_dirs(root: Path, *, ignored_dirs: Collection[str]) -> list[str]:
    """识别源码目录：src/lib 等约定名，以及根目录下含 __init__.py 的包目录。"""
    found = {name for name in SOURCE_DIR_CANDIDATES if (root / name).is_dir()}
    try:
        children = sorted(root.iterdir())
    except OSError:  # pragma: no cover - 目录不可读时降级为仅按约定名判断
        children = []
    for child in children:
        if not child.is_dir() or child.name in ignored_dirs:
            continue
        if child.name in TEST_DIR_CANDIDATES:
            continue
        if (child / "__init__.py").is_file():
            found.add(child.name)
    return sorted(found)


def _read_pyproject(path: Path) -> tuple[str | None, list[str]]:
    """读取 pyproject.toml 的项目名与依赖；文件缺失或格式错误时返回 (None, [])。"""
    if tomllib is None or not path.is_file():
        return None, []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None, []
    project = data.get("project")
    if not isinstance(project, dict):
        return None, []
    name = project.get("name") if isinstance(project.get("name"), str) else None
    raw_deps = project.get("dependencies")
    dependencies = (
        [dep for dep in raw_deps if isinstance(dep, str)] if isinstance(raw_deps, list) else []
    )
    return name, dependencies


def _read_requirements(path: Path) -> list[str]:
    """读取 requirements.txt 中的依赖；跳过注释、空行与 -r/-e 之类的选项行。"""
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return [
        entry
        for entry in (line.strip() for line in lines)
        if entry and not entry.startswith(("#", "-"))
    ]


def _merge_dependencies(*groups: list[str]) -> list[str]:
    """合并多个依赖列表，去重并保持出现顺序。"""
    merged: list[str] = []
    for group in groups:
        for dependency in group:
            if dependency not in merged:
                merged.append(dependency)
    return merged
