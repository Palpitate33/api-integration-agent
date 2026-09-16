"""Repository Scanner 单元测试。"""

from pathlib import Path

import pytest

from integration_agent.repository import (
    ProjectStructure,
    RepositoryError,
    iter_python_files,
    scan_repository,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """一个模拟的 src layout 项目，含虚拟环境、缓存与构建产物。"""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sample-app"\nversion = "0.2.0"\n'
        'dependencies = ["httpx>=0.27", "pydantic>=2.0"]\n',
        encoding="utf-8",
    )
    package = tmp_path / "src" / "sample_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "client.py").write_text("def fetch():\n    return None\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_client.py").write_text("def test_fetch():\n    assert True\n", encoding="utf-8")

    for ignored in (".venv/Lib", "__pycache__", ".git", "build"):
        target = tmp_path / ignored
        target.mkdir(parents=True)
        (target / "ignored.py").write_text("# noise\n", encoding="utf-8")
    return tmp_path


def test_python_files_exclude_ignored_dirs(project: Path) -> None:
    structure = scan_repository(project)
    assert sorted(structure.python_files) == [
        "src/sample_app/__init__.py",
        "src/sample_app/client.py",
        "tests/test_client.py",
    ]


def test_manifest_and_name_detected(project: Path) -> None:
    structure = scan_repository(project)
    assert structure.is_python_project is True
    assert structure.manifest_files == ["pyproject.toml"]
    assert structure.name == "sample-app"


def test_source_and_test_dirs_detected(project: Path) -> None:
    structure = scan_repository(project)
    assert structure.source_dirs == ["src"]
    assert structure.test_dirs == ["tests"]


def test_dependencies_from_pyproject(project: Path) -> None:
    structure = scan_repository(project)
    assert structure.dependencies == ["httpx>=0.27", "pydantic>=2.0"]


def test_root_is_absolute(project: Path) -> None:
    structure = scan_repository(project)
    assert Path(structure.root).is_absolute()
    assert Path(structure.root) == project.resolve()


def test_flat_layout_package_is_source_dir(tmp_path: Path) -> None:
    (tmp_path / "setup.py").write_text("", encoding="utf-8")
    package = tmp_path / "myapp"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text("", encoding="utf-8")
    (tmp_path / "test").mkdir()
    test_package = tmp_path / "test" / "__init__.py"
    test_package.write_text("", encoding="utf-8")

    structure = scan_repository(tmp_path)
    assert structure.source_dirs == ["myapp"]  # tests 目录不当作源码目录
    assert structure.test_dirs == ["test"]
    assert structure.manifest_files == ["setup.py"]
    assert structure.name == tmp_path.name  # 无 pyproject.toml 时回退为目录名


def test_requirements_txt_parsing(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(
        "# comment\nrequests>=2.31\n\n-r other.txt\nflask\nrequests>=2.31\n",
        encoding="utf-8",
    )
    structure = scan_repository(tmp_path)
    assert structure.manifest_files == ["requirements.txt"]
    assert structure.dependencies == ["requests>=2.31", "flask"]


def test_empty_directory_is_not_python_project(tmp_path: Path) -> None:
    structure = scan_repository(tmp_path)
    assert structure.is_python_project is False
    assert structure.python_files == []
    assert structure.manifest_files == []
    assert structure.dependencies == []
    assert structure.name == tmp_path.name


def test_invalid_pyproject_falls_back_to_dir_name(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("this is [not valid toml", encoding="utf-8")
    structure = scan_repository(tmp_path)
    assert structure.name == tmp_path.name
    assert structure.dependencies == []


def test_scan_is_deterministic(project: Path) -> None:
    assert scan_repository(project).python_files == scan_repository(project).python_files


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(RepositoryError, match="项目目录不存在"):
        scan_repository(tmp_path / "nope")


def test_file_path_raises(tmp_path: Path) -> None:
    target = tmp_path / "afile.py"
    target.write_text("", encoding="utf-8")
    with pytest.raises(RepositoryError, match="不是目录"):
        scan_repository(target)


def test_iter_python_files_yields_absolute_paths(project: Path) -> None:
    files = list(iter_python_files(project))
    assert files
    assert all(file.is_absolute() for file in files)
    assert all(file.suffix == ".py" for file in files)


def test_custom_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("", encoding="utf-8")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "lib.py").write_text("", encoding="utf-8")
    assert scan_repository(tmp_path).python_files == ["keep.py", "vendor/lib.py"]
    assert scan_repository(tmp_path, ignored_dirs={"vendor"}).python_files == ["keep.py"]


def test_scan_demo_project() -> None:
    structure = scan_repository(PROJECT_ROOT / "examples" / "demo_project")
    assert structure.is_python_project is True
    assert structure.name == "demo-project"
    assert structure.python_files == [
        "demo_project/__init__.py",
        "demo_project/service.py",
    ]


def test_scan_own_project_root() -> None:
    structure = scan_repository(PROJECT_ROOT)
    assert isinstance(structure, ProjectStructure)
    assert structure.is_python_project is True
    assert "pyproject.toml" in structure.manifest_files
    assert "pydantic>=2.0" in structure.dependencies
    assert "src/integration_agent/__init__.py" in structure.python_files
    assert structure.source_dirs == ["src"]
    assert structure.test_dirs == ["tests"]
    # 忽略目录中的文件不得出现在结果里
    assert not [path for path in structure.python_files if path.startswith((".venv/", ".git/"))]
