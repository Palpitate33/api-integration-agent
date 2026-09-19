"""PatchGenerator 单元测试（全部离线）。

覆盖要求场景：
    1. create file → 正确 diff        9. JSON round-trip
    2. modify file with old/new → diff 10. deterministic output
    3. multiple files → 汇总          11. empty content
    4. dependency changes             12. unicode content
    5. no changes                     13. large content limit
    6. path traversal                 14. warnings
    7. absolute path                  15. Pipeline integration
    8. Windows path
"""

from pathlib import Path

from integration_agent import generation, patch, pipeline

CREATE_CONTENT = "VALUE = 1\n"


def _artifacts(
    files: list[generation.GeneratedFile] | None = None,
) -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(
        files=files or [],
        dependency_changes=[],
        summary="合成产物（单元测试用）",
    )


def _create(path: str, content: str = CREATE_CONTENT) -> generation.GeneratedFile:
    return generation.GeneratedFile(path=path, action="create", content=content)


def _modify_snippet(path: str) -> generation.GeneratedFile:
    return generation.GeneratedFile(
        path=path,
        action="modify",
        content='"pydantic>=2.0",\n',
        purpose="声明新增依赖",
        insertion_point="[project] 的 dependencies 列表内",
        changes=['"pydantic>=2.0"'],
    )


def _generate(artifacts, original_files=None) -> patch.PatchResult:
    return patch.DeterministicPatchGenerator().generate(artifacts, original_files=original_files)


# ------------------------------------------- 场景 1：create → 正确 diff


def test_create_file_diff() -> None:
    result = _generate(_artifacts([_create("src/client.py")]))

    assert result.files_created == 1
    assert result.files_modified == 0
    item = result.files[0]
    assert item.action == "create"
    assert item.diff_available is True
    assert item.new_content == CREATE_CONTENT
    assert "--- /dev/null" in item.diff
    assert "+++ b/src/client.py" in item.diff
    assert "+VALUE = 1" in item.diff
    assert item.diff in result.unified_diff


# ------------------------------ 场景 2：modify old/new → 真实 diff


def test_modify_with_old_and_new_generates_diff() -> None:
    full_modify = generation.GeneratedFile(
        path="src/client.py",
        action="modify",
        content="VALUE = 2\n",  # 全文模式：无 insertion_point
        purpose="修正返回值",
    )
    result = _generate(
        _artifacts([full_modify]),
        original_files={"src/client.py": "VALUE = 1\n"},
    )

    item = result.files[0]
    assert item.action == "modify"
    assert item.diff_available is True
    assert item.old_content == "VALUE = 1\n"
    assert item.new_content == "VALUE = 2\n"
    assert "--- a/src/client.py" in item.diff
    assert "+++ b/src/client.py" in item.diff
    assert "-VALUE = 1" in item.diff
    assert "+VALUE = 2" in item.diff


# ------------------------------------- 场景 3：multiple files → 汇总


def test_multiple_files_summary() -> None:
    result = _generate(
        _artifacts(
            [
                _create("src/a.py", "A = 1\n"),
                _create("src/b.py", "B = 2\n"),
                _modify_snippet("pyproject.toml"),
            ]
        )
    )

    assert result.files_changed == 3
    assert result.files_created == 2
    assert result.files_modified == 1
    assert result.summary.total_files == 3
    assert result.summary.created == 2
    assert result.summary.modified == 1
    assert "+++ b/src/a.py" in result.unified_diff
    assert "+++ b/src/b.py" in result.unified_diff


# --------------------------------------- 场景 4：dependency changes


def test_dependency_changes_preserved() -> None:
    artifacts = _artifacts([_create("src/a.py")])
    artifacts.dependency_changes = [
        generation.DependencyChange(name="pydantic", version=">=2.0", action="add")
    ]
    result = _generate(artifacts)

    assert len(result.dependency_changes) == 1
    assert result.dependency_changes[0].name == "pydantic"
    assert result.summary.dependencies == 1


# --------------------------------------------- 场景 5：no changes


def test_no_changes() -> None:
    result = _generate(_artifacts([]))

    assert result.files == []
    assert result.unified_diff == ""
    assert result.files_changed == 0
    assert result.summary.total_files == 0
    assert result.warnings == []


# --------------------------------------- 场景 6-8：非法 path 拒绝


def test_parent_traversal_rejected() -> None:
    result = _generate(_artifacts([_create("../evil.py")]))

    assert result.files == []
    assert any("越界" in item for item in result.warnings)


def test_absolute_unix_path_rejected() -> None:
    result = _generate(_artifacts([_create("/etc/passwd")]))

    assert result.files == []
    assert any("绝对路径" in item for item in result.warnings)


def test_windows_absolute_path_rejected() -> None:
    result = _generate(_artifacts([_create("C:\\evil.py")]))

    assert result.files == []
    assert any("Windows 绝对路径" in item for item in result.warnings)


# ---------------------------------------------- 场景 9：JSON round-trip


def test_json_roundtrip() -> None:
    result = _generate(_artifacts([_create("src/a.py"), _modify_snippet("pyproject.toml")]))

    restored = patch.PatchResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.unified_diff == result.unified_diff


# --------------------------------------------- 场景 10：deterministic


def test_deterministic_output() -> None:
    artifacts = _artifacts(
        [_create("src/a.py"), _create("src/b.py", "B\n"), _modify_snippet("x.toml")]
    )
    assert _generate(artifacts) == _generate(artifacts)


# ---------------------------------------------- 场景 11：empty content


def test_empty_content_create() -> None:
    result = _generate(_artifacts([_create("src/empty.py", "")]))

    item = result.files[0]
    assert item.diff_available is True
    assert "--- /dev/null" in item.diff
    assert "+++ b/src/empty.py" in item.diff
    # 无内容行：除 +++/--- 头外没有 "+" 开头的代码行
    added = [
        line
        for line in item.diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert added == []


# --------------------------------------------- 场景 12：unicode content


def test_unicode_content() -> None:
    result = _generate(_artifacts([_create("src/中文.py", "# 注释：你好世界\n")]))

    assert "+++ b/src/中文.py" in result.unified_diff
    assert "+# 注释：你好世界" in result.unified_diff


# ------------------------------------------- 场景 13：large content limit


def test_large_content_skipped() -> None:
    generator = patch.DeterministicPatchGenerator(max_content_chars=10)
    result = generator.generate(_artifacts([_create("src/big.py", "X" * 100)]))

    assert result.files == []
    assert any("超过上限" in item for item in result.warnings)


# -------------------------------------------------- 场景 14：warnings


def test_snippet_modify_warns_and_preserves_structure() -> None:
    snippet = _modify_snippet("pyproject.toml")
    result = _generate(_artifacts([snippet]))

    item = result.files[0]
    assert item.action == "modify"
    assert item.diff_available is False
    assert item.diff is None
    assert item.old_content is None
    assert "插入点" in item.summary  # 保留插入点与片段等结构化修改信息
    assert any("修改片段" in warning for warning in result.warnings)


def test_modify_full_content_without_original_warns() -> None:
    full = generation.GeneratedFile(
        path="src/x.py", action="modify", content="NEW\n", purpose="修正"
    )
    result = _generate(_artifacts([full]))

    assert result.files[0].diff_available is False
    assert any("缺少原始文件内容" in item for item in result.warnings)


def test_snippet_with_original_records_old_content() -> None:
    result = _generate(
        _artifacts([_modify_snippet("demo_project/__init__.py")]),
        original_files={"demo_project/__init__.py": '"""旧内容。"""\n'},
    )

    item = result.files[0]
    assert item.old_content == '"""旧内容。"""\n'
    assert item.diff_available is False  # 片段语义：不伪造 diff


# -------------------------------------------- 场景 15：Pipeline integration


def test_pipeline_integration(tmp_path: Path) -> None:
    """真实 Pipeline：patch 正常生成；create 有 diff，modify 片段如实标记。"""
    result = pipeline.run_pipeline(
        Path(__file__).resolve().parent.parent / "examples" / "openapi" / "petstore.yaml",
        Path(__file__).resolve().parent.parent / "examples" / "demo_project",
    )

    assert result.status == "passed"
    assert result.patch is not None
    assert result.patch.summary.created == 7
    assert result.patch.summary.modified == 2
    assert result.patch.summary.dependencies == 1  # pydantic
    for item in result.patch.files:
        if item.action == "create":
            assert item.diff_available is True
            assert item.diff is not None
        else:
            assert item.diff_available is False  # 片段语义：不伪造 diff
            assert item.diff is None
    assert any("修改片段" in warning for warning in result.patch.warnings)


def test_pipeline_stage_error_has_no_patch(tmp_path: Path) -> None:
    result = pipeline.run_pipeline(tmp_path / "missing.yaml", tmp_path / "repo")

    assert result.status == "error"
    assert result.patch is None
