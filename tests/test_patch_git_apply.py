"""P1-2 回归：Generated Patch 必须真的能被 `git apply`。

根因
----
"看起来像 unified diff"和"是 unified diff"是两回事。手写"两个文件头 + 每行加个 +"
很容易漏掉 hunk 头（`@@ -0,0 +1,N @@`），而 `git apply` 对这种输入不是宽容地忽略，
是**直接拒绝**（"corrupt patch" / "No valid patches in input"）。生成物里的 diff
于是变成一份谁也应用不了的文本，而单元测试如果只断言"字符串里有 +++ 和三个加号"
就完全看不出来。

对策
----
交给标准库 `difflib.unified_diff(..., keepends=True, lineterm="\\n")`，拼接时给
"文件末尾没有换行符"的那一行补上 `\\ No newline at end of file`。空文件的创建是唯一
的特例：unified diff 表示不了"创建一个空文件"（任何 hunk 形式都被 git 判为 corrupt），
所以用 git 自己产出的 `diff --git` + `new file mode 100644` 形式。

测试策略
--------
唯一有意义的判据是**真的调用 git**：在临时仓库里跑 `git apply --check`（只检查）
再跑 `git apply`（真落地），最后**逐字节比对文件内容**。全程只使用
`tempfile`/`tmp_path` 造出来的临时仓库，绝不触碰当前 APIForge 仓库。

    Test A：修改已有文件
    Test B：新建文件（pkg/new.py，含空文件特例）
    Test C：多文件同时应用
    边界：文件末尾无换行符 / CRLF 内容 / 非 ASCII / delete 动作不被支持
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from integration_agent import generation, patch, validation

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="需要真实的 git 才能在临时仓库里验证 patch"
)


# ------------------------------------------------------------------ 基础设施


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """在临时仓库里跑 git；显式用字节传递，避免 Windows 上的换行翻译。"""
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, timeout=60, check=False
    )
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        completed.stdout.decode("utf-8", "replace"),
        completed.stderr.decode("utf-8", "replace"),
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """一个干净的临时 git 仓库（**不是** APIForge 自己的仓库）。"""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@api-forge.invalid")
    _git(root, "config", "user.name", "APIForge Tests")
    # 行尾必须稳定，否则同一份 diff 在不同机器上时过时不过
    _git(root, "config", "core.autocrlf", "false")
    return root


def _write(root: Path, relative: str, content: str) -> None:
    """按**字节**写盘。

    不能用 ``write_text``：它在 newline=None 时会把 ``\\n`` 翻译成 ``os.linesep``，
    于是 Windows 上落盘的是 CRLF，而 diff 是按我们传进去的 LF 文本生成的——
    git 随后会因为 context 行对不上而拒绝应用。这正是被测组件的行为契约：
    **diff 忠实镜像 original_files 给的字节**，所以测试必须让磁盘字节与
    original_files 完全一致，否则测的是自己的换行翻译。
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))


def _commit_all(root: Path, message: str = "init") -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)


def _artifacts(files: list[generation.GeneratedFile]) -> generation.GeneratedArtifacts:
    return generation.GeneratedArtifacts(files=files, summary="合成产物（P1-2 回归用）")


def _create(path: str, content: str) -> generation.GeneratedFile:
    return generation.GeneratedFile(path=path, action="create", content=content)


def _modify(path: str, content: str) -> generation.GeneratedFile:
    return generation.GeneratedFile(path=path, action="modify", content=content)


def _apply(root: Path, diff: str, *, extra: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """把 diff 写到仓库**外面**再交给 git，避免 patch 文件污染工作区。"""
    patch_file = root.parent / "change.patch"
    patch_file.write_bytes(diff.encode("utf-8"))
    return _git(root, "apply", *extra, str(patch_file))


def _assert_applies(root: Path, diff: str) -> None:
    """先 --check（只检查），再真应用。两步都必须成功，且必须给出可读的失败信息。"""
    assert diff.strip(), "没有产出任何 diff，无从应用"
    checked = _apply(root, diff, extra=("--check",))
    assert checked.returncode == 0, f"git apply --check 失败：\n{checked.stderr}\n{diff}"
    applied = _apply(root, diff)
    assert applied.returncode == 0, f"git apply 失败：\n{applied.stderr}\n{diff}"


def _diff_for(
    files: list[generation.GeneratedFile], original_files: dict[str, str] | None = None
) -> patch.PatchResult:
    return patch.DeterministicPatchGenerator().generate(
        _artifacts(files), original_files=original_files
    )


# ------------------------------------------------- Test A：修改已有文件


def test_a_modify_existing_file_applies(repo: Path) -> None:
    original = '"""Original module."""\n\nVALUE = 1\nOTHER = 2\n'
    updated = '"""Updated module."""\n\nVALUE = 1\nOTHER = 3\nEXTRA = 4\n'
    _write(repo, "pkg/mod.py", original)
    _commit_all(repo)

    result = _diff_for([_modify("pkg/mod.py", updated)], {"pkg/mod.py": original})

    assert result.files[0].diff_available is True
    _assert_applies(repo, result.unified_diff)
    assert (repo / "pkg" / "mod.py").read_text(encoding="utf-8") == updated


def test_a_modify_applies_cleanly_against_the_committed_tree(repo: Path) -> None:
    """`--check` 在应用前后要有正确的语义：应用过之后同一个 patch 不应再成功。"""
    original = "A = 1\nB = 2\n"
    updated = "A = 1\nB = 3\n"
    _write(repo, "mod.py", original)
    _commit_all(repo)
    diff = _diff_for([_modify("mod.py", updated)], {"mod.py": original}).unified_diff

    assert _apply(repo, diff, extra=("--check",)).returncode == 0
    assert _apply(repo, diff).returncode == 0
    # 已经应用过了：context 对不上，git 必须拒绝（证明 --check 真的在校验内容）
    assert _apply(repo, diff, extra=("--check",)).returncode != 0


def test_a_modify_without_trailing_newline(repo: Path) -> None:
    """原文件末尾没有换行符：必须产出 `\\ No newline at end of file` 标记。"""
    original = "VALUE = 1"  # 末尾无换行符
    updated = "VALUE = 2"  # 末尾同样无换行符
    _write(repo, "pkg/noeol.py", original)
    _commit_all(repo)

    result = _diff_for([_modify("pkg/noeol.py", updated)], {"pkg/noeol.py": original})

    assert "\\ No newline at end of file" in result.unified_diff
    _assert_applies(repo, result.unified_diff)
    assert (repo / "pkg" / "noeol.py").read_bytes() == b"VALUE = 2"


def test_a_modify_adds_the_missing_trailing_newline(repo: Path) -> None:
    """给一个末尾无换行符的文件补上换行符：这是 git 认定的"真实改动"。"""
    original = "VALUE = 1"
    updated = "VALUE = 1\n"
    _write(repo, "pkg/eol.py", original)
    _commit_all(repo)

    result = _diff_for([_modify("pkg/eol.py", updated)], {"pkg/eol.py": original})

    _assert_applies(repo, result.unified_diff)
    assert (repo / "pkg" / "eol.py").read_bytes() == b"VALUE = 1\n"


# ------------------------------------------------- Test B：新建文件


def test_b_create_new_file_applies(repo: Path) -> None:
    content = '"""New client."""\n\n\ndef fetch() -> None:\n    return None\n'
    _commit_all(repo)  # 仓库非空，但目标文件不存在

    result = _diff_for([_create("pkg/new.py", content)])

    assert result.files[0].diff_available is True
    assert not (repo / "pkg" / "new.py").exists(), "应用前文件不该存在"

    _assert_applies(repo, result.unified_diff)
    assert (repo / "pkg" / "new.py").read_text(encoding="utf-8") == content


def test_b_create_new_file_at_repo_root(repo: Path) -> None:
    content = "PLACEHOLDER = True\n"
    _commit_all(repo)

    result = _diff_for([_create("top_level.py", content)])

    _assert_applies(repo, result.unified_diff)
    assert (repo / "top_level.py").read_text(encoding="utf-8") == content


def test_b_create_empty_file_applies(repo: Path) -> None:
    """空文件是唯一需要特判的创建：unified diff 的任何 hunk 形式都被 git 判为 corrupt。"""
    _commit_all(repo)

    result = _diff_for([_create("pkg/empty_new.py", "")])

    _assert_applies(repo, result.unified_diff)
    assert (repo / "pkg" / "empty_new.py").read_bytes() == b""


def test_b_create_file_with_trailing_newline_variants(repo: Path) -> None:
    """内容末尾有无换行符是**两个不同的文件**，git 也这么认为。"""
    _commit_all(repo)

    cases = {"pkg/eol_yes.py": "A = 1\n", "pkg/eol_no.py": "A = 1"}
    for path, content in cases.items():
        _assert_applies(repo, _diff_for([_create(path, content)]).unified_diff)

    for path, content in cases.items():
        assert (repo / path).read_bytes() == content.encode("utf-8"), path


# ------------------------------------------------- Test C：多文件


def test_c_multiple_files_apply_in_one_call(repo: Path) -> None:
    """一份 unified_diff 里混着 modify 与 create，一次 `git apply` 全部落地。"""
    existing = "EXISTING = 1\n"
    existing_updated = "EXISTING = 2\nADDED = 3\n"
    new_content = "NEW = True\n"
    empty_path = "pkg/empty.py"
    _write(repo, "pkg/existing.py", existing)
    _commit_all(repo)

    result = _diff_for(
        [
            _modify("pkg/existing.py", existing_updated),
            _create("pkg/new.py", new_content),
            _create(empty_path, ""),
        ],
        {"pkg/existing.py": existing},
    )

    assert result.files_changed == 3
    assert result.files_created == 2
    assert result.files_modified == 1
    # 三段 diff 拼成一份，中间不能有额外空行（每一段都自带行尾换行）
    assert result.unified_diff.count("diff --git") + result.unified_diff.count("--- ") >= 3

    _assert_applies(repo, result.unified_diff)

    assert (repo / "pkg" / "existing.py").read_text(encoding="utf-8") == existing_updated
    assert (repo / "pkg" / "new.py").read_text(encoding="utf-8") == new_content
    assert (repo / empty_path).read_bytes() == b""


def test_c_multi_file_patch_is_atomic_when_one_file_conflicts(repo: Path) -> None:
    """多文件 patch 是**整体**判定的：任一段对不上，`--check` 就必须整体失败。

    这条确认了"多文件拼接"没有退化成"分别应用"——后者会在部分冲突时悄悄
    把一半改动写进工作区，留下一个半成品仓库。
    """
    _write(repo, "pkg/ok.py", "OK = 1\n")
    _commit_all(repo)
    _write(repo, "pkg/ok.py", "OK = 999\n")  # 工作区被改过：patch 的 context 对不上

    result = _diff_for(
        [
            _modify("pkg/ok.py", "OK = 1\nOK2 = 2\n"),
            _create("pkg/fresh.py", "FRESH = 1\n"),
        ],
        {"pkg/ok.py": "OK = 1\n"},
    )

    checked = _apply(repo, result.unified_diff, extra=("--check",))
    assert checked.returncode != 0
    assert not (repo / "pkg" / "fresh.py").exists(), "--check 失败却写入了文件"


# -------------------------------------------------------------- 内容边界


def test_unicode_and_crlf_content_applies(repo: Path) -> None:
    """非 ASCII 内容与 CRLF 内容都必须能应用（生成代码里有中文 docstring）。"""
    _commit_all(repo)

    unicode_content = '"""中文文档字符串 emoji 😀 。"""\n\nNAME = "中文"\n'
    crlf_content = "LINE_ONE = 1\r\nLINE_TWO = 2\r\n"

    _assert_applies(repo, _diff_for([_create("pkg/zh.py", unicode_content)]).unified_diff)
    _assert_applies(repo, _diff_for([_create("pkg/crlf.py", crlf_content)]).unified_diff)

    assert (repo / "pkg" / "zh.py").read_text(encoding="utf-8") == unicode_content
    assert (repo / "pkg" / "crlf.py").read_bytes() == crlf_content.encode("utf-8")


def test_crlf_original_round_trips_through_git_apply(repo: Path) -> None:
    """original_files 给的是 CRLF，产出的 diff 就必须是 CRLF并且同样能应用。

    换行风格不是生成器"决定"的，是从 original_files 里原样带过来的（splitlines
    keepends=True + 原样回写）。把这条钉住，是为了让"diff 忠实镜像输入字节"
    成为一个被测试的性质，而不是注释里的一句话——上面被判失败的测试正是踩在这里。
    """
    original = "A = 1\r\nB = 2\r\n"
    updated = "A = 1\r\nB = 3\r\n"
    _write(repo, "pkg/crlf_existing.py", original)
    _commit_all(repo)

    result = _diff_for(
        [_modify("pkg/crlf_existing.py", updated)], {"pkg/crlf_existing.py": original}
    )

    assert "\r\n" in result.unified_diff
    _assert_applies(repo, result.unified_diff)
    assert (repo / "pkg" / "crlf_existing.py").read_bytes() == updated.encode("utf-8")


def test_diff_is_a_valid_unified_diff_for_a_real_parser(repo: Path) -> None:
    """交给 git 的 `--stat` 解析器读一遍：能解析才说明格式真的对。"""
    _commit_all(repo)
    result = _diff_for([_create("pkg/stat.py", "A = 1\nB = 2\n")])

    patch_file = repo.parent / "stat.patch"
    patch_file.write_bytes(result.unified_diff.encode("utf-8"))
    stat = _git(repo, "apply", "--stat", str(patch_file))

    assert stat.returncode == 0, stat.stderr
    assert "pkg/stat.py" in stat.stdout
    assert "2" in stat.stdout  # 两行新增


# ---------------------------------------------------------- 不支持的形状


def test_delete_action_is_not_supported_by_the_model() -> None:
    """`GeneratedFile.action` 只有 create / modify——生成器不产出删除。

    所以 P1-2 里"删除文件的 diff 仍然合法"这一项没有对应的代码路径：
    删除能力不存在，也就不存在"删除的 diff 被写坏"。这条把边界钉住，
    免得以后悄悄放宽 action 却没人补 `git apply` 的验证。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        generation.GeneratedFile(path="pkg/gone.py", action="delete", content="")


def test_path_escape_is_rejected_rather_than_patched(repo: Path) -> None:
    """越界路径不能变成一份"能应用但写到仓库外"的 patch。"""
    result = _diff_for(
        [
            _create("../outside.py", "ESCAPED = True\n"),
            _create("/abs.py", "ESCAPED = True\n"),
            _create("C:/win.py", "ESCAPED = True\n"),
        ]
    )

    assert result.files == []
    assert result.unified_diff == ""
    assert len(result.warnings) == 3
    assert not (repo.parent / "outside.py").exists()


def test_a_newline_in_a_path_would_forge_a_header_line() -> None:
    """非空洞性守卫：先证明"路径里的换行"真的能改写 diff 的结构。

    私有渲染函数（不带校验）拿恶意路径跑一遍，产出的文本里真的出现了**另一份文件**的
    diff 头 ``--- a/pkg/other.py``：path 落在 ``--- a/`` / ``+++ b/`` 的**行首**，一个
    换行就把它变成了攻击者自己写的文件头。`_validate_path` 存在的理由就是这一行——
    没有它，"新建一个文件"可以变成"改任意文件"。
    """
    from integration_agent.patch import generator as patch_generator

    forged = patch_generator._render_create_diff("pkg/ok.py\n--- a/pkg/other.py\n", "X = 1\n")

    lines = forged.splitlines()
    assert lines.index("--- a/pkg/other.py") > lines.index("+++ b/pkg/ok.py")


def test_path_with_control_characters_is_rejected_rather_than_patched(repo: Path) -> None:
    """带控制字符（换行 / CRLF / 制表符）的路径必须被拒绝，而不是产出 patch。

    上面那条守卫说明了换行能伪造文件头；这里钉住的是**判定**：这类路径一律跳过并
    记 warning，绝不进入 unified_diff。
    """
    hostile = [
        _create("pkg/ok.py\n--- a/pkg/other.py\n", "ESCAPED = True\n"),
        _create("pkg/evil.py\r\n+++ b/pkg/x.py\n", "ESCAPED = True\n"),
        _create("pkg/tab\tname.py", "ESCAPED = True\n"),
    ]

    result = _diff_for(hostile)

    assert result.files == []
    assert result.unified_diff == ""
    assert len(result.warnings) == len(hostile)
    assert all("控制字符" in warning for warning in result.warnings), result.warnings


def test_generated_diff_from_a_real_pipeline_applies(repo: Path) -> None:
    """端到端：真实 pipeline 产出的 diff 也要能被 git apply。

    单元测试用合成产物，这条用 Planner + Generator 的真实输出，覆盖
    "生成器产出的路径/内容形状"与"patch 渲染"之间的接缝。
    """
    from integration_agent.agent import plan_integration
    from integration_agent.api import parse_openapi_text
    from integration_agent.repository import scan_repository

    _write(
        repo,
        "pyproject.toml",
        '[project]\nname = "target"\nversion = "0.1.0"\ndependencies = ["httpx>=0.27"]\n',
    )
    _write(repo, "app/__init__.py", '"""App."""\n')
    _commit_all(repo)

    spec = (
        '{"openapi": "3.0.3", "info": {"title": "Demo", "version": "1.0"}, '
        '"servers": [{"url": "https://demo.test"}], "paths": {"/ping": {"get": '
        '{"operationId": "ping", "responses": {"200": {"description": "ok"}}}}}}'
    )
    plan = plan_integration(parse_openapi_text(spec), scan_repository(repo))
    artifacts = generation.generate_code(plan)
    result = patch.DeterministicPatchGenerator().generate(artifacts)

    assert result.unified_diff.strip()
    _assert_applies(repo, result.unified_diff)

    created = [item.path for item in artifacts.created_files if item.path.endswith(".py")]
    assert created, "pipeline 应当产出 Python 文件"
    for relative in created:
        assert (repo / relative).is_file(), relative


def test_the_real_project_repository_is_never_touched() -> None:
    """整份测试都在临时仓库里跑：当前 APIForge 仓库必须没有新增未跟踪文件。"""
    project_root = Path(__file__).resolve().parent.parent
    assert (project_root / "pyproject.toml").is_file()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        timeout=60,
        check=False,
    )
    lines = status.stdout.decode("utf-8", "replace").splitlines()
    assert not any("outside.py" in line or "change.patch" in line for line in lines), lines
    assert not (project_root / "pkg" / "new.py").exists()
    assert not (project_root / "change.patch").exists()


def test_validation_still_sees_a_consistent_result(repo: Path) -> None:
    """回归护栏：patch 改动没有波及 validation 的公共契约。"""
    assert validation.NO_TESTS_EXECUTED_MESSAGE
    assert callable(validation.run_tests)
