"""Deterministic Patch Generator：把最终 GeneratedArtifacts 变成结构化 PatchResult。

契约：GeneratedArtifacts → PatchResult。

设计原则：
    - 纯函数：不修改真实 repository、不执行 git / shell / subprocess、
      不访问网络、不读取 secrets、不自行扫描仓库。
    - 原始文件内容由调用方显式提供（original_files: Mapping[path, content]），
      本组件没有文件系统副作用。
    - 不伪造数据：
        create  → 完整 content 生成真实 unified diff（--- /dev/null +++ b/path）
        modify 片段（带 insertion_point）→ diff_available=False，保留结构化信息
        modify 全文（无 insertion_point）+ 提供原始内容 → 真实 unified diff
        modify 全文但缺原始内容 → diff_available=False + warning
    - path 安全校验（拒绝越界 / 绝对路径 / Windows 盘符），文件数量与
      内容大小有硬上限；任何问题只跳过该文件并记录 warning。
    - 确定性：文件顺序 = artifacts 顺序，无随机/时间/无序集合。
"""

import difflib
import re
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from integration_agent.generation import GeneratedArtifacts, GeneratedFile
from integration_agent.patch.models import PatchFile, PatchResult, PatchSummary

DEFAULT_MAX_FILES = 500  # 单次 Patch 允许的最大文件数
DEFAULT_MAX_CONTENT_CHARS = 200_000  # 单文件内容上限
DEFAULT_MAX_TOTAL_DIFF_CHARS = 2_000_000  # unified_diff 总量上限

_WINDOWS_ABSOLUTE = re.compile(r"^[a-zA-Z]:")

# 控制字符（含 \n \r \t 与 DEL）。路径会出现在 `diff --git a/... b/...`、
# `--- a/...`、`+++ b/...` 这三处的**行首**位置：带换行的路径会让这一行提前结束，
# 剩下的文本就变成攻击者可控的 diff 正文（足以塞进额外 hunk 或整段文件头）。
# 控制字符在路径里没有正当用途，直接拒绝。
_PATH_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@runtime_checkable
class PatchGenerator(Protocol):
    """PatchGenerator 接口：确定性实现与未来扩展共用同一契约。"""

    def generate(
        self,
        artifacts: GeneratedArtifacts,
        *,
        original_files: Mapping[str, str] | None = None,
    ) -> PatchResult:
        """把生成产物转换为最终修改结果。"""
        ...


class DeterministicPatchGenerator:
    """规则驱动的确定性 Patch 生成器：只读输入，只产数据。"""

    def __init__(
        self,
        *,
        max_files: int = DEFAULT_MAX_FILES,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        max_total_diff_chars: int = DEFAULT_MAX_TOTAL_DIFF_CHARS,
    ) -> None:
        self.max_files = max_files
        self.max_content_chars = max_content_chars
        self.max_total_diff_chars = max_total_diff_chars

    def generate(
        self,
        artifacts: GeneratedArtifacts,
        *,
        original_files: Mapping[str, str] | None = None,
    ) -> PatchResult:
        original = dict(original_files or {})
        files: list[PatchFile] = []
        warnings: list[str] = []
        total_diff_chars = 0

        for item in artifacts.files:
            if len(files) >= self.max_files:
                warnings.append(f"文件数量超过上限 {self.max_files}，剩余文件被跳过")
                break
            patch_file, warning = self._build_file(item, original)
            if patch_file is None:
                warnings.append(warning or f"跳过文件 {item.path}")
                continue
            if warning:
                warnings.append(warning)
            if patch_file.diff and total_diff_chars + len(patch_file.diff) > (
                self.max_total_diff_chars
            ):
                warnings.append(
                    f"累计 diff 超过总上限 {self.max_total_diff_chars} 字符，"
                    f"{item.path} 及之后的 diff 不再并入 unified_diff"
                )
                patch_file = patch_file.model_copy(update={"diff": None, "diff_available": False})
            if patch_file.diff:
                total_diff_chars += len(patch_file.diff)
            files.append(patch_file)

        # 每段 diff 都已自带行尾换行，直接拼接；再 join("\n") 会在文件之间插入空行
        unified_diff = "".join(item.diff for item in files if item.diff)
        created = sum(1 for item in files if item.action == "create")
        modified = len(files) - created
        return PatchResult(
            files=files,
            dependency_changes=list(artifacts.dependency_changes),
            unified_diff=unified_diff,
            files_changed=len(files),
            files_created=created,
            files_modified=modified,
            warnings=warnings,
            summary=PatchSummary(
                total_files=len(files),
                created=created,
                modified=modified,
                dependencies=len(artifacts.dependency_changes),
            ),
        )

    def _build_file(
        self, item: GeneratedFile, original: dict[str, str]
    ) -> tuple[PatchFile | None, str | None]:
        """把单个 GeneratedFile 转为 PatchFile；不合法/超限时返回 (None, 原因)。"""
        path_error = _validate_path(item.path)
        if path_error:
            return None, f"跳过 {item.path}：{path_error}"
        if len(item.content) > self.max_content_chars:
            return (
                None,
                f"跳过 {item.path}：内容 {len(item.content)} 字符超过上限 {self.max_content_chars}",
            )
        if item.action == "create":
            diff = _render_create_diff(item.path, item.content)
            return (
                PatchFile(
                    path=item.path,
                    action="create",
                    old_content=None,
                    new_content=item.content,
                    diff=diff,
                    diff_available=True,
                    summary=item.purpose,
                ),
                None,
            )
        # modify
        old_content = original.get(item.path)
        if item.insertion_point:
            # 片段语义：无法安全推导完整新内容，不伪造 diff
            snippet = item.content.strip().splitlines()
            preview = " ".join(snippet[:2]) if snippet else item.content[:200]
            summary = f"插入点：{item.insertion_point}；片段：{preview}"
            return (
                PatchFile(
                    path=item.path,
                    action="modify",
                    old_content=old_content,
                    new_content=None,
                    diff=None,
                    diff_available=False,
                    summary=summary,
                ),
                f"{item.path} 是修改片段（insertion_point），无法安全推导完整新内容，"
                "未生成 unified diff",
            )
        if old_content is None:
            return (
                PatchFile(
                    path=item.path,
                    action="modify",
                    old_content=None,
                    new_content=None,
                    diff=None,
                    diff_available=False,
                    summary=item.purpose,
                ),
                f"{item.path} 缺少原始文件内容（original_files 未提供），无法生成 unified diff",
            )
        diff = _render_diff(f"a/{item.path}", f"b/{item.path}", old_content, item.content)
        return (
            PatchFile(
                path=item.path,
                action="modify",
                old_content=old_content,
                new_content=item.content,
                diff=diff,
                diff_available=True,
                summary=item.purpose,
            ),
            None,
        )


def _render_create_diff(path: str, new_text: str) -> str:
    """create 文件的 unified diff：标准 --- /dev/null +++ b/path 头 + 新增行。

    手写"头 + 每行加个 +"是不行的：unified diff 的 hunk 必须带 ``@@ -0,0 +1,N @@``，
    漏掉它产出的就不是 unified diff，`git apply` 会直接拒绝（而不是宽容地忽略）。
    交给 difflib：从空序列推到 N 行时它给出的 hunk 头本来就是正确的。

    空文件是唯一需要特判的情况，而且**不是因为 difflib 不给头**——是 unified diff
    根本表示不了"创建一个空文件"：一个 hunk 至少要有一行正文（``@@ -0,0 +1,0 @@``
    与 ``@@ -0,0 +0,0 @@`` 都被 git 判为 corrupt），只给 ``--- /dev/null`` / ``+++``
    两个头则被 git 判为 "No valid patches in input"。
    所以空文件用 git 自己产出的那种形式（``diff --git`` + ``new file mode``），
    并保留 --- /dev/null 与 +++ 两个头，让两种 create 的 diff 长得一致。
    """
    if not new_text:
        return f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
    return _render_diff("/dev/null", f"b/{path}", "", new_text)


def _render_diff(from_label: str, to_label: str, old_text: str, new_text: str) -> str:
    """unified diff：标准库 difflib，输出保证能被 `git apply` 接受。

    ``keepends=True`` 是必需的：行尾换行必须留在行内容里，否则"文件末尾没有换行符"
    与"末尾有换行符"两种文件会 diff 成同一个结果，而 git 认为它们是不同的文件。
    ``lineterm="\\n"`` 让控制行（``---`` / ``+++`` / ``@@``）自己带上换行，
    于是**每个分块都是自洽的一行**，拼接时不需要再猜哪里该补换行。
    """
    chunks = list(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=from_label,
            tofile=to_label,
            lineterm="\n",
        )
    )
    return _join_diff_chunks(chunks)


def _join_diff_chunks(chunks: list[str]) -> str:
    """把 difflib 的分块拼成 unified diff 文本。

    分块只有两类：控制行（自带换行）与正文行（keepends=True，自带换行）。
    因此**唯一不带换行的分块，就是"文件末尾没有换行符"的那一行**——不需要按前缀
    猜哪块是控制行（那样会把内容为 ``-- x`` 的删除行误判成文件头）。

    这种行后面必须跟一条 ``\\ No newline at end of file`` 标记（git 的格式要求）：
    缺了它，下一段 hunk 头会被当成这一行的内容，整个 patch 作废。difflib 不会加，
    这里补上。标记行本身也是 patch 的一部分，所以只补一次。
    """
    output: list[str] = []
    for chunk in chunks:
        output.append(chunk)
        if not chunk.endswith("\n"):
            output.append("\n\\ No newline at end of file\n")
    return "".join(output)


def _validate_path(path: str) -> str | None:
    """校验 path 为安全相对路径；返回错误描述，合法时返回 None。"""
    if not path or not path.strip():
        return f"path 为空：{path!r}"
    if _PATH_CONTROL.search(path):
        # 路径不是"内容"而是 diff 的**结构**：换行能让它从路径头里逃出来，
        # 后续文本就成了攻击者书写的 hunk（见 _PATH_CONTROL 注释）。
        return f"path 含控制字符：{path!r}"
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        return f"path 不能是绝对路径：{path!r}"
    if _WINDOWS_ABSOLUTE.match(normalized):
        return f"path 不能是 Windows 绝对路径：{path!r}"
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return f"path 包含越界片段：{path!r}"
    return None
