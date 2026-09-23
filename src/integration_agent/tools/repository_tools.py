"""三个面向项目的只读工具：inspect_project / search_code / read_file。

默认注册表 ``build_default_registry`` 也放在这里，它同时注册 api_tools 的
inspect_api。注册表暂住本模块是历史原因，更好的位置是 tools/__init__.py；
挪动它会改动公开导入路径，留到后续 commit 一起做。

复用而非重写：
    项目扫描复用 ``repository.scanner.scan_repository``，代码搜索复用
    ``repository.code_search.search_code``。这里只做三件它们不做的事——
    把不可信入参收敛成安全形状、把结果裁剪成适合进 LLM 上下文的规模、
    把失败翻译成结构化 ToolResult。

安全边界：
    这是只读阶段。三个工具都不写文件、不删文件、不执行 shell / git / 子进程、
    不发网络请求、不读环境变量、不读 API Key。一切路径都必须落在
    ``ToolContext.project_root`` 之内，且经 ``resolve()`` 确认（可挡 symlink 逃逸）。
    失败一律返回 ``ok=False`` 的 ToolResult，错误文本不含 traceback、
    不含绝对服务器路径——异常原文（例如 RepositoryError 会带上绝对路径）
    绝不直接透传。

依赖方向：
    tools → repository，不反向依赖 agent / repair / pipeline / api_server。
"""

import logging
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from integration_agent.repository import (
    ProjectStructure,
    RepositoryError,
    scan_repository,
)
from integration_agent.repository import search_code as search_project_code
from integration_agent.repository.scanner import validate_project_root
from integration_agent.tools.api_tools import InspectAPITool
from integration_agent.tools.models import ToolResult, ToolSpec
from integration_agent.tools.registry import AgentTool, ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

# 各工具的硬上限。这些值是天花板而非默认值：模型只能收紧，不能放宽，
# 否则一次大范围搜索就能把整个仓库灌进上下文。
MAX_INSPECT_CHARS = 4000
MAX_INSPECT_FILES = 80
MAX_INSPECT_DEPENDENCIES = 40
MAX_SEARCH_CHARS = 6000
MAX_SEARCH_RESULTS = 20
MAX_SEARCH_QUERY_CHARS = 200
MAX_READ_CHARS = 8000
MAX_READ_BYTES = 256 * 1024

# 拒绝读取的文件名/后缀：这些文件按惯例装着凭据，读取它们没有任何正当用途。
_SECRET_FILE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        "_netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        ".htpasswd",
        ".pgpass",
        "credentials",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)
_SECRET_FILE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".keystore")
_ENV_FILE_PATTERN = re.compile(r"^\.env(\..+)?$")

# 结果被裁剪时追加的提示。长度会被预留进预算，见 _join_within_cap
_TRUNCATION_NOTE = "\n... 输出已被截断，请缩小范围后重试"


class _ArgRejected(ValueError):
    """工具入参不满足约束。

    消息可以直接回给模型：只说参数哪里不对，不含服务器路径、环境变量或异常原文。
    """


def _ok(tool: str, content: str, *, truncated: bool = False) -> ToolResult:
    """构造成功结果。chars 显式取实际长度，供调用方累计观察预算。"""
    return ToolResult(
        call_id="",
        tool=tool,
        ok=True,
        content=content,
        truncated=truncated,
        chars=len(content),
    )


def _fail(tool: str, message: str) -> ToolResult:
    """构造失败结果。content 保持空串，原因只放在 error 里。"""
    return ToolResult(call_id="", tool=tool, ok=False, content="", error=message, chars=0)


def _join_within_cap(blocks: list[str], cap: int, *, note: str = "") -> tuple[str, bool]:
    """按顺序拼接文本块，保证结果不超过 cap 个字符。

    这是最后一道兜底：每个小节自己也会截断（见 ``_render_section``），
    但项目目录是不可信输入，谁也不能保证某一小节不会长到离谱。

    note 是截断时追加的提示，它的长度会**预先从预算里扣掉**。
    否则"结果被截断"这句话本身可能把结果顶出上限——一个自相矛盾的边界。
    """
    budget = max(cap - len(note), 1)
    kept: list[str] = []
    used = 0
    for block in blocks:
        cost = len(block) + (1 if kept else 0)  # 拼接用的换行本身也占字符
        if used + cost > budget:
            remaining = budget - used - 1
            if remaining > 0:
                kept.append(block[:remaining])
            return "\n".join(kept) + note, True
        kept.append(block)
        used += cost
    return "\n".join(kept), False


def _render_section(label: str, values: list[str], cap: int) -> tuple[str, bool]:
    """渲染一个列表小节；超过 cap 条时保序截断。

    省略数量写在**标题行**而不是列表末尾：末尾会被字符上限先切掉，
    于是模型看到一份缺了很多条、却没有任何提示的清单——比直接报错更误导。
    """
    if not values:
        return f"{label}: (none)", False
    if len(values) > cap:
        omitted = len(values) - cap
        header = f"{label} (共 {len(values)} 条，仅列前 {cap} 条，其余 {omitted} 条已省略):"
        truncated = True
    else:
        header = f"{label} ({len(values)}):"
        truncated = False
    lines = "\n".join(f"  {value}" for value in values[:cap])
    return f"{header}\n{lines}", truncated


def _require_str(args: dict[str, Any], key: str) -> str:
    """取一个必填的非空字符串参数。"""
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _ArgRejected(f"{key} 必须是非空字符串")
    return value.strip()


def _optional_bool(args: dict[str, Any], key: str, *, default: bool) -> bool:
    """取一个可选布尔参数；缺失时用默认值。"""
    value = args.get(key, default)
    if not isinstance(value, bool):
        raise _ArgRejected(f"{key} 必须是布尔值")
    return value


def _optional_bounded_int(args: dict[str, Any], key: str, *, ceiling: int, default: int) -> int:
    """取一个只允许收紧、不允许放宽的整数上限参数。

    模型可以把 max_results 调小（结果太吵时很有用），但调不大——
    天花板由工具常量决定，不交给调用方。
    """
    value = args.get(key)
    if value is None:
        return default
    # bool 是 int 的子类，True 会被 isinstance 放过，这里显式挡掉
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ArgRejected(f"{key} 必须是整数")
    if value < 1:
        raise _ArgRejected(f"{key} 必须大于 0")
    return min(value, ceiling)


def _validate_root(ctx: ToolContext) -> Path:
    """校验并解析 project_root；失败时抛出 _ArgRejected，不泄漏异常原文。

    RepositoryError 的消息里带着项目绝对路径（scanner 的 validate_project_root
    就是这么写的），所以这里必须换一句，不能透传。
    """
    try:
        return validate_project_root(ctx.project_root)
    except (RepositoryError, OSError) as exc:
        raise _ArgRejected("项目目录不存在或不可访问") from exc


def _is_secret_file(name: str) -> bool:
    """判断文件名是否属于"按惯例装凭据"的那一类。"""
    lowered = name.lower()
    return (
        lowered in _SECRET_FILE_NAMES
        or lowered.endswith(_SECRET_FILE_SUFFIXES)
        or _ENV_FILE_PATTERN.match(lowered) is not None
    )


def _resolve_inside_project(root: Path, raw: str) -> Path:
    """把不可信的 path 解析成 project_root 内的真实路径。

    检查顺序是刻意的：先判形状、再判包含关系、最后才碰文件系统。
    如果在包含关系之前就去查"文件存不存在"，越界路径会得到"文件不存在"而不是
    "越界"——那等于用报错信息告诉调用方 root 外面有什么。
    """
    text = raw.strip()
    if not text:
        raise _ArgRejected("path 不能为空")
    if "\x00" in text:
        raise _ArgRejected("path 不允许包含空字节")
    if text.startswith("~"):
        raise _ArgRejected("path 不允许使用 ~ 展开")

    # 两套规则都要判：本机是 Windows，但 "C:\\..." 在 POSIX 规则下不是绝对路径，
    # 只判一套会漏掉另一种平台的写法。
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise _ArgRejected("path 必须是项目内的相对路径，不允许绝对路径")
    if PureWindowsPath(text).drive:
        # 形如 "C:foo" 的盘符相对路径不满足 is_absolute()，但它同样能跳出当前目录
        raise _ArgRejected("path 必须是项目内的相对路径，不允许盘符")
    if ".." in PurePosixPath(text.replace("\\", "/")).parts:
        raise _ArgRejected("path 不允许包含 .. 逃逸")

    resolved = (root / text).resolve()
    if not resolved.is_relative_to(root):
        raise _ArgRejected("path 越出项目目录")
    return resolved


class _ReadOnlyTool:
    """只读工具的公共外壳：把意外异常挡在工具边界之内。

    AgentTool 契约要求工具不向上抛异常，所以这里兜住一切未预期的错误。
    但兜住不等于掩盖：完整堆栈用 ``logger.exception`` 写进服务端日志，
    回给模型的只有一句话——这正是 Pipeline 处理 traceback 的既有约定。
    若把异常原文当 error 返回，本地绝对路径与源码行就进了 LLM 上下文。
    """

    spec: ToolSpec

    def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            return self._run(args, ctx)
        except Exception:
            logger.exception("工具 %s 执行时发生未预期错误", self.spec.name)
            return _fail(self.spec.name, "工具内部错误，请换一种方式重试")

    def _run(self, args: dict, ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


class InspectProjectTool(_ReadOnlyTool):
    """按 project_root 扫描项目结构，返回精简的只读快照。"""

    spec = ToolSpec(
        name="inspect_project",
        description=(
            "扫描目标项目结构，返回项目名、清单文件、源码目录、测试目录、"
            "依赖与 .py 文件清单（均为相对路径）。不含文件内容。"
        ),
        parameters={"type": "object", "properties": {}},
    )

    def _run(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            root = _validate_root(ctx)
        except _ArgRejected as exc:
            return _fail(self.spec.name, str(exc))

        # 从 project_root 重新扫描，而不是直接读 ctx.project：工具应该在调用的
        # 这一刻看到磁盘真相，否则上层快照过期时工具会给出"看起来很确定"的旧答案。
        try:
            project = scan_repository(root)
        except (RepositoryError, OSError):
            return _fail(self.spec.name, "项目扫描失败")

        blocks, truncated = _render_project(project)
        text, cut = _join_within_cap(blocks, MAX_INSPECT_CHARS, note=_TRUNCATION_NOTE)
        return _ok(self.spec.name, text, truncated=truncated or cut)


def _render_project(project: ProjectStructure) -> tuple[list[str], bool]:
    """把 ProjectStructure 渲染成精简文本。

    刻意不输出 ``project.root``：它是绝对路径（scanner 里就是这么定义的），
    对"该改哪个文件"没有任何帮助，却会把服务器目录结构带进 LLM 上下文。
    """
    truncated = False
    blocks = [
        f"name: {project.name}",
        f"is_python_project: {project.is_python_project}",
    ]
    if project.skipped_paths:
        # 只在真的跳过过东西时才出现：这一节是"你看到的快照不完整"的提示，
        # 平时不占预算，出现时必须在模型下判断之前就摆在它眼前。
        section, cut = _render_section(
            "skipped_inaccessible", project.skipped_paths, MAX_INSPECT_DEPENDENCIES
        )
        blocks.append(section)
        truncated = truncated or cut
    for label, values, cap in (
        ("manifest_files", project.manifest_files, MAX_INSPECT_DEPENDENCIES),
        ("source_dirs", project.source_dirs, MAX_INSPECT_DEPENDENCIES),
        ("test_dirs", project.test_dirs, MAX_INSPECT_DEPENDENCIES),
        ("dependencies", project.dependencies, MAX_INSPECT_DEPENDENCIES),
        ("python_files", project.python_files, MAX_INSPECT_FILES),
    ):
        section, cut = _render_section(label, values, cap)
        blocks.append(section)
        truncated = truncated or cut
    return blocks, truncated


class SearchCodeTool(_ReadOnlyTool):
    """在项目内按关键词搜索 .py 文件，返回带上下文的受限片段。"""

    spec = ToolSpec(
        name="search_code",
        description=(
            "在目标项目的 .py 文件中按关键词做子串搜索（非正则，默认不区分大小写），"
            "返回带行号的上下文片段。只会读到 .py 文件内容。"
            "结果条数与总字符数都有上限，命中过多时请换更具体的关键词。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "case_sensitive": {
                    "type": "boolean",
                    "description": "是否区分大小写，默认 false",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_SEARCH_RESULTS,
                    "description": f"最多返回多少条命中，上限 {MAX_SEARCH_RESULTS}",
                },
            },
            "required": ["query"],
        },
    )

    def _run(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            query = _require_str(args, "query")
            if len(query) > MAX_SEARCH_QUERY_CHARS:
                raise _ArgRejected(f"query 过长，最多 {MAX_SEARCH_QUERY_CHARS} 个字符")
            case_sensitive = _optional_bool(args, "case_sensitive", default=False)
            max_results = _optional_bounded_int(
                args, "max_results", ceiling=MAX_SEARCH_RESULTS, default=MAX_SEARCH_RESULTS
            )
            root = _validate_root(ctx)
        except _ArgRejected as exc:
            return _fail(self.spec.name, str(exc))

        try:
            result = search_project_code(
                root,
                query,
                case_sensitive=case_sensitive,
                max_results=max_results,
                max_total_chars=MAX_SEARCH_CHARS,
            )
        except RepositoryError:
            # 走到这里说明是搜索器自己的参数校验失败或目录不可用；
            # 它的消息可能带绝对路径，一律换成固定文案。
            return _fail(self.spec.name, "搜索失败：查询或项目目录不可用")
        except OSError:
            return _fail(self.spec.name, "搜索失败：无法读取项目文件")

        if not result.matches:
            return _ok(
                self.spec.name,
                f"未找到匹配（已扫描 {result.files_scanned} 个 .py 文件）",
            )

        header = f"命中 {len(result.matches)} 处（扫描 {result.files_scanned} 个 .py 文件）"
        # 空串占位，join 之后就是标题与正文之间的空行
        blocks = [header, "", *(f"{m.path}:{m.line_number}\n{m.snippet}" for m in result.matches)]
        text, cut = _join_within_cap(blocks, MAX_SEARCH_CHARS, note=_TRUNCATION_NOTE)
        truncated = result.truncated or cut
        if truncated and not cut:
            # 搜索器自己截断了（命中数或总字符到顶），但拼装时没截。
            # 预算已按 note 预留，所以这里补上提示仍不会超过上限。
            text += _TRUNCATION_NOTE
        return _ok(self.spec.name, text, truncated=truncated)


class ReadFileTool(_ReadOnlyTool):
    """读取项目内某个文件的内容，带路径围栏与长度上限。"""

    spec = ToolSpec(
        name="read_file",
        description=(
            f"读取目标项目内某个文件的内容（最多 {MAX_READ_CHARS} 个字符，超出会被截断）。"
            "path 必须是相对于项目根目录的相对路径；绝对路径、盘符、.. 逃逸一律拒绝。"
            "不能读取目录、二进制文件与 .env 等凭据文件。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对项目根目录的文件路径，例如 src/pkg/client.py",
                }
            },
            "required": ["path"],
        },
    )

    def _run(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            raw_path = _require_str(args, "path")
            root = _validate_root(ctx)
            target = _resolve_inside_project(root, raw_path)
        except _ArgRejected as exc:
            return _fail(self.spec.name, str(exc))

        # 之后的报错都可以带上相对路径：它是模型自己给的值，不含服务器信息。
        relative = target.relative_to(root).as_posix()

        if target.is_dir():
            return _fail(self.spec.name, f"path 指向的是目录，不是文件：{relative}")
        if not target.is_file():
            return _fail(self.spec.name, f"文件不存在：{relative}")
        # 名字检查放在 resolve() 之后：symlink 指向 .env 时，名字也会变过来
        if _is_secret_file(target.name):
            return _fail(self.spec.name, f"拒绝读取可能包含凭据的文件：{relative}")

        try:
            with target.open("rb") as handle:
                # 多读一个字节用来判断"是否还有更多"，避免读完整个大文件
                data = handle.read(MAX_READ_BYTES + 1)
        except PermissionError:
            return _fail(self.spec.name, f"没有读取权限：{relative}")
        except OSError:
            return _fail(self.spec.name, f"读取失败：{relative}")

        truncated = len(data) > MAX_READ_BYTES
        data = data[:MAX_READ_BYTES]
        if b"\x00" in data:
            return _fail(self.spec.name, f"拒绝读取二进制文件：{relative}")

        text = data.decode("utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS]
            truncated = True
        return _ok(self.spec.name, text, truncated=truncated)


def build_default_registry() -> ToolRegistry:
    """构造注册了全部只读工具的注册表。

    每次调用都返回新实例：注册表本身不可变，但工具对象没有共享状态，
    反复构造的成本可以忽略，而共享一个全局单例会引入没必要的测试耦合。
    """
    tools: list[AgentTool] = [
        InspectProjectTool(),
        SearchCodeTool(),
        ReadFileTool(),
        InspectAPITool(),
    ]
    return ToolRegistry(tools)


# 供调用方按需取用例，避免在别处硬编码工具名
__all__ = [
    "MAX_INSPECT_CHARS",
    "MAX_READ_CHARS",
    "MAX_READ_BYTES",
    "MAX_SEARCH_CHARS",
    "MAX_SEARCH_RESULTS",
    "InspectProjectTool",
    "ReadFileTool",
    "SearchCodeTool",
    "build_default_registry",
]
