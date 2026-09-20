"""inspect_api：查看**已经解析好**的 API 信息。

只读 ``ToolContext.api``，不重新解析 OpenAPI 文档、不重新下载 spec、不访问网络。
解析是 api/parser.py 的职责，这里只负责把解析结果按需裁剪成适合进 LLM 上下文的形状。

安全边界：
    只读工具。不修改 APIInfo（连字段都不改写），不写文件、不执行 shell / git /
    子进程、不发网络请求、不读环境变量、不读 API Key。回给模型的内容里
    不含任何 json_schema 原文，也不含可能夹在 base_url 里的凭据段。

依赖方向：
    tools → api，不反向依赖 agent / repair / pipeline / api_server。

实现说明：
    这里的 ``_ReadOnlyTool`` 外壳与文本预算函数与 repository_tools.py 中的那份
    是同构的、刻意各留一份：两个工具模块保持互不依赖的平级关系，避免
    api_tools ←→ repository_tools 互相 import 成环（repository_tools 需要 import
    本模块来完成默认注册）。等出现第三个工具家族时，再把两者提取到
    tools/base.py 之类的公共模块更合适。
"""

import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from integration_agent.api import APIEndpoint, APIInfo
from integration_agent.tools.models import ToolResult, ToolSpec
from integration_agent.tools.registry import ToolContext

logger = logging.getLogger(__name__)

# 输出上限。与 repository_tools 的常量一样是天花板：模型只能收紧不能放宽。
MAX_SUMMARY_CHARS = 1200
MAX_ENDPOINT_LINES = 40
MAX_ENDPOINTS_CHARS = 4000
MAX_FIELD_CHARS = 120  # 单个字段（名称 / summary / operation_id）的字符上限
MAX_LIST_ITEMS = 15  # tags / parameters / responses 每行最多列几项
MAX_METHOD_INPUT_CHARS = 20
MAX_PATH_INPUT_CHARS = 300

_MODES = ("summary", "endpoints")

# 截断提示的预留方式见 _fit_under_cap
_TRUNCATION_NOTE = "\n... 输出已被截断，请缩小范围后重试"
_INTERNAL_ERROR = "工具内部错误，请换一种方式重试"


class _ArgRejected(ValueError):
    """入参不满足约束。消息可直接回给模型，不含服务器路径或异常原文。"""


def _ok(tool: str, content: str, *, truncated: bool = False) -> ToolResult:
    return ToolResult(
        call_id="", tool=tool, ok=True, content=content, truncated=truncated, chars=len(content)
    )


def _fail(tool: str, message: str) -> ToolResult:
    return ToolResult(call_id="", tool=tool, ok=False, content="", error=message, chars=0)


def _clip(text: str, limit: int) -> tuple[str, bool]:
    """按字符上限裁剪单个字段，返回 (文本, 是否被裁过)。

    返回布尔而不是只返回文本：字段被裁掉一半却把 ``truncated`` 留成 False，
    等于告诉模型"这就是全部内容"。凡是缩短过输出，就必须说清楚。
    """
    if len(text) <= limit:
        return text, False
    return text[:limit] + "…", True


def _join_items(items: list[str], *, cap: int = MAX_LIST_ITEMS) -> tuple[str, bool]:
    """把一小组条目拼成一行；超过 cap 项时说明还剩多少项。"""
    if len(items) <= cap:
        return ", ".join(items), False
    return f"{', '.join(items[:cap])} … 其余 {len(items) - cap} 项", True


def _safe_base_url(raw: str | None) -> str | None:
    """去掉 base_url 里的凭据段（``user:pass@``）后再回给模型。

    spec 的作者完全可能把 token 写进 servers[0].url。base_url 是 Agent 需要的
    信息，但凭据不是——所以只保留 scheme / host / port / path。
    """
    if not raw:
        return None
    parts = urlsplit(raw)
    if not (parts.username or parts.password):
        return raw
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _assemble(header: str, kept: list[str], note: str) -> str:
    """把标题与已保留的条目行拼起来。"""
    body = "\n".join(kept)
    return f"{header}\n{body}{note}" if body else f"{header}{note}"


def _fit_under_cap(header: str, lines: list[str], cap: int, *, note: str) -> tuple[str, bool]:
    """把标题与条目行拼进 cap 之内；返回 (文本, 是否因字符上限被截断)。

    note 的长度**预先从预算里扣掉**。若改成事后追加，"输出已被截断"这句话本身
    就会把结果顶出上限——一个自相矛盾的边界。
    """
    budget = cap - len(header) - 1 - len(note)  # 标题后面还有一个换行
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) + (1 if kept else 0)
        if used + cost > budget:
            return _assemble(header, kept, note), True
        kept.append(line)
        used += cost
    return _assemble(header, kept, ""), False


def _normalize_path(text: str) -> str:
    """去掉尾部斜杠（根路径 "/" 除外），让 /pets 与 /pets/ 视为同一路径。"""
    if len(text) > 1:
        return text.rstrip("/") or "/"
    return text


def _optional_text(args: dict[str, Any], key: str, max_chars: int) -> str | None:
    """取一个可选的非空字符串参数。"""
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _ArgRejected(f"{key} 必须是非空字符串")
    text = value.strip()
    if len(text) > max_chars:
        raise _ArgRejected(f"{key} 过长，最多 {max_chars} 个字符")
    return text


def _parse_mode(args: dict[str, Any]) -> str:
    """解析 mode；缺省（含显式 null）为 summary——输出最小的一档。"""
    raw = args.get("mode")
    if raw is None:
        return "summary"
    if not isinstance(raw, str) or not raw.strip():
        raise _ArgRejected("mode 必须是非空字符串")
    mode = raw.strip().lower()
    if mode not in _MODES:
        raise _ArgRejected('mode 只能是 "summary" 或 "endpoints"')
    return mode


def _parse_filters(args: dict[str, Any], mode: str) -> tuple[str | None, str | None]:
    """解析 method / path 过滤参数，返回 (大写 method, 归一化后的 path)。

    summary 模式下给了过滤参数会直接报错，而不是悄悄忽略：静默忽略会让模型
    以为自己拿到了过滤后的结果，这比报错更难排查。

    method / path 显式传 null 视同没传——null 不携带任何过滤意图，为此让模型
    重试一次没有收益。空字符串则仍然报错：那更像模板没填上，而不是"不筛选"。
    """
    method = _optional_text(args, "method", MAX_METHOD_INPUT_CHARS)
    path = _optional_text(args, "path", MAX_PATH_INPUT_CHARS)
    if mode == "summary" and (method or path):
        raise _ArgRejected('summary 模式不接受 method / path；要按端点过滤请用 mode="endpoints"')
    return (method.upper() if method else None), (_normalize_path(path) if path else None)


def _select(api: APIInfo, method: str | None, path: str | None) -> list[APIEndpoint]:
    """按 method / path 过滤端点，返回按 (path, method) 排序的结果。

    排序而不是沿用文档顺序：相关的端点（GET /pets 与 POST /pets）会挨在一起，
    而且输出稳定性来自内容本身，不依赖解析时的列表顺序。
    """
    wanted_path = _normalize_path(path) if path else None
    selected = [
        endpoint
        for endpoint in api.endpoints
        if (method is None or endpoint.method.upper() == method)
        and (wanted_path is None or _normalize_path(endpoint.path) == wanted_path)
    ]
    return sorted(selected, key=lambda endpoint: (endpoint.path, endpoint.method))


def _no_match_message(api: APIInfo, method: str | None, path: str | None) -> str:
    """构造没匹配到端点时的说明；回显的是模型自己给的过滤值。"""
    scope = " ".join(part for part in (method, path) if part)
    methods = sorted({endpoint.method.upper() for endpoint in api.endpoints})
    if methods:
        detail = f"可用方法：{', '.join(methods[:MAX_LIST_ITEMS])}"
    else:
        detail = "该 API 没有任何端点"
    return f"没有匹配 {scope} 的端点（{detail}）"


def _render_summary(api: APIInfo) -> tuple[str, bool]:
    """summary 模式：名称 / 版本 / base_url / 端点总数 / 支持的 HTTP 方法。"""
    methods = sorted({endpoint.method.upper() for endpoint in api.endpoints})
    fields = (
        ("name", api.name),
        ("version", api.version),
        ("base_url", _safe_base_url(api.base_url)),
        ("endpoints", str(len(api.endpoints))),
        ("methods", ", ".join(methods) if methods else None),
    )
    lines: list[str] = []
    clipped = False
    for label, value in fields:
        text = "(none)" if value is None else value
        if len(text) > MAX_FIELD_CHARS:
            text, cut = _clip(text, MAX_FIELD_CHARS)
            clipped = clipped or cut
        lines.append(f"{label}: {text}")
    rendered = "\n".join(lines)
    if len(rendered) > MAX_SUMMARY_CHARS:
        return rendered[:MAX_SUMMARY_CHARS], True
    return rendered, clipped


def _index_line(endpoint: APIEndpoint) -> tuple[str, bool]:
    """端点索引的一行：method path (operation_id) summary。"""
    line = f"- {endpoint.method} {endpoint.path}"
    clipped = False
    if endpoint.operation_id:
        operation_id, cut = _clip(endpoint.operation_id, MAX_FIELD_CHARS)
        line += f" ({operation_id})"
        clipped = clipped or cut
    if endpoint.summary:
        summary, cut = _clip(endpoint.summary, MAX_FIELD_CHARS)
        line += f" {summary}"
        clipped = clipped or cut
    return line, clipped


def _index_label(method: str | None, path: str | None) -> str:
    criteria = " ".join(part for part in (method, path) if part)
    return f"匹配 {criteria} 的端点" if criteria else "端点"


def _render_index(
    selected: list[APIEndpoint], method: str | None, path: str | None
) -> tuple[str, bool]:
    """endpoints 模式：端点索引。省略数量写在标题行，不会被字符上限先切掉。"""
    label = _index_label(method, path)
    total = len(selected)
    if total > MAX_ENDPOINT_LINES:
        omitted = total - MAX_ENDPOINT_LINES
        header = (
            f"{label} {total} 个，仅列前 {MAX_ENDPOINT_LINES} 个，"
            f"其余 {omitted} 个已省略（按 path 排序）:"
        )
        count_truncated = True
    else:
        header = f"{label} {total} 个（按 path 排序）:"
        count_truncated = False
    rendered = [_index_line(endpoint) for endpoint in selected[:MAX_ENDPOINT_LINES]]
    field_clipped = any(clipped for _, clipped in rendered)
    lines = [line for line, _ in rendered]
    text, cut = _fit_under_cap(header, lines, MAX_ENDPOINTS_CHARS, note=_TRUNCATION_NOTE)
    return text, count_truncated or field_clipped or cut


def _render_detail(endpoint: APIEndpoint) -> tuple[list[str], bool]:
    """单个端点的详情行。

    只给结构信息：参数名与位置、请求体类型、响应状态码。
    **刻意不含任何 json_schema 原文**——完整 schema 属于生成代码时才需要的东西，
    塞进 Agent 上下文只会挤掉真正有用的信息。
    """
    lines = [f"{endpoint.method} {endpoint.path}"]
    truncated = False

    if endpoint.operation_id:
        operation_id, cut = _clip(endpoint.operation_id, MAX_FIELD_CHARS)
        lines.append(f"operation_id: {operation_id}")
        truncated = truncated or cut
    if endpoint.summary:
        summary, cut = _clip(endpoint.summary, MAX_FIELD_CHARS)
        lines.append(f"summary: {summary}")
        truncated = truncated or cut
    if endpoint.tags:
        tags, cut = _join_items(list(endpoint.tags))
        lines.append(f"tags: {tags}")
        truncated = truncated or cut
    if endpoint.parameters:
        items = [
            f"{param.name} ({param.location}{', required' if param.required else ''})"
            for param in endpoint.parameters
        ]
        parameters, cut = _join_items(items)
        lines.append(f"parameters: {parameters}")
        truncated = truncated or cut
    if endpoint.request_body is not None:
        body = endpoint.request_body
        parts = [body.content_type or "(unknown content type)"]
        if body.required:
            parts.append("required")
        if body.schema_ref:
            parts.append(f"schema: {body.schema_ref}")
        lines.append(f"request_body: {', '.join(parts)}")
    if endpoint.responses:
        codes, cut = _join_items([response.status_code for response in endpoint.responses])
        lines.append(f"responses: {codes}")
        truncated = truncated or cut

    return lines, truncated


class _ReadOnlyTool:
    """只读工具外壳：把意外异常挡在工具边界之内。

    契约要求工具不向上抛异常，但兜住不等于掩盖：完整堆栈写进服务端日志，
    回给模型的只有一句话。异常原文带着本机绝对路径与源码行，绝不能进 LLM 上下文。
    """

    spec: ToolSpec

    def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            return self._run(args, ctx)
        except Exception:
            logger.exception("工具 %s 执行时发生未预期错误", self.spec.name)
            return _fail(self.spec.name, _INTERNAL_ERROR)

    def _run(self, args: dict, ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


class InspectAPITool(_ReadOnlyTool):
    """查看已经解析好的 API 信息：概要 / 端点索引 / 单个端点详情。"""

    spec = ToolSpec(
        name="inspect_api",
        description=(
            '查看本次集成目标 API 的已解析信息。mode="summary"（默认）返回名称、版本、'
            "base_url、端点总数与支持的 HTTP 方法；"
            'mode="endpoints" 返回端点索引（method / path / operation_id / summary）。'
            'mode="endpoints" 时可用 method / path 过滤，两者同时给出且命中时返回该端点详情。'
            "不返回完整的 request / response schema。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(_MODES),
                    "description": "summary（默认）返回 API 概要；endpoints 返回端点索引",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP 方法过滤，如 GET，大小写不敏感；仅 mode=endpoints 时可用",
                },
                "path": {
                    "type": "string",
                    "description": "端点路径过滤，如 /pets；仅 mode=endpoints 时可用",
                },
            },
        },
    )

    def _run(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            mode = _parse_mode(args)
            method, path = _parse_filters(args, mode)
        except _ArgRejected as exc:
            return _fail(self.spec.name, str(exc))

        api = ctx.api

        if mode == "summary":
            text, truncated = _render_summary(api)
            return _ok(self.spec.name, text, truncated=truncated)

        selected = _select(api, method, path)
        if not selected and (method or path):
            # 带过滤条件却一个都没命中，属于"问错了"，明确报错而不是回空列表
            return _fail(self.spec.name, _no_match_message(api, method, path))
        if not selected:
            # 没给过滤条件就只是"这个 API 没有端点"，不是问错了——照常返回空索引
            return _ok(self.spec.name, _render_index([], None, None)[0])

        if method and path and len(selected) == 1:
            lines, detail_cut = _render_detail(selected[0])
            text, cut = _fit_under_cap(
                "端点详情:", lines, MAX_ENDPOINTS_CHARS, note=_TRUNCATION_NOTE
            )
            return _ok(self.spec.name, text, truncated=detail_cut or cut)

        text, truncated = _render_index(selected, method, path)
        return _ok(self.spec.name, text, truncated=truncated)


__all__ = [
    "MAX_ENDPOINTS_CHARS",
    "MAX_ENDPOINT_LINES",
    "MAX_FIELD_CHARS",
    "MAX_SUMMARY_CHARS",
    "InspectAPITool",
]
