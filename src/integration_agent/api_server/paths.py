"""路径安全校验：HTTP 参数不允许访问 examples/ 之外的任何位置。

规则：
    - 只接受相对于 ALLOWED_ROOT（examples/）的相对路径。
    - 拒绝绝对路径（/xxx）、Windows 盘符（C:\\xxx）、UNC 路径。
    - 拒绝含 ".." 的越界片段。
    - 最终路径经 resolve()（解析符号链接）后必须仍位于 ALLOWED_ROOT 内，
      防止符号链接逃逸。
"""

import re
from pathlib import Path

_WINDOWS_ABSOLUTE = re.compile(r"^[a-zA-Z]:")

# 允许访问的根目录：项目 examples/（api_server/paths.py → parents[3] 为项目根）
ALLOWED_ROOT = (Path(__file__).resolve().parents[3] / "examples").resolve()


def resolve_allowed(relative: str) -> tuple[Path | None, str | None]:
    """把用户提交的路径安全解析到允许范围内；返回 (路径, 错误消息)。"""
    normalized = relative.replace("\\", "/").strip()
    if not normalized:
        return None, "路径为空"
    if normalized.startswith("/") or _WINDOWS_ABSOLUTE.match(normalized):
        return None, f"路径必须是相对于 examples/ 的相对路径：{relative!r}"
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None, f"路径包含越界片段：{relative!r}"
    candidate = (ALLOWED_ROOT.joinpath(*parts)).resolve()
    if not candidate.is_relative_to(ALLOWED_ROOT):
        return None, f"路径超出允许范围（examples/）：{relative!r}"
    return candidate, None
