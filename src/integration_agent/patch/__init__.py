"""Patch 阶段：把最终 GeneratedArtifacts 变成结构化、可审计的 PatchResult。

主要 API：
    DeterministicPatchGenerator().generate(artifacts, original_files=...)  —— 纯函数生成
    PatchGenerator                                  —— Protocol，未来扩展共用同一契约

输出契约：
    PatchResult / PatchFile / PatchSummary

约束：纯函数——不修改真实 repository、不执行 git / shell / subprocess、
不访问网络、不读取 secrets、不自行扫描仓库；原始文件内容由调用方显式提供。
"""

from integration_agent.patch.generator import (
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_MAX_FILES,
    DeterministicPatchGenerator,
    PatchGenerator,
)
from integration_agent.patch.models import PatchFile, PatchResult, PatchSummary

__all__ = [
    "DEFAULT_MAX_CONTENT_CHARS",
    "DEFAULT_MAX_FILES",
    "DeterministicPatchGenerator",
    "PatchFile",
    "PatchGenerator",
    "PatchResult",
    "PatchSummary",
]
