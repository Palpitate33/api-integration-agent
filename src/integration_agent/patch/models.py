"""Patch 阶段的数据契约：PatchFile / PatchSummary / PatchResult。

设计约束：
    - 全字段强类型化，可 JSON round-trip，供未来 Backend API / Frontend 直接消费。
    - 不伪造数据：
        create 文件有完整 content → 生成真实 unified diff（diff_available=True）；
        modify 片段（带 insertion_point）无法安全推导完整新内容 →
        diff_available=False，保留 insertion_point / 片段 / changes 等结构化信息；
        modify 全文（无 insertion_point）且提供了原始内容 → 真实 diff。
    - path 安全校验与数量/大小上限在生成阶段执行（见 generator）。
"""

from typing import Literal

from pydantic import BaseModel, Field

from integration_agent.generation import DependencyChange


class PatchFile(BaseModel):
    """一个文件的最终修改结果。"""

    path: str
    action: Literal["create", "modify"]
    old_content: str | None = None  # modify 且调用方提供了原始内容
    new_content: str | None = None  # create 或 modify 全文模式的完整内容
    diff: str | None = None  # unified diff 文本（diff_available=True 时非空）
    diff_available: bool = False
    summary: str = ""  # 用途 / 插入点 / 片段摘要


class PatchSummary(BaseModel):
    """修改规模汇总。"""

    total_files: int = 0
    created: int = 0
    modified: int = 0
    dependencies: int = 0


class PatchResult(BaseModel):
    """最终修改结果：结构化 + 可审计。"""

    files: list[PatchFile] = Field(default_factory=list)
    dependency_changes: list[DependencyChange] = Field(default_factory=list)
    unified_diff: str = ""  # 全部可用 diff 的拼接（文件顺序 = artifacts 顺序）
    files_changed: int = 0
    files_created: int = 0
    files_modified: int = 0
    warnings: list[str] = Field(default_factory=list)
    summary: PatchSummary = Field(default_factory=PatchSummary)
