"""Code Generator 阶段的输出契约：GeneratedArtifacts 及子模型。

设计约束：
    - 全部核心字段强类型化，Patch / Validation 阶段无需解析自由文本。
    - action 显式区分 create / modify：
        create 的 content 是完整文件内容，可直接落盘；
        modify 的 content 是待插入/替换的片段，配合 insertion_point 与
        changes 由 Patch 阶段应用，绝不重写整个既有文件。
    - 生成器是纯函数：只产出数据，不触碰磁盘。
"""

from typing import Literal

from pydantic import BaseModel, Field


class GeneratedFile(BaseModel):
    """一个生成产物文件。"""

    path: str
    action: Literal["create", "modify"]
    content: str  # create: 完整文件内容；modify: 待插入的片段
    purpose: str = ""  # 该文件的用途（来自 IntegrationPlan 或 TestingStrategy）
    insertion_point: str | None = None  # modify：插入位置锚点，如 "[project] 的 dependencies"
    changes: list[str] = Field(default_factory=list)  # modify：改动要点说明


class DependencyChange(BaseModel):
    """一项依赖变更；当前阶段只支持新增（add）。"""

    name: str
    version: str | None = None
    action: Literal["add"] = "add"
    reason: str = ""


class GeneratedArtifacts(BaseModel):
    """Code Generator 的完整输出：可被 Patch / Validation 阶段直接消费。"""

    files: list[GeneratedFile] = Field(default_factory=list)
    dependency_changes: list[DependencyChange] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)

    @property
    def created_files(self) -> list[GeneratedFile]:
        """action="create" 的文件（完整内容，可直接落盘）。"""
        return [item for item in self.files if item.action == "create"]

    @property
    def modified_files(self) -> list[GeneratedFile]:
        """action="modify" 的文件（结构化修改建议）。"""
        return [item for item in self.files if item.action == "modify"]
