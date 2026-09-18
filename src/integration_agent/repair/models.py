"""Repair Loop 阶段的数据契约：RepairPlan / RepairAction / RepairApplicationResult。

设计约束：
    - 全字段强类型化，后续 ApplyRepair 阶段可直接消费，无需解析自由文本。
    - RepairAction 只描述"改哪里、为什么、怎么改"，本阶段绝不直接落盘。
    - confidence 限定在 [0, 1]；iteration / max_iterations 有界且交叉校验。
    - 保证 Pydantic JSON round-trip。
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from integration_agent.generation import GeneratedArtifacts
from integration_agent.validation import TestResult

FailureCategory = Literal[
    "assertion",
    "import",
    "dependency",
    "collection",
    "timeout",
    "environment",
    "unknown",
    "none",
]


class RepairAction(BaseModel):
    """一个具体、有限、可审计的修复动作（描述性，不修改任何文件）。"""

    file: str  # 待修改文件；无法从证据确定时为 ""，由计划层在 warnings 中说明
    action: Literal["modify", "create"]
    reason: str
    target: str | None = None  # 修改目标位置描述，如 "import section" / "line 12"
    changes: str  # 修改内容描述，如 "Add the missing module import ..."
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class RepairPlan(BaseModel):
    """RepairPlanner 的完整输出：为什么修复、修复什么、如何有限地修复。"""

    should_repair: bool
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=3, gt=0)
    failure_category: FailureCategory = "none"
    analysis: str = ""
    actions: list[RepairAction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RepairPlan":
        if self.iteration > self.max_iterations:
            raise ValueError(
                f"iteration ({self.iteration}) 不能大于 max_iterations ({self.max_iterations})"
            )
        if not self.should_repair and self.actions:
            raise ValueError("should_repair=False 时 actions 必须为空")
        return self


class RepairApplicationResult(BaseModel):
    """RepairApplier 的输出：修复后的产物 + 应用/跳过审计。"""

    artifacts: GeneratedArtifacts
    applied_actions: list[RepairAction] = Field(default_factory=list)
    skipped_actions: list[RepairAction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    changed: bool = False  # 是否存在真正被应用的修改


class RepairLoopResult(BaseModel):
    """RepairLoop 编排器的最终输出。"""

    status: Literal["passed", "max_iterations", "no_progress", "not_repairable", "error"]
    artifacts: GeneratedArtifacts  # 循环结束时的最新产物（初始 artifacts 永不被修改）
    test_result: TestResult | None = None  # 结束时的最后一次测试结果
    iterations: int = Field(default=0, ge=0)  # 实际执行的修复次数（初始测试不算）
    repair_plans: list[RepairPlan] = Field(default_factory=list)
    application_results: list[RepairApplicationResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None  # status="error" 时的异常说明（完整 traceback 保留在 warnings 中）


class LLMFileChange(BaseModel):
    """LLM 输出的一条文件修改（严格 JSON 契约的一部分）。

    modify：content 是修改后的完整文件内容；create：content 是新文件的完整内容。
    path 的安全校验（相对路径、禁止越界）在应用阶段逐条执行，失败只跳过不崩溃。
    """

    path: str
    action: Literal["modify", "create"]
    content: str
    reason: str = ""


class LLMRepairResponse(BaseModel):
    """LLM 修复响应的严格结构：非法 JSON / 缺字段 / 类型错误都会被拒绝。"""

    changes: list[LLMFileChange] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
