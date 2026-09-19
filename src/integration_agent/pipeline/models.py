"""Pipeline 阶段的输出契约：PipelineResult。

设计约束：
    - 全字段强类型化，可 JSON round-trip。
    - 前置阶段失败时允许后续产物为 None（包括 api / project），
      绝不为了类型完整性伪造对象。
    - status 三态：
        passed        测试全部通过（RepairLoop 返回 passed）
        tests_failed  流程走完但最终测试未通过（RepairLoop 的
                      max_iterations / no_progress / not_repairable / error 归一）
        error         Pipeline 自身的前置阶段（parse/scan/plan/generate）异常
"""

from typing import Literal

from pydantic import BaseModel, Field

from integration_agent.agent import IntegrationPlan
from integration_agent.api import APIInfo
from integration_agent.generation import GeneratedArtifacts
from integration_agent.repair import RepairLoopResult
from integration_agent.repository import ProjectStructure
from integration_agent.validation import TestResult


class PipelineResult(BaseModel):
    """End-to-End Pipeline 的最终输出。"""

    status: Literal["passed", "tests_failed", "error"]
    api: APIInfo | None = None  # 第 1 步产物；parse 失败时为 None
    project: ProjectStructure | None = None  # 第 2 步产物；scan 失败时为 None
    plan: IntegrationPlan | None = None
    artifacts: GeneratedArtifacts | None = None  # 修复后的最终产物
    initial_test_result: TestResult | None = None
    # 当前 RepairLoop 不保存初始测试结果，Pipeline 不为获取它而重复跑测试，
    # 因此该字段保持 None（除非将来 Loop 显式提供）。
    repair_loop_result: RepairLoopResult | None = None
    failed_stage: str | None = None  # status="error" 时定位：parse/scan/plan/generate
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None  # status="error" 时的异常说明（完整 traceback 在 warnings 中）
