"""Benchmark 数据模型：Case / Result / Summary。

设计约束：
    - Case 描述"一个固定的 API 集成任务"，Result 是"那次真实执行的结构化结论"，
      Summary 是"一轮 Benchmark 的汇总指标"。
    - 所有指标（pass_rate / repair_trigger_rate / repair_recovery_rate / 平均值）
      都是 Summary 的**计算属性**，从 results 动态得出——本模块里没有任何一个
      写死的成功率数字。
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BenchmarkCase(BaseModel):
    """一个固定的 API 集成任务。

    spec_path / project_path 是相对 examples/ 的路径，运行期按 api_server.paths
    的同一套规则解析（拒绝绝对路径与越界片段）。

    三个执行旋钮各对应默认套件里的一个 case，不为想象中的未来做通用执行框架：
        sabotage      注入确定性失败（复用 api_server.demo.SabotagedGenerator）
        max_endpoints 换用更小/更大的 DeterministicPlanner 端点数
        repeat        同一 case 连续执行次数（deterministic_repeat 用 2）
    """

    name: str
    description: str
    spec_path: str
    project_path: str
    request: str = ""
    expected_outcome: Literal["passed", "failed"]
    sabotage: bool = False
    max_endpoints: int | None = None
    repeat: int = Field(default=1, ge=1)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name 不能为空")
        return value


class BenchmarkResult(BaseModel):
    """一个 case 真实执行后的结构化结论。

    除 case_name / status / duration 外，每个数字都来自那次运行的 Execution Trace
    （test_completed / repair_started / generation_completed / patch_generated），
    不是常量也不是推算值。
    """

    case_name: str
    status: Literal["passed", "failed", "error"]
    duration: float  # 秒；repeat > 1 时是各次运行的平均值
    test_passed: int
    test_failed: int
    test_errors: int
    repair_attempts: int  # 触发的修复尝试次数（repair_started 事件数）
    patch_files: int  # 最终 patch 涉及的文件数
    generated_source_count: int = 0
    generated_test_count: int = 0


class BenchmarkSummary(BaseModel):
    """一轮 Benchmark 的汇总。计数是字段，比率是计算属性（不写死、不冗余存储）。"""

    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int

    repair_triggered: int  # repair_attempts > 0 的 case 数
    repair_recovered: int  # 其中最终通过的 case 数

    average_duration: float
    average_repair_attempts: float

    results: list[BenchmarkResult]

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases else 0.0

    @property
    def repair_trigger_rate(self) -> float:
        return self.repair_triggered / self.total_cases if self.total_cases else 0.0

    @property
    def repair_recovery_rate(self) -> float:
        if not self.repair_triggered:
            return 0.0
        return self.repair_recovered / self.repair_triggered


__all__ = ["BenchmarkCase", "BenchmarkResult", "BenchmarkSummary"]
