"""Test Runner 阶段的输出契约：TestResult 及 FailureDetail。

设计约束：
    - 全部核心字段强类型化，Repair Loop 可直接消费，无需解析自由文本。
    - status 四态明确区分：passed / failed（测试失败）/ error（环境级失败）/
      timeout（超时终止）。
    - failure_details 携带可定位的结构化失败信息（名称/文件/行号/消息/回溯）。
    - stdout / stderr 只作为完整原始输出保留，核心结论一律走结构化字段。
"""

from typing import Literal

from pydantic import BaseModel, Field


class FailureDetail(BaseModel):
    """一个失败/错误测试的结构化信息。"""

    test_name: str
    file: str | None = None  # 测试文件相对路径（相对测试工作区）
    line: int | None = None
    message: str = ""  # pytest short summary 中的失败原因摘要
    traceback: str | None = None  # --tb=short 输出的回溯段


class TestResult(BaseModel):
    """一次测试运行的结构化结果。"""

    status: Literal["passed", "failed", "error", "timeout"]
    exit_code: int | None = None  # timeout / 环境级失败时为 None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0  # 秒；只记录耗时，不含随机信息
    stdout: str = ""
    stderr: str = ""
    failure_details: list[FailureDetail] = Field(default_factory=list)
    dependency_warnings: list[str] = Field(default_factory=list)  # 未满足的依赖说明
