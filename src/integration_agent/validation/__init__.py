"""Validation 阶段：在隔离的临时工作区运行生成产物的测试并返回结构化结果。

主要 API：
    run_tests(artifacts)                          —— 便捷入口
    DeterministicTestRunner().run(artifacts)      —— 可配置实现
    TestRunner                                    —— Protocol，未来实现共用同一契约
    OfflineDependencyPreparer                     —— 默认依赖策略（离线检查，不联网安装）

输出契约：
    TestResult / FailureDetail

约束：只操作临时目录——不修改真实 Repository、不写真实项目清单、
不联网安装依赖、不修复失败代码（Repair Loop 是下一阶段）。
"""

from integration_agent.validation.models import FailureDetail, TestResult
from integration_agent.validation.test_runner import (
    DEFAULT_TIMEOUT,
    DependencyPreparer,
    DeterministicTestRunner,
    OfflineDependencyPreparer,
    TestRunner,
    run_tests,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "DependencyPreparer",
    "DeterministicTestRunner",
    "FailureDetail",
    "OfflineDependencyPreparer",
    "TestResult",
    "TestRunner",
    "run_tests",
]
