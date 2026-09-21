"""Validation 阶段：在隔离的临时工作区运行生成产物的测试并返回结构化结果。

主要 API：
    run_tests(artifacts)                          —— 便捷入口
    DeterministicTestRunner().run(artifacts)      —— 可配置实现
    TestRunner                                    —— Protocol，未来实现共用同一契约
    OfflineDependencyPreparer                     —— 默认依赖策略（离线检查，不联网安装）
    build_subprocess_env(workspace)               —— pytest 子进程的最小环境（默认不继承）

输出契约：
    TestResult / FailureDetail

约束：只操作临时目录——不修改真实 Repository、不写真实项目清单、
不联网安装依赖、不修复失败代码（Repair Loop 是下一阶段）；
跑 pytest 的子进程只拿到 allowlist 里的无凭证环境变量。
"""

from integration_agent.validation.models import FailureDetail, TestResult
from integration_agent.validation.test_runner import (
    DEFAULT_TIMEOUT,
    NO_TESTS_EXECUTED_MESSAGE,
    NO_TESTS_FAILURE_NAME,
    SUBPROCESS_ENV_ALLOWLIST,
    DependencyPreparer,
    DeterministicTestRunner,
    OfflineDependencyPreparer,
    TestRunner,
    build_subprocess_env,
    run_tests,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "NO_TESTS_EXECUTED_MESSAGE",
    "NO_TESTS_FAILURE_NAME",
    "SUBPROCESS_ENV_ALLOWLIST",
    "DependencyPreparer",
    "DeterministicTestRunner",
    "FailureDetail",
    "OfflineDependencyPreparer",
    "TestResult",
    "TestRunner",
    "build_subprocess_env",
    "run_tests",
]
