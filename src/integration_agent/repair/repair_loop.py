"""Repair Loop 编排器：把 TestRunner / RepairPlanner / RepairApplier 编排成受控循环。

契约：GeneratedArtifacts → RepairLoopResult。

迭代语义（无 off-by-one）：
    max_iterations = 3 表示最多执行 3 次"修复"；初始测试不算修复。

    Initial Test
        ↓
    Repair #1 → Test
        ↓
    Repair #2 → Test
        ↓
    Repair #3 → Test → Stop（max_iterations）

终止条件（按优先级）：
    1. 测试通过（status="passed"）
    2. RepairPlan.should_repair == False（status="not_repairable"）
    3. RepairApplicationResult.changed == False（status="no_progress"，防止重复同一计划）
    4. 已达到 max_iterations 次修复（status="max_iterations"）
    5. 任一组件抛出未预期异常（status="error"，完整 traceback 只写服务端日志）

安全边界：
    - 不修改真实 repository、不执行 shell / git / pytest（pytest 由注入的
      TestRunner 在临时工作区执行）、不发网络请求、不安装依赖。
    - 初始 artifacts 永不被修改：循环从深拷贝开始，Applier 同样只产出新对象。
    - 只依赖 Protocol（TestRunner / RepairPlanner / RepairApplier），
      不关心实现是 deterministic 还是未来的 LLM。
    - RepairLoopResult 会经 PipelineResult 序列化给客户端，因此其中不得出现
      traceback / 绝对路径：诊断信息只进服务端日志。
"""

import logging
import time
from typing import Protocol, runtime_checkable

from integration_agent.generation import GeneratedArtifacts
from integration_agent.repair.models import (
    RepairApplicationResult,
    RepairLoopResult,
    RepairPlan,
)
from integration_agent.repair.repair_applier import (
    DeterministicRepairApplier,
    RepairApplier,
)
from integration_agent.repair.repair_planner import (
    DEFAULT_MAX_ITERATIONS,
    DeterministicRepairPlanner,
    RepairPlanner,
)
from integration_agent.trace import emit
from integration_agent.validation import DeterministicTestRunner, TestResult, TestRunner

# 组件异常的服务端日志出口：完整 traceback 只写这里，不进 API 响应。
_logger = logging.getLogger(__name__)


@runtime_checkable
class RepairLoop(Protocol):
    """RepairLoop 接口：确定性编排与未来扩展共用同一契约。"""

    def run(self, artifacts: GeneratedArtifacts) -> RepairLoopResult:
        """执行修复循环并返回最终结果。"""
        ...


def run_repair_loop(
    artifacts: GeneratedArtifacts,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    test_runner: TestRunner | None = None,
    repair_planner: RepairPlanner | None = None,
    repair_applier: RepairApplier | None = None,
) -> RepairLoopResult:
    """便捷入口：默认使用全套确定性组件。"""
    runner = test_runner or DeterministicTestRunner()
    planner = repair_planner or DeterministicRepairPlanner(max_iterations=max_iterations)
    applier = repair_applier or DeterministicRepairApplier()
    return RepairLoopRunner(
        test_runner=runner,
        repair_planner=planner,
        repair_applier=applier,
        max_iterations=max_iterations,
    ).run(artifacts)


class RepairLoopRunner:
    """把注入的 Runner / Planner / Applier 编排成受控的修复循环。"""

    def __init__(
        self,
        *,
        test_runner: TestRunner,
        repair_planner: RepairPlanner,
        repair_applier: RepairApplier,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError(f"max_iterations 必须大于 0，实际为 {max_iterations}")
        self.test_runner = test_runner
        self.repair_planner = repair_planner
        self.repair_applier = repair_applier
        self.max_iterations = max_iterations

    def run(self, artifacts: GeneratedArtifacts) -> RepairLoopResult:
        current = artifacts.model_copy(deep=True)  # 初始 artifacts 永不被修改
        plans: list[RepairPlan] = []
        applications: list[RepairApplicationResult] = []
        warnings: list[str] = []

        for repair_count in range(self.max_iterations + 1):
            # 1. 运行测试
            try:
                test_result = self.test_runner.run(current)
            except Exception as exc:  # noqa: BLE001 - 任意组件异常都必须转 error 而不是崩溃
                return self._error_result(
                    current,
                    plans,
                    applications,
                    warnings,
                    repair_count,
                    None,
                    "TestRunner",
                    exc,
                )
            if test_result.status == "passed":
                return RepairLoopResult(
                    status="passed",
                    artifacts=current,
                    test_result=test_result,
                    iterations=repair_count,
                    repair_plans=plans,
                    application_results=applications,
                    warnings=warnings,
                )
            if repair_count >= self.max_iterations:
                warnings.append("Maximum repair iterations reached.")
                return RepairLoopResult(
                    status="max_iterations",
                    artifacts=current,
                    test_result=test_result,
                    iterations=repair_count,
                    repair_plans=plans,
                    application_results=applications,
                    warnings=warnings,
                )

            # 2. 规划修复（把当前轮次传给支持 iteration 状态的 planner）
            # 埋点从"确定要修一次"开始：测试通过就不进这一支，也就没有修复这回事。
            attempt = repair_count + 1
            repair_started = time.perf_counter()
            emit(
                "repair",
                "repair_started",
                "开始一次修复",
                metadata={
                    "attempt": attempt,
                    "max_iterations": self.max_iterations,
                    "test_status": test_result.status,
                    "failed": test_result.failed,
                    "errors": test_result.errors,
                },
            )
            if hasattr(self.repair_planner, "iteration"):
                self.repair_planner.iteration = repair_count
            try:
                plan = self.repair_planner.plan(current, test_result)
            except Exception as exc:  # noqa: BLE001
                return self._error_result(
                    current,
                    plans,
                    applications,
                    warnings,
                    repair_count,
                    test_result,
                    "RepairPlanner",
                    exc,
                )
            plans.append(plan)
            if not plan.should_repair:
                warnings.extend(plan.warnings)
                return RepairLoopResult(
                    status="not_repairable",
                    artifacts=current,
                    test_result=test_result,
                    iterations=repair_count,
                    repair_plans=plans,
                    application_results=applications,
                    warnings=warnings,
                )

            # 3. 应用修复（声明 accepts_test_result 的 Applier 会额外收到 TestResult）
            try:
                application = _apply(self.repair_applier, current, plan, test_result)
            except Exception as exc:  # noqa: BLE001
                return self._error_result(
                    current,
                    plans,
                    applications,
                    warnings,
                    repair_count,
                    test_result,
                    "RepairApplier",
                    exc,
                )
            applications.append(application)
            warnings.extend(application.warnings)
            # 只记"改了几个文件、动了几条动作"，**不记 patch 内容**：
            # 修复片段是完整代码，trace 里该有的是这次修复的形状。
            emit(
                "repair",
                "repair_completed",
                "一次修复已应用",
                metadata={
                    "attempt": attempt,
                    "status": "applied" if application.changed else "no_progress",
                    "changed": application.changed,
                    "files_changed": len(
                        {action.file for action in application.applied_actions if action.file}
                    ),
                    "applied_actions": len(application.applied_actions),
                    "skipped_actions": len(application.skipped_actions),
                },
                duration=time.perf_counter() - repair_started,
                status="applied" if application.changed else "no_progress",
            )
            if not application.changed:
                # 有修复建议但应用层无法产生代码变化：停止，避免无限重复同一计划
                return RepairLoopResult(
                    status="no_progress",
                    artifacts=current,
                    test_result=test_result,
                    iterations=repair_count + 1,
                    repair_plans=plans,
                    application_results=applications,
                    warnings=warnings,
                )
            current = application.artifacts

        # 兜底：循环内每轮都会 return，此处仅防止未来修改引入无限循环
        warnings.append("Maximum repair iterations reached.")
        return RepairLoopResult(
            status="max_iterations",
            artifacts=current,
            test_result=None,
            iterations=self.max_iterations,
            repair_plans=plans,
            application_results=applications,
            warnings=warnings,
        )

    @staticmethod
    def _error_result(
        current: GeneratedArtifacts,
        plans: list[RepairPlan],
        applications: list[RepairApplicationResult],
        warnings: list[str],
        iterations: int,
        test_result: TestResult | None,
        component: str,
        exc: Exception,
    ) -> RepairLoopResult:
        # 组件异常同样走"日志留全量、响应只留安全说明"：RepairLoopResult 会被
        # PipelineResult 带到浏览器，异常消息本身也可能夹带本地绝对路径。
        _logger.error("RepairLoop 组件异常：component=%s", component, exc_info=exc)
        error = f"Repair loop failed during {component}."
        warnings.append(f"未预期异常（{component}），详情见服务端日志")
        return RepairLoopResult(
            status="error",
            artifacts=current,
            test_result=test_result,
            iterations=iterations,
            repair_plans=plans,
            application_results=applications,
            warnings=warnings,
            error=error,
        )


def _apply(applier, artifacts, plan, test_result):
    """调用 Applier：声明 accepts_test_result 的 Applier（如 LLMRepairApplier）
    会额外收到当前 TestResult；其余保持 2 参调用（向后兼容）。"""
    if getattr(applier, "accepts_test_result", False):
        return applier.apply(artifacts, plan, test_result)
    return applier.apply(artifacts, plan)
