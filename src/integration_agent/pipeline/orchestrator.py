"""End-to-End Pipeline 编排器：把 APIForge 各阶段串成完整工作流。

契约：
    OpenAPI Spec + Repository
        ↓
    parse_openapi → APIInfo
        ↓
    scan_repository → ProjectStructure
        ↓
    IntegrationPlanner.plan → IntegrationPlan
        ↓
    CodeGenerator.generate → GeneratedArtifacts
        ↓
    RepairLoop.run（内部：TestRunner ↔ RepairPlanner ↔ RepairApplier 循环）
        ↓
    PatchGenerator.generate（纯函数，可选 original_files）
        ↓
    PipelineResult（含 patch: PatchResult）

职责边界：
    - Pipeline 只做编排与装配，不复制任何已有模块的逻辑。
    - 全部组件依赖注入（Planner / CodeGenerator / TestRunner / RepairPlanner /
      RepairApplier），默认使用现有确定性实现（离线、零副作用）。
    - LLM 装配示例：
        run_pipeline(spec, repo,
                     repair_applier=StructuredLLMRepairApplier(
                         DeepSeekLLMClient(json_mode=True)))
    - 不修改真实 repository、不执行 git、不安装依赖、不读取/输出 API Key。

状态映射：
    RepairLoop.status == "passed"                  → Pipeline "passed"
    max_iterations / no_progress / not_repairable / error → Pipeline "tests_failed"
    前置阶段（parse/scan/plan/generate）异常           → Pipeline "error"（failed_stage 定位）
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from integration_agent.agent import DeterministicPlanner, IntegrationPlanner, PlannerState
from integration_agent.api import APIInfo, parse_openapi
from integration_agent.generation import CodeGenerator, DeterministicCodeGenerator
from integration_agent.patch import DeterministicPatchGenerator, PatchGenerator
from integration_agent.pipeline.models import PipelineResult
from integration_agent.repair import (
    DeterministicRepairApplier,
    DeterministicRepairPlanner,
    RepairApplier,
    RepairPlanner,
    run_repair_loop,
)
from integration_agent.repository import ProjectStructure, scan_repository
from integration_agent.validation import DeterministicTestRunner, TestRunner

# 阶段异常的服务端日志出口：完整 traceback 只写这里，不进 API 响应。
# 未配置 logging 时，Python 的 lastResort handler 会把 ERROR 及以上打到 stderr，
# 因此服务端控制台始终能看到失败详情，无需额外配置。
_logger = logging.getLogger(__name__)


@runtime_checkable
class Pipeline(Protocol):
    """Pipeline 接口：确定性实现与未来扩展共用同一契约。"""

    def run(
        self,
        spec_source: str | Path,
        repo_root: str | Path,
        *,
        request: str = "",
        max_iterations: int = 3,
    ) -> PipelineResult:
        """执行完整集成流程并返回最终结果。"""
        ...


def run_pipeline(
    spec_source: str | Path,
    repo_root: str | Path,
    *,
    request: str = "",
    max_iterations: int = 3,
    planner: IntegrationPlanner | None = None,
    code_generator: CodeGenerator | None = None,
    test_runner: TestRunner | None = None,
    repair_planner: RepairPlanner | None = None,
    repair_applier: RepairApplier | None = None,
    patch_generator: PatchGenerator | None = None,
    original_files: Mapping[str, str] | None = None,
) -> PipelineResult:
    """便捷入口：默认使用全套确定性组件，可覆盖任意单个组件。

    original_files：modify 文件的原始内容（path → content），由上层显式提供；
    不提供时 modify 片段仍会保留结构化信息，只是 diff_available=False。
    """
    return IntegrationPipeline(
        planner=planner,
        code_generator=code_generator,
        test_runner=test_runner,
        repair_planner=repair_planner,
        repair_applier=repair_applier,
        patch_generator=patch_generator,
    ).run(
        spec_source,
        repo_root,
        request=request,
        max_iterations=max_iterations,
        original_files=original_files,
    )


class IntegrationPipeline:
    """依赖注入的确定性 Pipeline 实现。"""

    def __init__(
        self,
        *,
        planner: IntegrationPlanner | None = None,
        code_generator: CodeGenerator | None = None,
        test_runner: TestRunner | None = None,
        repair_planner: RepairPlanner | None = None,
        repair_applier: RepairApplier | None = None,
        patch_generator: PatchGenerator | None = None,
    ) -> None:
        self.planner = planner or DeterministicPlanner()
        self.code_generator = code_generator or DeterministicCodeGenerator()
        self.test_runner = test_runner or DeterministicTestRunner()
        # repair_planner 的默认值依赖每次运行的 max_iterations（保持迭代上限一致），
        # 因此在 run() 中解析。
        self.repair_planner = repair_planner
        self.repair_applier = repair_applier or DeterministicRepairApplier()
        self.patch_generator = patch_generator or DeterministicPatchGenerator()

    def run(
        self,
        spec_source: str | Path,
        repo_root: str | Path,
        *,
        request: str = "",
        max_iterations: int = 3,
        original_files: Mapping[str, str] | None = None,
    ) -> PipelineResult:
        warnings: list[str] = []

        # 1. API Understanding
        try:
            api: APIInfo | None = parse_openapi(spec_source)
        except Exception as exc:  # noqa: BLE001 - 阶段异常统一转为 error 结果
            return self._stage_error("parse", None, None, None, None, warnings, exc)

        # 2. Repository Understanding
        try:
            project: ProjectStructure | None = scan_repository(repo_root)
        except Exception as exc:  # noqa: BLE001
            return self._stage_error("scan", api, None, None, None, warnings, exc)

        # 3. Integration Planning
        try:
            plan = self.planner.plan(PlannerState(api=api, project=project, request=request))
        except Exception as exc:  # noqa: BLE001
            return self._stage_error("plan", api, project, None, None, warnings, exc)

        # 4. Code Generation
        try:
            artifacts = self.code_generator.generate(plan)
        except Exception as exc:  # noqa: BLE001
            return self._stage_error("generate", api, project, plan, None, warnings, exc)

        # 5. Repair Loop（初始测试 + 修复循环都在其中）
        repair_planner = self.repair_planner or DeterministicRepairPlanner(
            max_iterations=max_iterations
        )
        loop_result = run_repair_loop(
            artifacts,
            max_iterations=max_iterations,
            test_runner=self.test_runner,
            repair_planner=repair_planner,
            repair_applier=self.repair_applier,
        )
        warnings.extend(loop_result.warnings)

        # 6. Final Patch / Diff（纯函数组件；失败不改变 status，只记录 warning）
        try:
            patch_result = self.patch_generator.generate(
                loop_result.artifacts, original_files=original_files
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Patch 生成失败：{type(exc).__name__}: {exc}")
            patch_result = None

        status = "passed" if loop_result.status == "passed" else "tests_failed"
        return PipelineResult(
            status=status,
            api=api,
            project=project,
            plan=plan,
            artifacts=loop_result.artifacts,
            initial_test_result=None,  # 当前 RepairLoop 不保存初始测试结果，不重复跑测试
            repair_loop_result=loop_result,
            patch=patch_result,
            warnings=warnings,
        )

    @staticmethod
    def _stage_error(
        stage: str,
        api: APIInfo | None,
        project: ProjectStructure | None,
        plan,
        artifacts,
        warnings: list[str],
        exc: Exception,
    ) -> PipelineResult:
        """前置阶段异常 → status="error"，保留已产生的上游结果。

        诊断信息的分流（安全边界）：
            - **完整 traceback 只进服务端日志**。PipelineResult 是 FastAPI 的
              response_model，它会原样序列化给浏览器——traceback 里带着绝对路径、
              模块名、源码行，异常消息也可能夹带本地路径，绝不能出现在响应里。
            - 客户端只拿到 stage 级的安全说明；失败原因由 failed_stage + error
              表达，details 去服务端日志取。
        """
        _logger.error("Pipeline 阶段失败：stage=%s", stage, exc_info=exc)
        error = f"Integration failed during {stage}."
        warnings.append(f"{stage} 阶段失败（详情见服务端日志）")
        return PipelineResult(
            status="error",
            api=api,
            project=project,
            plan=plan,
            artifacts=artifacts,
            failed_stage=stage,
            warnings=warnings,
            error=error,
        )
