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
    max_iterations / no_progress / not_repairable   → Pipeline "tests_failed"
    RepairLoop.status == "error"                   → Pipeline "error"（failed_stage="repair"）
    前置阶段（parse/scan/plan/generate）异常          → Pipeline "error"（failed_stage 定位）

    tests_failed 与 error 的区别是"谁坏了"：前者是测试确实跑了但没通过，
    后者是流程自身出错（测试结论不可信，也不代表代码有问题）。

Execution Trace：
    run() 接受可选的 trace=TraceCollector()。整条流程跑在同一个 trace 作用域里，
    各阶段（含 Planner / CodeGenerator / TestRunner / RepairLoop）在**自己的实现内部**
    用 trace.emit() 埋点——collector 因此不需要沿调用链往下传。
    默认 trace=None：不打开作用域，emit 是空操作，行为与引入 trace 之前一致。
    注意 trace= 不在 Pipeline Protocol 上：那是"编排契约"，而 trace 是旁路的
    **观测通道**，不该变成每个未来实现都必须接受的参数。
"""

import logging
import time
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
from integration_agent.trace import TraceCollector, TraceStage, emit, trace_scope
from integration_agent.validation import DeterministicTestRunner, TestRunner

# 阶段异常的服务端日志出口：完整 traceback 只写这里，不进 API 响应。
# 未配置 logging 时，Python 的 lastResort handler 会把 ERROR 及以上打到 stderr，
# 因此服务端控制台始终能看到失败详情，无需额外配置。
_logger = logging.getLogger(__name__)

# PipelineResult.failed_stage 的取值（parse / scan / plan / generate）是对外契约的
# 一部分，不能为了 trace 改名；这里只做一次"短名 → trace 阶段名"的翻译。
_STAGE_NAME: dict[str, TraceStage] = {
    "parse": "api_understanding",
    "scan": "repository_understanding",
    "plan": "planner",
    "generate": "generation",
}


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
    trace: TraceCollector | None = None,
) -> PipelineResult:
    """便捷入口：默认使用全套确定性组件，可覆盖任意单个组件。

    original_files：modify 文件的原始内容（path → content），由上层显式提供；
    不提供时 modify 片段仍会保留结构化信息，只是 diff_available=False。

    trace：可选的观测出口。不传（默认）时行为与没有 trace 时**完全一致**——
    收集器只是旁路，不参与任何阶段的输入、输出与判定。
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
        trace=trace,
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
        trace: TraceCollector | None = None,
    ) -> PipelineResult:
        """执行完整集成流程。

        trace 是**观测**出口：传了就收集，不传（默认）时行为与之前一字不差。
        整个流程跑在同一个 trace 作用域里，各阶段因此可以自己用 emit() 埋点，
        不必把 collector 层层往下传，各阶段组件的签名也就不用为了埋点而改动。
        """
        with trace_scope(trace):
            return self._run(
                spec_source,
                repo_root,
                request=request,
                max_iterations=max_iterations,
                original_files=original_files,
            )

    def _run(
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
        started = time.perf_counter()
        emit("api_understanding", "stage_started", "开始解析 OpenAPI Spec")
        try:
            api: APIInfo | None = parse_openapi(spec_source)
        except Exception as exc:  # noqa: BLE001 - 阶段异常统一转为 error 结果
            return self._stage_error("parse", None, None, None, None, warnings, exc)
        emit(
            "api_understanding",
            "stage_completed",
            "OpenAPI Spec 解析完成",
            metadata={"endpoints": len(api.endpoints)},  # 不记 base_url：URL 里可能带凭据
            duration=time.perf_counter() - started,
            status="completed",
        )

        # 2. Repository Understanding
        started = time.perf_counter()
        emit("repository_understanding", "stage_started", "开始扫描目标仓库")
        try:
            project: ProjectStructure | None = scan_repository(repo_root)
        except Exception as exc:  # noqa: BLE001
            return self._stage_error("scan", api, None, None, None, warnings, exc)
        emit(
            "repository_understanding",
            "stage_completed",
            "目标仓库扫描完成",
            # 只记名字与数量，不记 root：那是这台机器上的绝对路径，
            # 而 trace 是会被展示、被粘贴、被存档的东西。
            metadata={
                "is_python_project": project.is_python_project,
                "python_files": len(project.python_files),
                "source_dirs": len(project.source_dirs),
                "test_dirs": len(project.test_dirs),
                "dependencies": len(project.dependencies),
            },
            duration=time.perf_counter() - started,
            status="completed",
        )

        # 3. Integration Planning（planner 自己还会发 planning_* / llm_* 事件）
        started = time.perf_counter()
        emit(
            "planner",
            "stage_started",
            "开始制定集成方案",
            metadata={"planner": type(self.planner).__name__},
        )
        try:
            plan = self.planner.plan(PlannerState(api=api, project=project, request=request))
        except Exception as exc:  # noqa: BLE001
            return self._stage_error("plan", api, project, None, None, warnings, exc)
        emit(
            "planner",
            "stage_completed",
            "集成方案阶段结束",
            metadata={"endpoints": len(plan.endpoints), "warnings": len(plan.warnings)},
            duration=time.perf_counter() - started,
            status="completed",
        )

        # 4. Code Generation
        # 这一段刻意不发 stage_started / stage_completed：生成器自己会发
        # stage_started + generation_completed（见 generation.code_generator），
        # 同一阶段两条 stage_started 只会让人以为生成跑了两次。
        try:
            artifacts = self.code_generator.generate(plan)
        except Exception as exc:  # noqa: BLE001
            return self._stage_error("generate", api, project, plan, None, warnings, exc)

        # 5. Repair Loop（初始测试 + 修复循环都在其中；test_* / repair_* 由内部自己发）
        repair_planner = self.repair_planner or DeterministicRepairPlanner(
            max_iterations=max_iterations
        )
        started = time.perf_counter()
        emit(
            "repair",
            "stage_started",
            "开始 Repair Loop",
            metadata={"max_iterations": max_iterations},
        )
        loop_result = run_repair_loop(
            artifacts,
            max_iterations=max_iterations,
            test_runner=self.test_runner,
            repair_planner=repair_planner,
            repair_applier=self.repair_applier,
        )
        warnings.extend(loop_result.warnings)
        # Loop 的 error 是"流程自身出错"，不是"测试没通过"——两者在 trace 里也必须分得开。
        emit(
            "repair",
            "stage_failed" if loop_result.status == "error" else "stage_completed",
            "Repair Loop 结束",
            metadata={
                "status": loop_result.status,
                "iterations": loop_result.iterations,
                "warnings": len(loop_result.warnings),
            },
            duration=time.perf_counter() - started,
            status=loop_result.status,
        )

        # 6. Final Patch / Diff（纯函数组件；失败不改变 status，只记录 warning）
        started = time.perf_counter()
        emit("patch", "stage_started", "开始生成最终 Patch")
        try:
            patch_result = self.patch_generator.generate(
                loop_result.artifacts, original_files=original_files
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Patch 生成失败：{type(exc).__name__}: {exc}")
            patch_result = None
        if patch_result is None:
            emit(
                "patch",
                "stage_failed",
                "Patch 生成失败",
                duration=time.perf_counter() - started,
                status="error",
            )
        else:
            # patch_generated 同时充当这个阶段的完成信号：这里没有第二个产物可报，
            # 再发一条 stage_completed 只会是同一件事的第二遍。
            emit(
                "patch",
                "patch_generated",
                "最终 Patch 已生成",
                metadata={
                    "files_changed": patch_result.files_changed,
                    "added": patch_result.files_created,
                    "modified": patch_result.files_modified,
                    # PatchFile.action 只有 create / modify 两种取值，本阶段结构上
                    # 不可能产出删除——这个 0 是陈述类型事实，不是"这次恰好没删"。
                    "deleted": 0,
                },
                duration=time.perf_counter() - started,
                status="completed",
            )

        # 状态映射：RepairLoop 的 error 表示"流程自身出错"（TestRunner / RepairPlanner /
        # RepairApplier 抛异常），与"测试确实执行了并且没通过"是两回事，不能归一成
        # tests_failed——否则调用方既无法判断该重试还是该改代码，又会把不可信的测试
        # 结论当成真实结论。
        if loop_result.status == "error":
            status = "error"
            failed_stage = "repair"
            # loop_result.error 已是安全文案（"Repair loop failed during X."），
            # 完整 traceback 只进服务端日志，因此可原样提升到 Pipeline 层。
            error = loop_result.error or "Repair loop failed."
        else:
            status = "passed" if loop_result.status == "passed" else "tests_failed"
            failed_stage = None
            error = None

        return PipelineResult(
            status=status,
            api=api,
            project=project,
            plan=plan,
            artifacts=loop_result.artifacts,
            initial_test_result=None,  # 当前 RepairLoop 不保存初始测试结果，不重复跑测试
            repair_loop_result=loop_result,
            patch=patch_result,
            failed_stage=failed_stage,
            warnings=warnings,
            error=error,
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
            - trace 同理：只记"哪个阶段失败"，异常原文留给服务端日志。
        """
        _logger.error("Pipeline 阶段失败：stage=%s", stage, exc_info=exc)
        emit(
            _STAGE_NAME[stage],
            "stage_failed",
            "阶段失败",
            metadata={"failed_stage": stage},
            status="error",
        )
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
