"""Repair Loop 阶段：根据测试与校验反馈自动修复生成的集成代码。

已实现（三个阶段）：
    plan_repair(artifacts, test_result)          —— 修复计划层（纯分析）
    apply_repair(artifacts, plan)                —— 修复应用层（纯内存，安全中间层）
    run_repair_loop(artifacts)                   —— 修复循环编排层（注入 Runner/Planner/Applier）
    DeterministicRepairPlanner / DeterministicRepairApplier / RepairLoopRunner
    RepairPlanner / RepairApplier / RepairLoop    —— Protocol，未来 LLM 实现共用同一契约

输出契约：
    RepairPlan / RepairAction / RepairApplicationResult / RepairLoopResult

约束：只分析、只在内存中变换生成产物、只编排注入组件——不修改仓库、
不创建/删除文件、不执行 shell / git、不发网络请求、不安装依赖、不编造代码。
"""

from integration_agent.repair.llm_client import FakeLLMClient, LLMClient
from integration_agent.repair.llm_repair_applier import (
    LLMRepairApplier,
    StructuredLLMRepairApplier,
    build_repair_prompt,
    parse_llm_response,
    validate_path,
)
from integration_agent.repair.models import (
    LLMFileChange,
    LLMRepairResponse,
    RepairAction,
    RepairApplicationResult,
    RepairLoopResult,
    RepairPlan,
)
from integration_agent.repair.repair_applier import (
    DeterministicRepairApplier,
    RepairApplier,
    apply_repair,
)
from integration_agent.repair.repair_loop import (
    RepairLoop,
    RepairLoopRunner,
    run_repair_loop,
)
from integration_agent.repair.repair_planner import (
    DEFAULT_MAX_ITERATIONS,
    DeterministicRepairPlanner,
    RepairPlanner,
    plan_repair,
)

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DeterministicRepairApplier",
    "DeterministicRepairPlanner",
    "FakeLLMClient",
    "LLMClient",
    "LLMFileChange",
    "LLMRepairApplier",
    "LLMRepairResponse",
    "RepairAction",
    "RepairApplicationResult",
    "RepairApplier",
    "RepairLoop",
    "RepairLoopResult",
    "RepairLoopRunner",
    "RepairPlan",
    "RepairPlanner",
    "StructuredLLMRepairApplier",
    "apply_repair",
    "build_repair_prompt",
    "parse_llm_response",
    "plan_repair",
    "run_repair_loop",
    "validate_path",
]
