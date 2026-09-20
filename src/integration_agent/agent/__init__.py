"""Integration Planner 阶段：基于 OpenAPI 解析结果与仓库扫描结果生成集成方案并编排各阶段。

主要 API：
    plan_integration(api, project, request=...)  —— 一次性完成规划
    DeterministicPlanner()                       —— 可配置的确定性 Planner
    PlannerState(api=..., project=...)           —— 规划输入状态（保留检索证据）

输出契约：
    IntegrationPlan —— 全字段强类型的 Pydantic Model，可被 Code Generator 直接消费
    TargetAPI / RepositoryContext / IntegrationStrategy / PlannedFile / FileModification /
    DependencyRequirement / PlannedEndpoint / AuthenticationPlan / ErrorHandlingPlan /
    TestingStrategy / Risk

约束：Planner 只分析、只规划——不修改仓库、不创建文件、不执行 shell 命令。
"""

from integration_agent.agent.agent_loop import AgentLoop, AgentLoopConfig, AgentRunResult
from integration_agent.agent.deepseek_planner import (
    DeepSeekPlanner,
    DeepSeekPlannerError,
    DeepSeekPlannerParseError,
    DeepSeekPlannerValidationError,
)
from integration_agent.agent.models import (
    AuthenticationPlan,
    DependencyRequirement,
    ErrorHandlingPlan,
    ErrorHandlingRule,
    FileModification,
    IntegrationPlan,
    IntegrationStrategy,
    PlannedEndpoint,
    PlannedFile,
    RepositoryContext,
    Risk,
    TargetAPI,
    TestingStrategy,
    TestSpec,
)
from integration_agent.agent.planner import (
    DEFAULT_HTTP_CLIENT,
    DEFAULT_MAX_ENDPOINTS,
    KNOWN_HTTP_CLIENTS,
    DeterministicPlanner,
    IntegrationPlanner,
    PlanningError,
    api_slug,
    plan_integration,
)
from integration_agent.agent.state import CodeEvidence, PlannerState
from integration_agent.agent.tool_planner import ToolUsingPlanner

__all__ = [
    "AgentLoop",
    "AgentLoopConfig",
    "AgentRunResult",
    "DEFAULT_HTTP_CLIENT",
    "DEFAULT_MAX_ENDPOINTS",
    "KNOWN_HTTP_CLIENTS",
    "AuthenticationPlan",
    "CodeEvidence",
    "DeepSeekPlanner",
    "DeepSeekPlannerError",
    "DeepSeekPlannerParseError",
    "DeepSeekPlannerValidationError",
    "DependencyRequirement",
    "DeterministicPlanner",
    "ErrorHandlingPlan",
    "ErrorHandlingRule",
    "FileModification",
    "IntegrationPlan",
    "IntegrationPlanner",
    "IntegrationStrategy",
    "PlannedEndpoint",
    "PlannedFile",
    "PlannerState",
    "PlanningError",
    "RepositoryContext",
    "Risk",
    "TargetAPI",
    "TestSpec",
    "TestingStrategy",
    "ToolUsingPlanner",
    "api_slug",
    "plan_integration",
]
