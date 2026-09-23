"""ToolUsingPlanner：让 Agent 自己决定"还要看什么"，再由程序守住事实边界。

与 DeterministicPlanner / DeepSeekPlanner 的关系
------------------------------------------------
三者实现同一个 IntegrationPlanner 协议，差别只在"怎么得到那份 JSON"：

    DeterministicPlanner  规则算出来，不调 LLM
    DeepSeekPlanner       一次 prompt → 一次回答，证据由程序预先检索好
    ToolUsingPlanner      多轮回路，证据由**模型自己**按需调用只读工具获取

这正是从"确定性 Planner"转向"Agent Planner"的关键差别：不再由程序猜检索词、
不再提前把所有材料摆好，而是给模型一组只读工具，让它看完再决定。

分工原则（与前两者一致，没有放松）
----------------------------------
LLM 负责**判断**：集成目标、策略取舍、要新建/修改哪些文件、依赖、认证方式、
错误处理、测试策略、风险与不确定性。

程序负责**约束**（全部复用 DeepSeekPlanner 已有的纯函数，不复制第二份实现）：
    resolve_endpoints    —— 端点必须能在真实 APIInfo 中按 method + path 解析，
                            同 path 不同 method 是两个端点，幻觉端点直接丢弃
    ground_facts         —— 事实字段（API 名称/版本/base_url/端点总数、仓库
                            root/name/source_dirs/test_dirs）由 PlannerState 回填，
                            LLM 无权改写
    validate_plan        —— 结构必须满足 IntegrationPlan 的模型约束
    enforce_constraints  —— 路径 / 依赖 / 凭据逐条对照证据，不合规的丢弃并记 warning
所有丢弃都写进 IntegrationPlan.warnings —— **不静默吞掉任何东西**。

安全边界（本模块不做的事）
--------------------------
    - 不写文件、不创建文件、不执行 shell / git / subprocess、不发网络请求。
    - 不读取 API Key，也不检查环境变量：凭据完全封装在注入的 llm 客户端内部。
    - **不重新解析 OpenAPI、不重新扫描仓库**：api / project 一律来自 PlannerState。
      唯一可能触发仓库扫描的是模型主动调用的 inspect_project 工具——那是只读的。
    - 失败一律抛 PlanningError（含 Agent Loop 未完成、最终 JSON 非法/不完整），
      不做静默 fallback：是否退回 DeterministicPlanner 由调用方决定，本类不替它做主。

依赖方向：
    agent → tools → api / repository。本模块不在运行时依赖 repair / generation /
    pipeline / api_server。resolve_endpoints 等纯函数 import 自同包的
    deepseek_planner（它对本模块没有任何反向依赖，不成环）。
"""

import logging
import time
from pathlib import Path
from typing import Any

from integration_agent.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRunner,
    AgentRunResult,
)
from integration_agent.agent.deepseek_planner import (
    DeepSeekPlannerError,
    enforce_constraints,
    ground_facts,
    known_repository_files,
    parse_plan_json,
    resolve_endpoints,
    sanitize,
    validate_plan,
)
from integration_agent.agent.models import IntegrationPlan
from integration_agent.agent.planner import DEFAULT_MAX_ENDPOINTS, PlanningError
from integration_agent.agent.prompt import DEFAULT_MAX_PROMPT_CHARS
from integration_agent.agent.state import PlannerState
from integration_agent.agent.tool_prompt import build_tool_planner_prompts
from integration_agent.llm import ToolCallingClient
from integration_agent.tools import build_default_registry
from integration_agent.tools.registry import ToolContext, ToolRegistry
from integration_agent.trace import emit

logger = logging.getLogger(__name__)

# 最终回答的包络键：模型只被要求输出 {"final": {...}} 这一种形状
FINAL_KEY = "final"

MIN_PROMPT_CHARS = 1_000


class ToolUsingPlanner:
    """用 Agent Loop 收集证据、再用程序做 grounded 校验的 Planner。

    构造时要给的是一个**已经能 chat + tool calling** 的客户端（ToolCallingClient），
    不是 generate(prompt) 的单轮客户端：本 Planner 的核心就是多轮回路。
    """

    def __init__(
        self,
        llm: ToolCallingClient,
        *,
        registry: ToolRegistry | None = None,
        loop: AgentLoop | None = None,
        config: AgentLoopConfig | None = None,
        max_endpoints: int = DEFAULT_MAX_ENDPOINTS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        if not callable(getattr(llm, "chat", None)):
            raise PlanningError(
                "llm 必须实现 chat(messages, tools=...) -> AssistantTurn（ToolCallingClient 协议）"
            )
        if max_endpoints < 1:
            raise PlanningError("max_endpoints 必须 >= 1")
        if max_prompt_chars < MIN_PROMPT_CHARS:
            raise PlanningError(f"max_prompt_chars 过小，至少 {MIN_PROMPT_CHARS}")
        self.llm = llm
        # 默认注册表就是四个只读工具（inspect_project / search_code / read_file /
        # inspect_api）；每次构造新实例，避免共享可变状态带来的测试耦合。
        self.registry = registry if registry is not None else build_default_registry()
        self.loop = loop if loop is not None else AgentLoopRunner(config)
        self.max_endpoints = max_endpoints
        self.max_prompt_chars = max_prompt_chars

    def plan(self, state: PlannerState) -> IntegrationPlan:
        """跑完 Agent 回路并产出经过约束的 IntegrationPlan。

        整个过程只读、可重放（同一输入 + 同一模型输出 = 同一计划）。
        """
        started = time.perf_counter()
        emit(
            "planner",
            "planning_started",
            "开始规划集成方案",
            metadata={"planner": "tool_using", "tools": len(self.registry.specs())},
        )
        warnings: list[str] = []

        # 1. 初始上下文：只给事实与已有证据，要不要补由模型自己决定
        context = ToolContext(
            project_root=Path(state.project.root),
            project=state.project,
            api=state.api,
            known_files=known_repository_files(state),
        )
        system_prompt, user_prompt = build_tool_planner_prompts(
            state,
            tools_description=self.registry.describe(),
            max_chars=self.max_prompt_chars,
        )

        # 2. Agent 回路：LLM ↔ 只读工具 ↔ 消息历史
        result = self.loop.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            registry=self.registry,
            llm=self.llm,
        )
        # 回路自己的告警（工具失败、重复调用、预算到顶）与后面的约束告警并到一起，
        # 都进 IntegrationPlan.warnings：调用方在一处就能看到"这次规划有多可靠"。
        warnings.extend(f"Agent Loop：{item}" for item in result.warnings)
        if not result.completed:
            raise PlanningError(_incomplete_message(result))

        # 3. 解析包络 → 约束 → 校验（全部复用 DeepSeekPlanner 的纯函数）
        data = parse_final_plan(result.content)
        endpoints = resolve_endpoints(data, state, warnings, max_endpoints=self.max_endpoints)
        target_api, repository, grounding_warnings = ground_facts(data, state)
        warnings.extend(grounding_warnings)

        data["target_api"] = target_api.model_dump()
        data["repository"] = repository.model_dump()
        data["endpoints"] = endpoints

        try:
            plan = validate_plan(data)
        except DeepSeekPlannerError as exc:
            raise PlanningError(f"LLM 返回的集成计划不满足模型约束：{sanitize(str(exc))}") from None

        # 4. 约束层：把 LLM 的"判断"逐条对照证据，不合规的丢弃并记录
        plan, constraint_warnings = enforce_constraints(plan, state)
        warnings.extend(constraint_warnings)

        plan = plan.model_copy(update={"warnings": [*plan.warnings, *warnings]})
        # agent_status 把"这次规划背后那条回路跑得怎么样"带进 planner 事件：
        # 计划看起来正常、回路其实一路降级，是这类 Planner 最需要被看见的情况。
        emit(
            "planner",
            "planning_completed",
            "集成方案已生成",
            metadata={
                "planner": "tool_using",
                "agent_status": result.status,
                "turns": result.turns,
                "tool_calls": result.call_count,
                "endpoints": len(plan.endpoints),
                "files_to_create": len(plan.files_to_create),
                "files_to_modify": len(plan.files_to_modify),
                "warnings": len(plan.warnings),
            },
            duration=time.perf_counter() - started,
            status="completed",
        )
        return plan


def parse_final_plan(raw: str) -> dict[str, Any]:
    """解析最终回答的包络 ``{"final": {...}}``，返回其中的计划字典。

    只接受这一种形状：顶层的任何其它写法（裸计划、code fence 外的解释文字、
    包在别的键里）都判失败。包络存在的意义就是让"计划正文"与"解释性文字"在结构上
    分得开——一旦允许两可，解析就变成了猜。
    """
    try:
        envelope = parse_plan_json(raw)
    except DeepSeekPlannerError as exc:
        raise PlanningError(f"无法解析 LLM 返回的集成计划：{sanitize(str(exc))}") from None
    data = envelope.get(FINAL_KEY)
    if not isinstance(data, dict):
        raise PlanningError(
            f'LLM 最终回答缺少 "{FINAL_KEY}" 对象：计划必须包在 {{"{FINAL_KEY}": {{...}}}} 中'
        )
    return data


def _incomplete_message(result: AgentRunResult) -> str:
    """把"回路没跑完"翻译成一句可读、可脱敏的错误。

    result.error 由 AgentLoopRunner 保证是固定安全文案，这里仍然过一遍 sanitize：
    loop 是可替换的实现（AgentLoop 协议），不能假设每个实现都不回显原文。
    """
    detail = sanitize(result.error) if result.error else "未给出原因"
    return f"Agent Loop 未完成（status={result.status}）：{detail}"


__all__ = [
    "FINAL_KEY",
    "ToolUsingPlanner",
    "parse_final_plan",
]
