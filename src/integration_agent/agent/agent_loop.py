"""Agent Loop：把多轮 tool calling 变成一条受控的、有界的、只读的执行回路。

这一层负责的是"回路"本身，不是业务：

    LLM → 决策 → Tool Call → ToolRegistry → 执行只读工具 → ToolResult
        → 原样回填完整消息历史 → 再次决策 → …… → 最终回答

职责边界：
    - 只做 LLM ↔ Tools ↔ Messages 的编排，**不生成 IntegrationPlan**：
      结构化解析属于 ToolUsingPlanner（见 tool_planner.py）。
    - 不 import repair / generation / pipeline / api_server，也不认识任何具体工具：
      工具的查找与执行只经过 ToolRegistry.get() 与 AgentTool.invoke()，没有 if/elif 分派。
    - 不认识 API Key：凭据只存在于注入的 ToolCallingClient 内部，本模块拿到的
      只有消息与一轮响应，没有任何读取环境变量或凭据的接口。

安全边界：
    - **绝不伪造 tool call**：assistant 消息一律由 AssistantTurn.to_message() 原样
      回填，程序不会自己造一个 tool_call，也不会改写模型给的 id / arguments。
      tool call 与 tool result 一旦对不上，整轮判 invalid_response 并停止，不做修复。
    - **只读**：本模块不写文件、不执行 shell / subprocess、不发网络请求；它能造成的
      副作用完全由注册进 registry 的工具决定（当前注册表里全部是只读工具）。
    - tool arguments 是**不可信 JSON**：只做 json.loads，不 eval、不当路径用、
      不自动重试同一个失败调用；解析失败只产出结构化失败结果并交回模型。
    - 一切失败都是**有界**的：轮次、工具调用数、观察字符数、历史字符数、连续工具
      错误数各有上限，循环是明确的 for 而不是 while True。

依赖方向：
    agent → tools → api / repository。本模块不反向依赖 repair / pipeline / api_server。
"""

import hashlib
import json
import logging
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from integration_agent.agent.llm import (
    ChatMessage,
    ToolCallingClient,
    ToolCallRequest,
    parse_tool_arguments,
    tool_spec_to_deepseek_function,
)
from integration_agent.tools.models import ToolResult
from integration_agent.tools.registry import AgentTool, ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

# 循环的终止原因。刻意做成显式枚举而不是自由字符串：调用方（ToolUsingPlanner /
# Pipeline / 前端）需要按状态分支处理，拼错一个状态名不该变成一个静默的"其他"。
AgentStatus = Literal[
    "completed",
    "max_tool_calls",
    "max_turns",
    "repeated_call",
    "tool_error",
    "llm_error",
    "invalid_response",
    "observation_limit",
    "error",
]

# 固定的安全文案：异常原文（可能含 Authorization 头、完整请求体/响应体、traceback）
# 只经 logger.exception 写进服务端日志，绝不进返回值。
LLM_CALL_FAILED = "调用 LLM 失败，本轮已终止（详细信息见服务端日志）"
TOOL_CRASHED = "工具内部错误，本轮已终止（详细信息见服务端日志）"
INVALID_ASSISTANT_TURN = (
    "模型返回的 assistant 轮次结构不合法（缺 id / name / arguments，或既无正文也无 tool_calls）"
)
EMPTY_FINAL_ANSWER = "模型既没有给出最终回答，也没有请求调用工具"
INVALID_PROMPT = "system_prompt 与 user_prompt 都必须是非空字符串"
UNKNOWN_TOOL = "工具未注册"
INVALID_ARGUMENTS = "invalid tool arguments"
REPEATED_CALL = "重复调用：相同的工具与相同的参数已经执行过"

# 工具名/错误里的模型文本压到这个长度再回填：工具名模型可以随便写，
# 不该让一个超长字符串无限制地进到消息、告警或日志里。
MAX_LABEL_CHARS = 64


class AgentLoopConfig(BaseModel):
    """Agent Loop 的硬上限。全部可配置，且全部必须 > 0。

    这些数字是"天花板不是默认值"：配置得越小越保守，但不存在"关掉上限"的选项——
    没有上限的 Agent 回路就是一个能把上下文吃光的死循环。
    """

    max_tool_calls: int = Field(default=8, gt=0)
    max_turns: int = Field(default=6, gt=0)
    max_observation_chars: int = Field(default=40_000, gt=0)
    max_consecutive_tool_errors: int = Field(default=2, gt=0)
    # 历史预算与观察预算是两件事：前者约束"发回去的上下文"，后者约束"工具产出的观察"
    max_history_chars: int = Field(default=60_000, gt=0)


class AgentRunResult(BaseModel):
    """一次 Agent 运行的结果。

    成功与失败同一个结构：失败时 status 说明原因、error 说明可读的原因文本，
    已发生的 tool_calls / tool_results 一律保留——"跑到哪一步停的"是排障的第一步，
    把它丢掉只会让失败变成一个无法复现的黑盒。
    """

    status: AgentStatus
    final_message: ChatMessage | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    turns: int = Field(default=0, ge=0)  # LLM 请求次数（一轮可能带多个 tool call）
    call_count: int = Field(default=0, ge=0)  # 实际处理过的 tool call 数
    observation_chars: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def content(self) -> str:
        """最终回答的正文；没有拿到最终回答时为空串。"""
        return self.final_message.content if self.final_message is not None else ""

    @property
    def completed(self) -> bool:
        return self.status == "completed"


@runtime_checkable
class AgentLoop(Protocol):
    """Agent Loop 的契约：一次有界、只读、失败即停的 Agent 运行。

    刻意只约束这一个方法：循环的"怎么跑"是实现细节，调用方只关心"给我一个结果"。
    """

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: ToolContext,
        registry: ToolRegistry,
        llm: ToolCallingClient,
    ) -> AgentRunResult:
        """跑完一次 Agent 回路并返回结果；不抛异常表示业务失败。"""
        ...


class AgentLoopRunner:
    """AgentLoop 的具体实现：有界、只读、失败即停。

    不是 Pydantic 模型：它没有需要序列化的状态，唯一的配置本身就是不可变的
    Pydantic 模型（AgentLoopConfig）。
    """

    def __init__(self, config: AgentLoopConfig | None = None) -> None:
        self.config = config if config is not None else AgentLoopConfig()

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: ToolContext,
        registry: ToolRegistry,
        llm: ToolCallingClient,
    ) -> AgentRunResult:
        """执行回路。

        控制流是**明确的 for 循环**（不是 while True），并且每一条退出路径都返回
        一个带状态的 AgentRunResult：没有任何一条路会静默地"跑完了但什么都没说"。
        """
        config = self.config
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            return AgentRunResult(status="error", error=INVALID_PROMPT)
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return AgentRunResult(status="error", error=INVALID_PROMPT)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        history_chars = len(system_prompt) + len(user_prompt)
        # 顺序取 registry.specs()（按名字排序），保证同样输入产生同样的 tools 定义
        tools = [tool_spec_to_deepseek_function(spec) for spec in registry.specs()]

        calls: list[ToolCallRequest] = []
        results: list[ToolResult] = []
        warnings: list[str] = []
        seen_calls: set[str] = set()
        observation_chars = 0
        consecutive_errors = 0
        turns = 0

        def finish(
            status: AgentStatus,
            *,
            final_message: ChatMessage | None = None,
            error: str | None = None,
        ) -> AgentRunResult:
            """收口：把"已经发生过什么"原样带出，不让任何一次失败变成黑盒。"""
            return AgentRunResult(
                status=status,
                final_message=final_message,
                tool_calls=list(calls),
                tool_results=list(results),
                turns=turns,
                call_count=len(calls),
                observation_chars=observation_chars,
                warnings=list(warnings),
                error=error,
            )

        for _ in range(config.max_turns):
            # 历史预算：宁可停下来，也不静默丢弃历史——被截断的历史会让模型
            # 基于残缺上下文继续决策，比直接失败危险得多。
            if history_chars > config.max_history_chars:
                warnings.append(
                    f"消息历史累计 {history_chars} 字符，"
                    f"超过上限 {config.max_history_chars}，已停止"
                )
                return finish(
                    "observation_limit",
                    error="消息历史超出上限，已停止（不静默丢弃历史）",
                )

            try:
                turn = llm.chat(messages=messages, tools=tools)
            except Exception:  # noqa: BLE001 - 任何 LLM 侧故障都收敛成有界状态
                logger.exception("Agent Loop 调用 LLM 失败")
                return finish("llm_error", error=LLM_CALL_FAILED)
            turns += 1

            # ---- 情况一：模型没有再要求调用工具 → 这就是最终回答 ----
            if not turn.tool_calls:
                if not turn.content.strip():
                    return finish("invalid_response", error=EMPTY_FINAL_ANSWER)
                try:
                    final_message = turn.to_message()
                except ValidationError:
                    return finish("invalid_response", error=INVALID_ASSISTANT_TURN)
                messages.append(final_message)
                return finish("completed", final_message=final_message)

            # ---- 情况二：模型要求调用工具 ----
            # 先把这一轮**所有** tool call 的字段校验完再回填 assistant 消息：
            # 半个 turn 已经进历史只会让"到底哪一步不合法"更难查，而 §不修复
            # 的原则要求整轮判废。
            fields = [_call_fields(call) for call in turn.tool_calls]
            if any(item is None for item in fields):
                return finish("invalid_response", error=INVALID_ASSISTANT_TURN)

            try:
                assistant_message = turn.to_message()
            except ValidationError:
                return finish("invalid_response", error=INVALID_ASSISTANT_TURN)
            messages.append(assistant_message)
            history_chars += _message_chars(assistant_message)

            for call, field in zip(turn.tool_calls, fields, strict=True):
                # any(item is None) 已经挡掉了 None；这里的断言是为了让类型与意图一致
                assert field is not None
                call_id, name, raw_arguments = field

                if len(calls) >= config.max_tool_calls:
                    return finish(
                        "max_tool_calls",
                        error=f"工具调用次数达到上限 {config.max_tool_calls}，已停止",
                    )
                calls.append(call)

                # 1) 按名字查注册表：没有 if/elif 分派，也没有"猜一个相近的工具名"
                tool = registry.get(name)
                if tool is None:
                    result = _failure(call_id, _short(name), f"{UNKNOWN_TOOL}：{_short(name)}")
                else:
                    # 2) arguments 是不可信 JSON：解析失败只回结构化失败，不执行
                    parsed = parse_tool_arguments(raw_arguments)
                    if not parsed.ok:
                        result = _failure(
                            call_id,
                            tool.spec.name,
                            f"{INVALID_ARGUMENTS}：{parsed.error}",
                        )
                    else:
                        # 3) 重复调用检测：同一工具 + 同一参数 = 同一个动作，
                        #    第二次不执行（否则模型可以靠复读把预算烧光）
                        fingerprint = _fingerprint(name, parsed.arguments)
                        if fingerprint in seen_calls:
                            results.append(_failure(call_id, tool.spec.name, REPEATED_CALL))
                            warnings.append(
                                f"检测到重复调用 {_short(name)}，已停止（第二次未执行）"
                            )
                            return finish("repeated_call", error=REPEATED_CALL)
                        seen_calls.add(fingerprint)
                        result = _invoke(tool, parsed.arguments, context, call_id)

                # ---- 统一收尾：记录结果 → 回填 tool 消息 → 更新预算 ----
                results.append(result)
                observation = _observation(result)
                messages.append(ChatMessage(role="tool", tool_call_id=call_id, content=observation))
                history_chars += len(observation) + len(call_id)
                observation_chars += len(result.content) + len(result.error or "")

                if result.ok:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    warnings.append(f"工具 {result.tool} 执行失败：{_short(result.error or '')}")
                    if consecutive_errors >= config.max_consecutive_tool_errors:
                        return finish(
                            "tool_error",
                            error=(
                                f"连续 {consecutive_errors} 次工具调用失败"
                                f"（上限 {config.max_consecutive_tool_errors}），已停止"
                            ),
                        )

                if observation_chars > config.max_observation_chars:
                    warnings.append(
                        f"工具观察结果累计 {observation_chars} 字符，"
                        f"超过上限 {config.max_observation_chars}，已停止（结果未被截断，"
                        "也没有假装完整）"
                    )
                    return finish(
                        "observation_limit",
                        error=f"工具观察结果超过 {config.max_observation_chars} 字符上限，已停止",
                    )

        return finish(
            "max_turns",
            error=f"达到轮次上限 {config.max_turns} 仍未得到最终回答，已停止",
        )


# ------------------------------------------------------------------ 辅助


def _call_fields(call: Any) -> tuple[str, str, str] | None:
    """从一次 tool call 里取出 (id, name, arguments) 三个字符串字段。

    任一字段缺失 / 不是字符串 / 只有空白就返回 None：调用方据此判
    "模型返回的轮次结构不合法" 并终止整轮。**绝不猜测或补齐缺失的字段**——
    缺 id 就无法把 tool 消息对回它的调用，缺 name 就无从路由。

    返回的 id / name 不做 strip：它们会被逐字节回填进下一轮请求，
    任何改写都会让 API 侧对不上号。
    """
    call_id = getattr(call, "id", None)
    name = getattr(call, "name", None)
    arguments = getattr(call, "arguments", None)
    if not isinstance(call_id, str) or not call_id.strip():
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(arguments, str):
        return None
    return call_id, name, arguments


def _invoke(
    tool: AgentTool,
    args: dict[str, Any],
    context: ToolContext,
    call_id: str,
) -> ToolResult:
    """执行一次工具调用，把结果对回模型的 call id。

    AgentTool 契约要求工具不向上抛异常，这里仍然兜一层：一个第三方工具的实现
    缺陷不该把整个回路炸掉。兜住不等于掩盖——堆栈进服务端日志，回给模型的只有
    一句话。工具返回的 call_id 是空的（工具不知道自己的 id），这里用模型给的
    call_id 补上，让结果与调用一一对应。
    """
    try:
        result = tool.invoke(args, context)
    except Exception:  # noqa: BLE001 - 工具边界的最后一道防线
        logger.exception("Agent Loop 执行工具 %s 时发生未预期错误", tool.spec.name)
        return _failure(call_id, tool.spec.name, TOOL_CRASHED)
    if not isinstance(result, ToolResult):
        return _failure(call_id, tool.spec.name, "工具返回了不符合契约的结果")
    return result.model_copy(update={"call_id": call_id})


def _failure(call_id: str, tool: str, message: str) -> ToolResult:
    """构造失败结果：content 保持空串，原因只放在 error 里。"""
    return ToolResult(call_id=call_id, tool=tool, ok=False, content="", error=message, chars=0)


def _observation(result: ToolResult) -> str:
    """tool 消息的正文：成功给内容，失败给错误说明，两者都不为 None。"""
    return result.content if result.ok else (result.error or "工具执行失败")


def _message_chars(message: ChatMessage) -> int:
    """一条消息占用的历史预算。

    刻意把 tool_calls 的 id / name / arguments 也算进去：它们同样会被发回 API，
    同样占上下文。宁可略微高估（更早停下），也不要低估到把请求撑爆。
    """
    total = len(message.content) + len(message.reasoning_content or "")
    total += len(message.tool_call_id or "")
    for call in message.tool_calls:
        total += len(call.id) + len(call.name) + len(call.arguments)
    return total


def _fingerprint(name: str, args: dict[str, Any]) -> str:
    """由「工具名 + 规范化 JSON 参数」算出的确定性指纹。

    sort_keys 让 {"a":1,"b":2} 与 {"b":2,"a":1} 归一到同一个指纹——键顺序不改变
    语义，不该让同一个动作被判成两次不同的调用。哈希而不是留原文：参数是不可信
    输入，没有必要为了去重在内存里留一份全文。
    """
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(f"{name}\0{canonical}".encode()).hexdigest()
    return digest


def _short(text: str, limit: int = MAX_LABEL_CHARS) -> str:
    """把模型给的标识性文本压到可读长度（工具名可能被模型写成任意长字符串）。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


__all__ = [
    "LLM_CALL_FAILED",
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopRunner",
    "AgentRunResult",
    "AgentStatus",
]
