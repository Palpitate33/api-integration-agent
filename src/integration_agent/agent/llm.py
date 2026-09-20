"""支持多轮消息与 Tool Calling 的 LLM 客户端抽象。

这一层只描述「发消息 → 收模型响应」，不描述「谁来执行工具」：

    ToolCallingClient.chat(messages, tools=...) -> AssistantTurn
        messages  —— 结构化的 ChatMessage 列表（system / user / assistant / tool）
        tools     —— function calling 的 JSON Schema 列表，由
                     tool_spec_to_deepseek_function(ToolSpec) 转换而来
        返回值    —— AssistantTurn；其中的 tool_calls 是**模型的请求**，
                     不是执行结果，也不是执行许可

职责边界：
    本模块不 import ToolRegistry、不 import 任何具体 provider 客户端、不联网。
    工具的执行属于后续 Agent Loop 的职责：这里只把模型的意图变成结构化对象，
    让「要不要执行、执行结果怎么回填」成为一个显式的、可审阅的决定。

安全边界：
    - 这些模型里没有、也不允许出现 API Key / base_url / 环境变量的位置。
      凭据只存在于具体客户端实例内部，不进消息、不进返回值、不进日志。
    - tool arguments 是模型生成的**不可信 JSON 字符串**。本模块只提供把它解析
      成数据的接口（parse_tool_arguments），不提供任何执行、访问文件系统、
      发起网络请求或读取环境变量的接口。
    - 解析失败不是异常而是结构化错误（ToolArguments.ok=False）：模型给了一段
      坏 JSON 是可恢复的状况，应该把错误回给模型让它重试，而不是让整轮对话炸掉。

与 LLMClient 的关系：
    LLMClient.generate(prompt) -> str 是 Repair 流程在用的最小接口，保持不变。
    ToolCallingClient 是它的多轮版本：消息进、结构化的一轮出，二者互不影响，
    具体客户端（如 DeepSeekLLMClient）可以同时实现两者。
"""

import copy
import json
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from integration_agent.tools.models import ToolSpec

ChatRole = Literal["system", "user", "assistant", "tool"]


def _ensure_nonblank(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} 不能为空")


class ToolCallRequest(BaseModel):
    """模型发起的一次工具调用请求。

    arguments 刻意保存为**原始 JSON 字符串**，不在这里解析：

    1. 下一轮要把一模一样的 tool_calls 发回给 API，解析再序列化会改变字节；
    2. 解析失败是可以恢复的——把错误告诉模型让它重试即可，所以解析必须是一个
       独立的、由调用方决定怎么处理的步骤，见 ``parsed_arguments()``。

    id 与 name 都不可缺省：没有 name 无法路由到工具，没有 id 则下一轮的 tool
    消息无法与它对应，API 会直接拒绝。
    """

    id: str
    name: str
    arguments: str = ""

    @model_validator(mode="after")
    def _check_identity(self) -> "ToolCallRequest":
        _ensure_nonblank(self.id, "tool call id")
        _ensure_nonblank(self.name, "tool call name")
        return self

    def parsed_arguments(self) -> "ToolArguments":
        """解析 arguments；永远返回结构化结果，不抛异常。"""
        return parse_tool_arguments(self.arguments)


class ToolArguments(BaseModel):
    """tool call arguments 的解析结果。

    ok=False 时 arguments 为空 dict，error 说明原因。结构上与 ToolResult 同构：
    调用方永远不需要 try/except，也不会因为一段坏 JSON 丢掉整轮对话。
    """

    ok: bool
    arguments: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def parse_tool_arguments(raw: str) -> ToolArguments:
    """把 arguments 当**不可信输入**解析成 dict。

    解析成功只说明「它是一段 JSON 且顶层是对象」，不说明参数合法：类型、范围、
    枚举校验属于具体工具（ToolSpec.parameters）与后续 Agent Loop 的职责。
    空字符串按「无参数」处理，这是零参数工具的正常形态，不是错误。

    报错文本刻意**不回显 raw 的内容**，只说哪里不对：raw 是不可信文本，
    服务端完全可以把它原样塞回来。不把不可信文本搬进错误路径，这条泄漏通道
    就从结构上不存在，而不是靠脱敏去堵。
    """
    text = raw.strip()
    if not text:
        return ToolArguments(ok=True, arguments={})
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # exc 只含位置信息（line / column），不含 payload 本身
        return ToolArguments(ok=False, error=f"tool arguments 不是合法 JSON：{exc}")
    if not isinstance(data, dict):
        return ToolArguments(
            ok=False,
            error=f"tool arguments 顶层必须是 JSON 对象，实际是 {type(data).__name__}",
        )
    return ToolArguments(ok=True, arguments=data)


class ChatMessage(BaseModel):
    """一条对话消息。四个角色的合法形态由 model_validator 统一把关。

    system / user   必须有非空 content
    assistant       至少有 content 或 tool_calls 之一（只调工具不带正文是正常的）
                    可以携带 reasoning_content：thinking mode 下发起 tool calling
                    的轮次，后续请求必须把这一轮的思维链原样回传，否则 API 会
                    直接返回 HTTP 400
    tool            必须有 tool_call_id，且不能携带 tool_calls
    """

    role: ChatRole
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_role_contract(self) -> "ChatMessage":
        if self.reasoning_content is not None and self.role != "assistant":
            raise ValueError(
                f"只有 assistant 消息可以携带 reasoning_content，当前 role={self.role}"
            )
        if self.role == "tool":
            if not self.tool_call_id or not self.tool_call_id.strip():
                raise ValueError("role=tool 的消息必须带 tool_call_id")
            if self.tool_calls:
                raise ValueError("role=tool 的消息不能携带 tool_calls")
            return self

        if self.tool_calls and self.role != "assistant":
            raise ValueError(f"只有 assistant 消息可以携带 tool_calls，当前 role={self.role}")
        if self.tool_call_id:
            raise ValueError(f"只有 role=tool 的消息可以带 tool_call_id，当前 role={self.role}")

        if self.role in ("system", "user") and not self.content.strip():
            raise ValueError(f"role={self.role} 的消息必须有非空 content")
        if self.role == "assistant" and not self.content.strip() and not self.tool_calls:
            raise ValueError("assistant 消息必须至少有 content 或 tool_calls 之一")
        return self


class AssistantTurn(BaseModel):
    """模型返回的一轮结果。

    content          文本内容；只发起工具调用时为空串
    tool_calls       模型请求调用的工具（未执行）
    finish_reason    API 给的原因（stop / tool_calls / length / content_filter …）
    reasoning_content 推理模型的思维链；非推理模型为 None。thinking mode 下
                     发起 tool calling 的轮次，它必须连同 tool_calls 一起回传，
                     否则后续请求会被 API 以 HTTP 400 拒绝（见 to_message）。
    """

    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str | None = None
    reasoning_content: str | None = None

    def to_message(self) -> ChatMessage:
        """转成可以发回 API 的 assistant 消息。

        tool_calls 必须原样回填：下一轮的 tool 消息靠 tool_call_id 与它对应，
        漏掉就会被 API 拒绝，或者让模型重新发起已经做过的调用。

        reasoning_content 同样要回填（为 None 时才不带该字段）：thinking mode +
        tool calling 的场景下，后续请求必须带上这一轮的思维链，否则 API 可能
        返回 HTTP 400。

        两个字段都只是把已有的值搬过去，不重新解析、不重新序列化：arguments
        的字节必须与第一轮收到的完全一致，下一轮才有可能对得上号。
        """
        return ChatMessage(
            role="assistant",
            content=self.content,
            reasoning_content=self.reasoning_content,
            tool_calls=list(self.tool_calls),
        )


@runtime_checkable
class ToolCallingClient(Protocol):
    """多轮 + Tool Calling 的 LLM 客户端契约。

    只约束最小的公共面：消息进、一轮出。temperature / timeout / response_format
    这类 provider 参数不放进协议——它们是具体客户端的旋钮，不属于调用方必须
    知道的东西。
    """

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AssistantTurn:
        """把消息列表发给模型，返回结构化的一轮回复。"""
        ...


def tool_spec_to_deepseek_function(spec: ToolSpec) -> dict[str, Any]:
    """把 ToolSpec 转成 OpenAI / DeepSeek function calling 的 tools[] 元素。

    只搬运 name / description / parameters 三样：

    - read_only 是我们这边的策略标记（最终由 ToolRegistry 与调用方消费），
      API 不认识它，传过去只是浪费 token，所以不传；
    - ToolSpec 里既没有项目路径也没有凭据，转换结果天然不含敏感信息。

    返回的是**新构造的普通 dict**：parameters 深拷贝，调用方改它不会污染
    ToolSpec；两次调用结果逐键相等，可直接 json.dumps 进请求体。
    """
    # 兜底补上 type / properties：ToolSpec 只保证 parameters 可 JSON 序列化、
    # 且 type（若存在）为 object，而 API 侧期望一个完整可读的 object schema。
    parameters = copy.deepcopy(dict(spec.parameters))
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": parameters,
        },
    }


class FakeToolCallingClient:
    """按队列返回预设 AssistantTurn 的假客户端：确定性、不联网、不需要 API Key。

    主要用途是让后续 Agent Loop 能在离线状态下测完整回路：

        fake = FakeToolCallingClient([
            AssistantTurn(tool_calls=[ToolCallRequest(
                id="call-1", name="search_code", arguments='{"query": "httpx"}')]),
            AssistantTurn(content="完成", finish_reason="stop"),
        ])
        fake.chat(messages)   # -> 带 tool_calls 的一轮
        fake.chat(messages)   # -> 最终文本

    两种模式，由**传入的形式**决定，而不是由剩余个数决定：

        传单个 AssistantTurn  -> 每次都返回它（固定回答，永不耗尽）
        传列表                -> 严格按顺序弹出，用完后抛 AssertionError

    这里刻意与 FakeLLMClient 判然有别。FakeLLMClient 用「剩余个数 == 1」判断是否
    重复，结果是**传两个响应时第二个会被无限重复**，「调用次数超过预期」永远不会
    触发；一个多跑了一轮的 Agent Loop 会静默通过。假客户端存在的意义就是让这种
    问题炸出来，所以这里改成按传入形式判定：列表就必须恰好消耗完。

    requests / tools 记录每次调用的入参，供断言「消息与工具定义确实传对了」。
    """

    def __init__(self, turns: AssistantTurn | list[AssistantTurn] = ()) -> None:
        self._repeat = isinstance(turns, AssistantTurn)
        self._turns = [turns] if self._repeat else list(turns)
        self.calls = 0
        self.requests: list[list[ChatMessage]] = []
        self.tools: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AssistantTurn:
        self.calls += 1
        self.requests.append(list(messages))
        self.tools.append(list(tools) if tools else [])
        if not self._turns:
            raise AssertionError("FakeToolCallingClient 被调用的次数超过预期")
        if self._repeat:
            return self._turns[0]
        return self._turns.pop(0)


__all__ = [
    "AssistantTurn",
    "ChatMessage",
    "ChatRole",
    "FakeToolCallingClient",
    "ToolArguments",
    "ToolCallRequest",
    "ToolCallingClient",
    "parse_tool_arguments",
    "tool_spec_to_deepseek_function",
]
