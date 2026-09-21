"""Provider-neutral 的 LLM 数据模型：消息、一轮响应、工具调用请求。

这一层只描述「消息长什么样」「模型返回的一轮长什么样」，不描述「谁来执行工具」，
也不认识任何具体 provider：

    ChatMessage       —— 结构化的对话消息（system / user / assistant / tool）
    AssistantTurn     —— 模型返回的一轮；其中的 tool_calls 是**模型的请求**，
                         不是执行结果，也不是执行许可
    ToolCallRequest   —— 一次工具调用请求；arguments 保持原始 JSON 字符串
    ToolArguments     —— arguments 的解析结果（永不抛异常）

职责边界：
    本模块不 import ToolRegistry、不 import 任何 provider 客户端、不联网、
    不读环境变量。工具的执行属于 Agent Loop 的职责：这里只把模型的意图变成
    结构化对象，让「要不要执行、执行结果怎么回填」成为显式的、可审阅的决定。

安全边界：
    - 这些模型里没有、也不允许出现 API Key / base_url / 环境变量的位置。
      凭据只存在于具体客户端实例内部，不进消息、不进返回值、不进日志。
    - tool arguments 是模型生成的**不可信 JSON 字符串**。本模块只提供把它解析
      成数据的接口（parse_tool_arguments），不提供任何执行、访问文件系统、
      发起网络请求或读取环境变量的接口。
    - 解析失败不是异常而是结构化错误（ToolArguments.ok=False）：模型给了一段
      坏 JSON 是可恢复的状况，应该把错误回给模型让它重试，而不是让整轮对话炸掉。

与 LLMClient 的关系：
    LLMClient.generate(prompt) -> str 是 Repair 流程在用的最小接口（见 client.py），
    本模块与它无关。ToolCallingClient 是它的多轮版本：消息进、结构化的一轮出，
    二者互不影响，同一个客户端可以同时实现两者。

provider-neutral 的字面含义：
    本模块**不出现任何 provider 的名字**。不是"顺便没写"，而是由
    tests/test_llm_architecture.py 逐标识符检查——模型里一旦冒出某个厂商的
    专有字段，这里就不再是"所有 provider 都能用的消息模型"了。
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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


__all__ = [
    "AssistantTurn",
    "ChatMessage",
    "ChatRole",
    "ToolArguments",
    "ToolCallRequest",
    "parse_tool_arguments",
]
