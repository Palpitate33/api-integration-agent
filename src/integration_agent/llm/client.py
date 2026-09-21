"""Provider-neutral 的 LLM 客户端契约，以及两个离线假客户端。

两个契约，互不影响：

    LLMClient.generate(prompt) -> str
        最小接口：输入 prompt，输出原始文本。Repair 流程在用（LLMRepairApplier）。
    ToolCallingClient.chat(messages, tools=...) -> AssistantTurn
        多轮 + Tool Calling：消息进、结构化的一轮出。

具体 provider 的客户端可以同时实现两者（适配器模块见 llm/ 包内的其余文件）。

依赖方向：
    本模块不 import agent / repair / api_server / pipeline，也不 import 任何具体
    provider 的 transport：契约层不认识 HTTP、不认识 API Key、不认识 wire format。
    唯一的上游依赖是 integration_agent.tools.models —— ToolSpec 是**纯数据**声明，
    「要提供哪些工具」本来就是调用方与模型之间的事，与 provider 无关。

    ToolSpec → 某个 provider 的请求体长什么样，是**适配器**的职责，调用方不需要
    提前把 schema 拼好，更不需要知道 "type": "function" 这类 wire format。

provider-neutral 的字面含义：
    本模块**不出现任何 provider 的名字**：不 import 适配器模块，不在标识符、
    __all__ 或字符串里提厂商名。这一条由 tests/test_llm_architecture.py
    逐标识符检查——否则"契约层"会慢慢长出某一个厂商的专有字段。
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from integration_agent.llm.models import AssistantTurn, ChatMessage
from integration_agent.tools.models import ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """最小 LLM 接口：输入 prompt，输出原始文本。"""

    def generate(self, prompt: str) -> str:
        """调用模型生成回复。"""
        ...


class FakeLLMClient:
    """按队列返回固定响应的假 LLM：完全确定性，供测试与离线调试使用。

    只提供一个响应时，每次调用返回相同内容；提供多个响应时按调用顺序弹出。
    """

    def __init__(self, responses: str | list[str] = "") -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLMClient 被调用的次数超过预期")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


@runtime_checkable
class ToolCallingClient(Protocol):
    """多轮 + Tool Calling 的 LLM 客户端契约。

    只约束最小的公共面：消息进、一轮出。temperature / timeout / response_format
    这类 provider 参数不放进协议——它们是具体客户端的旋钮，不属于调用方必须
    知道的东西。

    tools 收的是 provider-neutral 的 ToolSpec：调用方说清「有哪些工具」，
    「这些工具怎么变成请求体里的 wire format」由适配器在内部完成。
    """

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AssistantTurn:
        """把消息列表发给模型，返回结构化的一轮回复。"""
        ...


class FakeToolCallingClient:
    """按队列返回预设 AssistantTurn 的假客户端：确定性、不联网、不需要 API Key。

    主要用途是让 Agent Loop 能在离线状态下测完整回路：

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
    记录的 tools 是调用方给的 ToolSpec 原样，不是任何 wire format —— 断言
    "这里是 ToolSpec" 本身就是在守 Agent Loop 不生成 provider JSON 这条边界。
    """

    def __init__(self, turns: AssistantTurn | list[AssistantTurn] = ()) -> None:
        self._repeat = isinstance(turns, AssistantTurn)
        self._turns = [turns] if self._repeat else list(turns)
        self.calls = 0
        self.requests: list[list[ChatMessage]] = []
        self.tools: list[list[ToolSpec]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
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
    "FakeLLMClient",
    "FakeToolCallingClient",
    "LLMClient",
    "ToolCallingClient",
]
