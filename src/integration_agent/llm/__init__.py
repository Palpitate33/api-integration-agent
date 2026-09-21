"""LLM 层：provider-neutral 契约 + DeepSeek 适配器。

这是 LLM 相关类型的**唯一归属地**：它既不属于 agent domain，也不属于 repair
domain，因此放在两者之下的独立底层包，被所有上层单向依赖。

    agent        ──┐
    repair       ──┤
    api_server   ──┼──→ integration_agent.llm ──→ DeepSeek Adapter ──→ DeepSeek API
    pipeline     ──┘

分层：

    models.py   provider-neutral 数据模型
                ChatRole / ChatMessage / AssistantTurn / ToolCallRequest /
                ToolArguments / parse_tool_arguments
    client.py   provider-neutral 契约与假客户端
                LLMClient / FakeLLMClient / ToolCallingClient / FakeToolCallingClient
    deepseek.py DeepSeek 适配器：整层的 provider schema 只在这里出现
                DeepSeekLLMClient / 5 个异常 / DEFAULT_MODEL /
                tool_spec_to_deepseek_function

两条契约，互不影响，同一个客户端可以同时实现：

    LLMClient.generate(prompt) -> str
    ToolCallingClient.chat(messages, tools: Sequence[ToolSpec] | None) -> AssistantTurn

依赖方向（单向，无环）：
    llm → tools.models（ToolSpec 是纯数据声明）
    llm 不依赖 agent / repair / pipeline / api_server，也不依赖 tools.registry：
    契约层不认识 ToolRegistry，更不认识任何工具的执行。

安全边界：
    本层不含凭据。DeepSeekLLMClient 的 API Key 只存在于实例内部，不进消息、
    不进返回值、不进日志；其余类型没有任何读取环境变量的入口。
"""

from integration_agent.llm.client import (
    FakeLLMClient,
    FakeToolCallingClient,
    LLMClient,
    ToolCallingClient,
)
from integration_agent.llm.deepseek import (
    DEFAULT_MODEL,
    DeepSeekAPIError,
    DeepSeekConfigError,
    DeepSeekLLMClient,
    DeepSeekLLMError,
    DeepSeekResponseError,
    DeepSeekTimeoutError,
    tool_spec_to_deepseek_function,
)
from integration_agent.llm.models import (
    AssistantTurn,
    ChatMessage,
    ChatRole,
    ToolArguments,
    ToolCallRequest,
    parse_tool_arguments,
)

__all__ = [
    "DEFAULT_MODEL",
    "AssistantTurn",
    "ChatMessage",
    "ChatRole",
    "DeepSeekAPIError",
    "DeepSeekConfigError",
    "DeepSeekLLMClient",
    "DeepSeekLLMError",
    "DeepSeekResponseError",
    "DeepSeekTimeoutError",
    "FakeLLMClient",
    "FakeToolCallingClient",
    "LLMClient",
    "ToolArguments",
    "ToolCallRequest",
    "ToolCallingClient",
    "parse_tool_arguments",
    "tool_spec_to_deepseek_function",
]
