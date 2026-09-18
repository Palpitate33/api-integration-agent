"""最小 LLM Client 抽象：把"调用模型"与"修复业务逻辑"解耦。

LLMRepairApplier 只依赖 LLMClient Protocol：
    - 未来接 OpenAI / Anthropic / Ollama / 本地模型时，实现 generate() 即可，
      修复逻辑与 Repair Loop 零改动。
    - 测试不需要真实 API Key：FakeLLMClient 按队列返回固定响应。
"""

from typing import Protocol, runtime_checkable


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
