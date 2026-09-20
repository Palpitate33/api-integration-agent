"""DeepSeek LLM Client：通过 DeepSeek 官方 OpenAI 兼容 API 调用真实模型。

两个能力，互不影响：

    generate(prompt) -> str          单轮文本。Repair 流程在用的最小接口。
    chat(messages, tools=...) -> AssistantTurn
                                     多轮 + Tool Calling。返回模型请求调用的
                                     工具，但**绝不执行它们**。

职责边界（与修复逻辑、与工具执行都完全解耦）：
    prompt / messages → DeepSeek → raw text / 结构化的一轮。
    不解析业务 JSON、不接触 RepairPlan / GeneratedArtifacts（那是 Applier 的职责）；
    不认识 ToolRegistry，也不 import tools 包（tools 的 JSON Schema 由调用方
    通过 agent.llm.tool_spec_to_deepseek_function 转好再传进来）。
    chat() 的产物是「模型想做什么」，不是「做了什么」——工具的查找、权限判断、
    参数校验与执行全部属于后续 Agent Loop。

安全边界：
    - API Key 只从环境变量 DEEPSEEK_API_KEY 读取（或显式构造传入），
      绝不写入代码 / 日志 / 异常信息 / Git。
    - 所有异常信息在生成前先对 API Key 做脱敏（_redact）。
    - 响应里回给调用方的每一个字符串字段同样过 _redact：服务端是外部系统，
      它把 Key 原样塞回来也不能让 Key 进到返回值里。
    - tool arguments 一律作为不可信数据搬运（字符串原样保留），本模块不执行、
      不 json.loads 后当配置用、不把任何字段当路径去访问文件。
    - 不依赖第三方 SDK：标准库 urllib 直接调用 OpenAI 兼容的
      POST {base_url}/chat/completions，零额外依赖。

依赖方向说明：
    本模块 import integration_agent.agent.llm 取消息与返回值的类型定义。
    这是「具体 provider 实现依赖抽象契约」，方向是对的；且实测
    ``import integration_agent.repair`` 本来就会经 repair → generation → agent
    加载 agent 包，这条 import 没有引入新的可达性，也不成环（agent 在运行时
    不依赖 repair，见 agent/deepseek_planner.py 的 TYPE_CHECKING 说明）。

配置：
    DEEPSEEK_API_KEY   必需；缺失时抛 DeepSeekConfigError
    DEEPSEEK_MODEL     可选；默认 deepseek-flash
                       （官方当前推荐的 Flash 模型 ID，对应 DeepSeek-V4.1-Flash；
                       deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用）
    DEEPSEEK_BASE_URL  可选；默认 https://api.deepseek.com
"""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from integration_agent.agent.llm import AssistantTurn, ChatMessage, ToolCallRequest

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_TIMEOUT = 60.0
DEFAULT_TEMPERATURE = 0.0  # 修复场景要求输出尽量确定

ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"


class DeepSeekLLMError(Exception):
    """DeepSeek 客户端错误基类。"""


class DeepSeekConfigError(DeepSeekLLMError):
    """配置错误（如未提供 API Key）。"""


class DeepSeekAPIError(DeepSeekLLMError):
    """API 返回错误状态码（401 / 403 / 429 / 5xx 等）。"""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"DeepSeek API 返回 HTTP {status}：{message}")


class DeepSeekTimeoutError(DeepSeekLLMError):
    """请求超时。"""


class DeepSeekResponseError(DeepSeekLLMError):
    """响应为空、非 JSON 或缺少有效文本。"""


class DeepSeekLLMClient:
    """DeepSeek 真实客户端：实现 LLMClient Protocol，可注入 StructuredLLMRepairApplier。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        json_mode: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv(ENV_API_KEY, "")
        if not self.api_key.strip():
            raise DeepSeekConfigError(
                f"未配置 API Key：请设置环境变量 {ENV_API_KEY}（或通过 api_key 参数传入）"
            )
        self.model = model or os.getenv(ENV_MODEL, DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv(ENV_BASE_URL, DEFAULT_BASE_URL)).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode

    def generate(self, prompt: str) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.json_mode:
            # 优先使用 DeepSeek 支持的 JSON 结构化输出；
            # 下游 Applier 仍会做 json.loads + Pydantic 校验，不信任模型声明。
            payload["response_format"] = {"type": "json_object"}
        return _extract_content(self._send(payload, self.timeout))

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> AssistantTurn:
        """多轮 + Tool Calling：把消息列表发给模型，返回结构化的一轮。

        tools 是**已经转换好的** JSON Schema 列表（调用方用
        agent.llm.tool_spec_to_deepseek_function(ToolSpec) 生成）。本方法不认识
        ToolSpec、不认识 ToolRegistry —— 它只把 schema 塞进请求体。

        返回的 tool_calls 是模型的请求，**没有被执行**：本方法不会去查工具、
        不会调用工具、更不会访问 tool arguments 里提到的任何地址或路径。

        response_format / temperature / timeout 是**单次调用**的覆盖值：
        缺省时用实例上的配置，显式传入时以传入值为准。
        json_mode=True 的实例在这里同样会带上 response_format=json_object，
        除非本次显式传了 response_format。
        """
        if not messages:
            raise DeepSeekLLMError("messages 不能为空")
        payload = self._build_chat_payload(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            temperature=temperature,
        )
        body = self._send(payload, timeout if timeout is not None else self.timeout)
        return _parse_assistant_turn(body, self.api_key)

    def _build_chat_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        """组装 /chat/completions 的请求体。

        只有 tools 非空时才带 tools / tool_choice：给一个空的 tools 数组与
        完全不带，语义并不相同，后者才是「本轮不提供工具」。
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_payload(message) for message in messages],
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = [dict(tool) if isinstance(tool, dict) else tool for tool in tools]
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        elif self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _send(self, payload: dict, timeout: float) -> str:
        """发送请求并返回响应正文；把传输层错误翻译成客户端异常。

        generate() 与 chat() 共用这一段：认证头、URL 与错误映射只在一处定义，
        否则两个入口迟早会在「哪个状态码算错」上分叉。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        try:
            status, body = _http_post(url, payload, headers, timeout)
        except TimeoutError as exc:
            raise DeepSeekTimeoutError(f"DeepSeek API 请求超时（{timeout}s）") from exc
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            raise DeepSeekLLMError(f"网络连接失败：{type(exc).__name__}") from exc
        if status != 200:
            raise DeepSeekAPIError(status, _extract_api_error(body, self.api_key))
        return body


def _http_post(url: str, payload: dict, headers: dict[str, str], timeout: float) -> tuple[int, str]:
    """发起 POST 请求并返回 (状态码, 响应文本)。独立函数，便于测试替换。"""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def _extract_api_error(body: str, api_key: str) -> str:
    """从错误响应中提取服务端 message；任何嵌入的文本都先对 API Key 脱敏。"""
    try:
        data = json.loads(body)
        message = data.get("error", {}).get("message")
        if isinstance(message, str) and message:
            return _redact(message, api_key)
    except (json.JSONDecodeError, AttributeError):
        pass
    return "请求失败（服务端未返回可读的错误信息）"


def _extract_content(body: str) -> str:
    """从 chat completions 响应中提取文本内容。"""
    if not body.strip():
        raise DeepSeekResponseError("DeepSeek API 返回空响应")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DeepSeekResponseError(f"DeepSeek API 响应不是有效 JSON：{exc}") from exc
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DeepSeekResponseError("DeepSeek API 响应中没有 choices")
    message = choices[0].get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekResponseError("DeepSeek API 响应中没有有效文本内容")
    return content


def _message_to_payload(message: ChatMessage) -> dict[str, Any]:
    """把一条 ChatMessage 转成 API 期望的 messages[] 元素。

    字段严格按角色给：assistant 才带 reasoning_content / tool_calls，tool 才带
    tool_call_id。content 始终下发（纯工具调用轮为 ""）——空字符串是 API
    接受的合法值，省略它反而让「这一轮到底有没有正文」变得不可测。

    reasoning_content 只在存在时下发：非推理模型根本没有这个字段，凭空补一个
    空串与「这一轮没有思维链」并不是一回事。值原样搬运，不重新编码。
    """
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.reasoning_content is not None:
        payload["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _parse_assistant_turn(body: str, api_key: str) -> AssistantTurn:
    """把 chat completions 响应解析成 AssistantTurn。

    与 _extract_content 的严格性刻意不同：那里要的是一段文本，所以空正文算失败；
    这里要的是**一轮动作**，带 tool_calls 的那一轮本来就没有正文，空 content
    是正常形态。真正的失败是既没有正文也没有 tool_calls。
    """
    if not body.strip():
        raise DeepSeekResponseError("DeepSeek API 返回空响应")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DeepSeekResponseError(f"DeepSeek API 响应不是有效 JSON：{exc}") from exc
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DeepSeekResponseError("DeepSeek API 响应中没有 choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise DeepSeekResponseError("DeepSeek API 响应的 choices[0] 不是对象")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise DeepSeekResponseError("DeepSeek API 响应的 choices[0] 缺少 message")

    turn = AssistantTurn(
        content=_redact(_as_text(message.get("content")), api_key),
        tool_calls=_parse_tool_calls(message.get("tool_calls"), api_key),
        finish_reason=_redact(_as_text(choice.get("finish_reason")), api_key) or None,
        reasoning_content=(_redact(_as_text(message.get("reasoning_content")), api_key) or None),
    )
    if not turn.content.strip() and not turn.tool_calls:
        raise DeepSeekResponseError("DeepSeek API 响应既没有文本内容也没有 tool_calls")
    return turn


def _parse_tool_calls(raw: Any, api_key: str) -> list[ToolCallRequest]:
    """解析 message.tool_calls。

    缺 id 或缺 name 的调用直接报错而不是跳过：跳过等于把模型的一个动作静默
    吞掉，调用方会看到「模型什么都没要求做」——这比报错更难排查。
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise DeepSeekResponseError("DeepSeek API 响应的 tool_calls 不是数组")

    calls: list[ToolCallRequest] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DeepSeekResponseError(f"DeepSeek API 响应的 tool_calls[{index}] 不是对象")
        function = item.get("function")
        if not isinstance(function, dict):
            raise DeepSeekResponseError(f"DeepSeek API 响应的 tool_calls[{index}] 缺少 function")
        try:
            calls.append(
                ToolCallRequest(
                    # id 会被逐字节回填进下一轮的 tool 消息，所以这里除了脱敏之外
                    # 不做任何改写（不 strip、不归一化）。脱敏在正常情况下是空操作：
                    # 只有服务端把 Key 塞进 id 时才会命中，而那种情况下宁可让它
                    # 对不上号（失败），也不让 Key 进到返回值里。
                    id=_redact(_as_text(item.get("id")), api_key),
                    name=_redact(_as_text(function.get("name")), api_key),
                    arguments=_redact(_coerce_arguments(function.get("arguments")), api_key),
                )
            )
        except ValidationError as exc:
            raise DeepSeekResponseError(
                f"DeepSeek API 响应的 tool_calls[{index}] 缺少 id 或 function.name"
            ) from exc
    return calls


def _coerce_arguments(value: Any) -> str:
    """把 function.arguments 归一成 JSON 字符串。

    正常情况下 API 给的就是字符串。个别 OpenAI 兼容实现会给一个已经解析好的
    对象，这时重新序列化而不是报错——丢掉这个调用等于把模型的动作吞掉。
    结果的 JSON 合法性由 agent.llm.parse_tool_arguments 判定，这里不预判。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise DeepSeekResponseError(
            "DeepSeek API 响应的 tool arguments 既不是字符串也不是可序列化的 JSON："
            f"{type(value).__name__}"
        ) from exc


def _as_text(value: Any) -> str:
    """把响应里可能为 null / 非字符串的字段归一成字符串。"""
    return value if isinstance(value, str) else ""


def _redact(text: str, secret: str) -> str:
    """把 secret 替换为 ***，保证任何错误信息都不会泄漏 Key。"""
    if secret:
        text = text.replace(secret, "***")
    return text
