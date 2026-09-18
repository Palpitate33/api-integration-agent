"""DeepSeek LLM Client：通过 DeepSeek 官方 OpenAI 兼容 API 调用真实模型。

职责边界（与修复逻辑完全解耦）：
    prompt → DeepSeek → raw text。
    不解析 JSON、不接触 RepairPlan / GeneratedArtifacts（那是 Applier 的职责）。

安全边界：
    - API Key 只从环境变量 DEEPSEEK_API_KEY 读取（或显式构造传入），
      绝不写入代码 / 日志 / 异常信息 / Git。
    - 所有异常信息在生成前先对 API Key 做脱敏（_redact）。
    - 不依赖第三方 SDK：标准库 urllib 直接调用 OpenAI 兼容的
      POST {base_url}/chat/completions，零额外依赖。

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        try:
            status, body = _http_post(url, payload, headers, self.timeout)
        except TimeoutError as exc:
            raise DeepSeekTimeoutError(f"DeepSeek API 请求超时（{self.timeout}s）") from exc
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            raise DeepSeekLLMError(f"网络连接失败：{type(exc).__name__}") from exc
        if status != 200:
            raise DeepSeekAPIError(status, _extract_api_error(body, self.api_key))
        return _extract_content(body)


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


def _redact(text: str, secret: str) -> str:
    """把 secret 替换为 ***，保证任何错误信息都不会泄漏 Key。"""
    if secret:
        text = text.replace(secret, "***")
    return text
