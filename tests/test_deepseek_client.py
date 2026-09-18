"""DeepSeekLLMClient 单元测试（全部离线，禁止真实网络请求）。

通过替换 deepseek_client._http_post 模拟 HTTP 层，覆盖要求场景：
    1. API Key 从环境变量读取    2. 显式 api_key 覆盖环境变量
    3. 缺少 API Key              4. model 配置
    5. base_url 配置             6. prompt 正确传给 client
    7. 正常 response 返回文本     8. 空 response
    9. 401 / 10. 403 / 11. 429   12. timeout
    13. 5xx                      14. 异常中不泄漏 API Key
    15. LLMClient Protocol 兼容   16. FakeLLMClient 旧测试仍通过（全套运行验证）

以及：HTTPError 分支、网络错误、无 choices、无 content、json_mode、
Authorization 头正确。
"""

import io
import json
import urllib.error
from typing import Any

import pytest

from integration_agent import repair
from integration_agent.repair import deepseek_client

DEFAULT_BODY = json.dumps({"choices": [{"message": {"content": "你好世界"}}]})


@pytest.fixture
def no_env_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def _patch_http(
    monkeypatch,
    *,
    status: int = 200,
    body: str = DEFAULT_BODY,
    error: Exception | None = None,
    sink: dict[str, Any] | None = None,
) -> None:
    def fake_post(url, payload, headers, timeout):
        if sink is not None:
            sink.update(url=url, payload=payload, headers=headers, timeout=timeout)
        if error is not None:
            raise error
        return status, body

    monkeypatch.setattr(deepseek_client, "_http_post", fake_post)


def _client(**kwargs) -> repair.DeepSeekLLMClient:
    return repair.DeepSeekLLMClient(api_key="test-key", **kwargs)


# ----------------------------------------- 场景 1-2：API Key 读取


def test_api_key_from_environment(monkeypatch, no_env_key) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, sink=sink)

    client = repair.DeepSeekLLMClient()
    assert client.api_key == "env-key"
    assert client.generate("hi") == "你好世界"
    assert sink["headers"]["Authorization"] == "Bearer env-key"


def test_explicit_api_key_overrides_environment(monkeypatch, no_env_key) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    client = repair.DeepSeekLLMClient(api_key="explicit-key")
    assert client.api_key == "explicit-key"


# ----------------------------------------------- 场景 3：缺少 API Key


def test_missing_api_key_raises_config_error(no_env_key) -> None:
    with pytest.raises(repair.DeepSeekConfigError, match="DEEPSEEK_API_KEY"):
        repair.DeepSeekLLMClient()


# ----------------------------------------------- 场景 4-5：model / base_url


def test_default_model_is_current_deepseek_id(no_env_key) -> None:
    client = _client()
    assert client.model == "deepseek-flash"


def test_model_configurable_via_env_and_kwarg(monkeypatch, no_env_key) -> None:
    # 显式覆盖能力保留：旧的 deepseek-v4-pro 等名称仍可显式传入
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    assert _client().model == "deepseek-v4-pro"
    assert _client(model="explicit-model").model == "explicit-model"


def test_base_url_default_and_configurable(monkeypatch, no_env_key) -> None:
    assert _client().base_url == "https://api.deepseek.com"

    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://proxy.example.com/v1")
    assert _client().base_url == "https://proxy.example.com/v1"
    assert _client(base_url="https://direct.example.com").base_url == ("https://direct.example.com")

    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, sink=sink)
    _client().generate("hi")
    assert sink["url"] == "https://proxy.example.com/v1/chat/completions"


# ----------------------------------------------- 场景 6：prompt 传递


def test_prompt_passed_to_api(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, sink=sink)

    _client().generate("修复 import 错误")

    messages = sink["payload"]["messages"]
    assert messages == [{"role": "user", "content": "修复 import 错误"}]
    assert sink["payload"]["model"] == "deepseek-flash"
    assert sink["payload"]["temperature"] == 0.0
    assert "response_format" not in sink["payload"]  # 默认不开启 json_mode


def test_json_mode_adds_response_format(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, sink=sink)

    _client(json_mode=True).generate("hi")

    assert sink["payload"]["response_format"] == {"type": "json_object"}


# ----------------------------------------- 场景 7-8：正常 / 空响应


def test_normal_response_returns_text(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch)
    assert _client().generate("hi") == "你好世界"


def test_empty_response_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body="")
    with pytest.raises(repair.DeepSeekResponseError, match="空响应"):
        _client().generate("hi")


# ----------------------------------------- 场景 9-11、13：错误状态码


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502])
def test_error_status_codes_raise_api_error(monkeypatch, no_env_key, status: int) -> None:
    _patch_http(
        monkeypatch,
        status=status,
        body=json.dumps({"error": {"message": "server rejected"}}),
    )
    with pytest.raises(repair.DeepSeekAPIError) as excinfo:
        _client().generate("hi")
    assert excinfo.value.status == status
    assert "server rejected" in str(excinfo.value)


def test_http_error_exception_branch(monkeypatch, no_env_key) -> None:
    """urllib 抛 HTTPError（真实 SDK 路径）时同样转换为 DeepSeekAPIError。"""
    body = json.dumps({"error": {"message": "invalid key"}}).encode("utf-8")
    http_error = urllib.error.HTTPError(
        "https://api.deepseek.com/chat/completions", 401, "Unauthorized", {}, io.BytesIO(body)
    )
    _patch_http(monkeypatch, error=http_error)
    with pytest.raises(repair.DeepSeekAPIError) as excinfo:
        _client().generate("hi")
    assert excinfo.value.status == 401
    assert "invalid key" in str(excinfo.value)


# ---------------------------------------------------- 场景 12：timeout


def test_timeout_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, error=TimeoutError("timed out"))
    with pytest.raises(repair.DeepSeekTimeoutError, match="超时"):
        _client().generate("hi")


# ------------------------------------------------------ 网络连接失败


def test_network_failure_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, error=urllib.error.URLError("connection refused"))
    with pytest.raises(repair.DeepSeekLLMError, match="网络连接失败"):
        _client().generate("hi")


# ------------------------------------------- 场景 14：异常不泄漏 API Key


def test_api_key_never_leaks_into_errors(monkeypatch, no_env_key) -> None:
    secret = "sk-super-secret-123"
    _patch_http(
        monkeypatch,
        status=401,
        body=json.dumps({"error": {"message": f"bad token {secret} for your account"}}),
    )
    with pytest.raises(repair.DeepSeekAPIError) as excinfo:
        repair.DeepSeekLLMClient(api_key=secret).generate("hi")
    assert secret not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_missing_content_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body=json.dumps({"choices": []}))
    with pytest.raises(repair.DeepSeekResponseError, match="choices"):
        _client().generate("hi")

    _patch_http(monkeypatch, body=json.dumps({"choices": [{"message": {"content": ""}}]}))
    with pytest.raises(repair.DeepSeekResponseError, match="有效文本"):
        _client().generate("hi")


# --------------------------------------------- 场景 15：Protocol 兼容


def test_llm_client_protocol_conformance(no_env_key) -> None:
    assert isinstance(_client(), repair.LLMClient)
