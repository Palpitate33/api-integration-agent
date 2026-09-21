"""DeepSeekLLMClient 单元测试（全部离线，禁止真实网络请求）。

通过替换 deepseek._http_post 模拟 HTTP 层，覆盖要求场景：
    1. API Key 从环境变量读取    2. 显式 api_key 覆盖环境变量
    3. 缺少 API Key              4. model 配置
    5. base_url 配置             6. prompt 正确传给 client
    7. 正常 response 返回文本     8. 空 response
    9. 401 / 10. 403 / 11. 429   12. timeout
    13. 5xx                      14. 异常中不泄漏 API Key
    15. LLMClient Protocol 兼容   16. FakeLLMClient 旧测试仍通过（全套运行验证）

以及：HTTPError 分支、网络错误、无 choices、无 content、json_mode、
Authorization 头正确。

chat() 多轮 / Tool Calling 部分覆盖：
    C. 多轮消息序列化   D. tool_calls 解析      E. arguments JSON 字符串
    G. tool_call id 保留  H. finish_reason 保留  I. reasoning_content
    J. tools / tool_choice 进入请求体             K. API error 不泄漏 Key
    M. 不执行任何工具    N. 不发生真实工具调用

全部用例都不联网：唯一出网口 _http_post 被打桩，并有哨兵用例证明
没有第二条出网路径。
"""

import ast
import io
import json
import logging
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from integration_agent import llm
from integration_agent.llm import (
    AssistantTurn,
    ChatMessage,
    FakeToolCallingClient,
    ToolCallingClient,
    ToolCallRequest,
    deepseek,
    tool_spec_to_deepseek_function,
)
from integration_agent.tools import build_default_registry

DEFAULT_BODY = json.dumps({"choices": [{"message": {"content": "你好世界"}}]})

SECRET = "sk-super-secret-123"

# llm/deepseek.py 允许 import 的全部模块。白名单而非黑名单：新增
# import 必须是有意识的决定。
#
# 这里**允许** integration_agent.tools.models：chat() 收到的 tools 是
# provider-neutral 的 ToolSpec，适配器必须认识这个纯数据契约，才能把它转成
# DeepSeek 的 tools[]。**不允许**的是 integration_agent.tools（包级，
# registry / 具体工具都在里面）、tools.registry 与任何 agent 模块——
# 适配器不认识 ToolRegistry、不认识 AgentLoop，更没有任何执行工具的入口。
ALLOWED_CLIENT_IMPORTS = {
    "copy",
    "json",
    "os",
    "urllib.error",
    "urllib.request",
    "collections.abc",
    "typing",
    "pydantic",
    "integration_agent.llm.models",
    "integration_agent.tools.models",
}


def _tool_call_payload(
    call_id: str = "call-1", name: str = "search_code", arguments: str = '{"query": "httpx"}'
) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _chat_body(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = "stop",
    reasoning_content: str | None = None,
    index: int = 0,
) -> str:
    """构造一个 /chat/completions 响应。缺省字段不出现，模拟真实响应的形状。"""
    message: dict[str, Any] = {}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return json.dumps(
        {"choices": [{"index": index, "message": message, "finish_reason": finish_reason}]}
    )


def _user(text: str = "把 httpx 换成官方 SDK") -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


def _imported_modules(source_file: Path) -> set[str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


@pytest.fixture
def no_env_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch) -> None:
    """本文件所有用例的出网哨兵：不许真的访问 DeepSeek。

    _http_post 是唯一出网口，正常用例都会用 _patch_http 把它换掉，这条哨兵
    不会被触发。它的价值在于「忘记打桩」那种情况：新写的用例若直接调 client，
    会立刻 AssertionError，而不是悄悄发出一发真实请求（可能带上真实 API Key，
    也可能因为网络不通而 flaky）。
    """

    def bomb(*args, **kwargs):
        raise AssertionError("单元测试不允许真实网络请求：请用 _patch_http 打桩")

    monkeypatch.setattr(deepseek.urllib.request, "urlopen", bomb)


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

    monkeypatch.setattr(deepseek, "_http_post", fake_post)


def _client(**kwargs) -> llm.DeepSeekLLMClient:
    return llm.DeepSeekLLMClient(api_key="test-key", **kwargs)


# ----------------------------------------- 场景 1-2：API Key 读取


def test_api_key_from_environment(monkeypatch, no_env_key) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, sink=sink)

    client = llm.DeepSeekLLMClient()
    assert client.api_key == "env-key"
    assert client.generate("hi") == "你好世界"
    assert sink["headers"]["Authorization"] == "Bearer env-key"


def test_explicit_api_key_overrides_environment(monkeypatch, no_env_key) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    client = llm.DeepSeekLLMClient(api_key="explicit-key")
    assert client.api_key == "explicit-key"


# ----------------------------------------------- 场景 3：缺少 API Key


def test_missing_api_key_raises_config_error(no_env_key) -> None:
    with pytest.raises(llm.DeepSeekConfigError, match="DEEPSEEK_API_KEY"):
        llm.DeepSeekLLMClient()


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
    with pytest.raises(llm.DeepSeekResponseError, match="空响应"):
        _client().generate("hi")


# ----------------------------------------- 场景 9-11、13：错误状态码


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502])
def test_error_status_codes_raise_api_error(monkeypatch, no_env_key, status: int) -> None:
    _patch_http(
        monkeypatch,
        status=status,
        body=json.dumps({"error": {"message": "server rejected"}}),
    )
    with pytest.raises(llm.DeepSeekAPIError) as excinfo:
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
    with pytest.raises(llm.DeepSeekAPIError) as excinfo:
        _client().generate("hi")
    assert excinfo.value.status == 401
    assert "invalid key" in str(excinfo.value)


# ---------------------------------------------------- 场景 12：timeout


def test_timeout_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, error=TimeoutError("timed out"))
    with pytest.raises(llm.DeepSeekTimeoutError, match="超时"):
        _client().generate("hi")


# ------------------------------------------------------ 网络连接失败


def test_network_failure_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, error=urllib.error.URLError("connection refused"))
    with pytest.raises(llm.DeepSeekLLMError, match="网络连接失败"):
        _client().generate("hi")


# ------------------------------------------- 场景 14：异常不泄漏 API Key


def test_api_key_never_leaks_into_errors(monkeypatch, no_env_key) -> None:
    secret = "sk-super-secret-123"
    _patch_http(
        monkeypatch,
        status=401,
        body=json.dumps({"error": {"message": f"bad token {secret} for your account"}}),
    )
    with pytest.raises(llm.DeepSeekAPIError) as excinfo:
        llm.DeepSeekLLMClient(api_key=secret).generate("hi")
    assert secret not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_missing_content_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body=json.dumps({"choices": []}))
    with pytest.raises(llm.DeepSeekResponseError, match="choices"):
        _client().generate("hi")

    _patch_http(monkeypatch, body=json.dumps({"choices": [{"message": {"content": ""}}]}))
    with pytest.raises(llm.DeepSeekResponseError, match="有效文本"):
        _client().generate("hi")


# --------------------------------------------- 场景 15：Protocol 兼容


def test_llm_client_protocol_conformance(no_env_key) -> None:
    assert isinstance(_client(), llm.LLMClient)


# ===========================================================================
# chat()：多轮消息与 Tool Calling
# ===========================================================================


# ---------------------------------------------------- 场景 C：多轮消息序列化


def test_chat_serializes_a_full_multi_round_conversation(monkeypatch, no_env_key) -> None:
    """一轮完整的「用户提问 → 模型要工具 → 工具结果 → 模型收尾」的消息列表。"""
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="完成"), sink=sink)

    _client().chat(
        [
            ChatMessage(role="system", content="你是集成助手"),
            ChatMessage(role="user", content="把 httpx 换成官方 SDK"),
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCallRequest(id="call-1", name="search_code", arguments='{"q": "x"}')
                ],
            ),
            ChatMessage(role="tool", content="src/pkg/client.py:12", tool_call_id="call-1"),
        ]
    )

    assert sink["payload"]["messages"] == [
        {"role": "system", "content": "你是集成助手"},
        {"role": "user", "content": "把 httpx 换成官方 SDK"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search_code", "arguments": '{"q": "x"}'},
                }
            ],
        },
        {"role": "tool", "content": "src/pkg/client.py:12", "tool_call_id": "call-1"},
    ]
    assert sink["url"] == "https://api.deepseek.com/chat/completions"


def test_chat_keeps_assistant_content_alongside_tool_calls(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client().chat(
        [
            ChatMessage(role="user", content="x"),
            ChatMessage(
                role="assistant",
                content="我先查一下",
                tool_calls=[ToolCallRequest(id="call-9", name="read_file", arguments="{}")],
            ),
        ]
    )

    assistant = sink["payload"]["messages"][1]
    assert assistant["content"] == "我先查一下"
    assert assistant["tool_calls"][0]["id"] == "call-9"


def test_chat_omits_fields_that_belong_to_other_roles(monkeypatch, no_env_key) -> None:
    """user 消息不带 tool_calls / tool_call_id，assistant 纯文本轮不带 tool_calls。"""
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client().chat(
        [
            ChatMessage(role="user", content="x"),
            ChatMessage(role="assistant", content="我看到了"),
            ChatMessage(role="user", content="那改吧"),
        ]
    )

    for message in sink["payload"]["messages"]:
        assert set(message) == {"role", "content"}


def test_chat_rejects_an_empty_message_list(no_env_key) -> None:
    with pytest.raises(llm.DeepSeekLLMError, match="messages 不能为空"):
        _client().chat([])


# ------------------------------------------------------ 场景 D/G/H/I：响应解析


def test_chat_parses_tool_calls(monkeypatch, no_env_key) -> None:
    _patch_http(
        monkeypatch,
        body=_chat_body(
            content="",
            tool_calls=[_tool_call_payload("call-1", "search_code", '{"query": "httpx"}')],
            finish_reason="tool_calls",
        ),
    )

    turn = _client().chat(_user())

    assert isinstance(turn, AssistantTurn)
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "search_code"
    # 场景 G：id 逐字节保留，下一轮的 tool 消息靠它对应
    assert turn.tool_calls[0].id == "call-1"
    # 场景 H
    assert turn.finish_reason == "tool_calls"


def test_chat_parses_parallel_tool_calls_in_order(monkeypatch, no_env_key) -> None:
    _patch_http(
        monkeypatch,
        body=_chat_body(
            tool_calls=[
                _tool_call_payload("call-a", "search_code", '{"query": "a"}'),
                _tool_call_payload("call-b", "read_file", '{"path": "b.py"}'),
                _tool_call_payload("call-c", "inspect_api", "{}"),
            ],
            finish_reason="tool_calls",
        ),
    )

    turn = _client().chat(_user())

    assert [call.id for call in turn.tool_calls] == ["call-a", "call-b", "call-c"]
    assert [call.name for call in turn.tool_calls] == ["search_code", "read_file", "inspect_api"]


def test_chat_keeps_finish_reason_verbatim(monkeypatch, no_env_key) -> None:
    for reason in (
        "stop",
        "tool_calls",
        "length",
        "content_filter",
        "insufficient_system_resource",
    ):
        _patch_http(monkeypatch, body=_chat_body(content="好", finish_reason=reason))
        assert _client().chat(_user()).finish_reason == reason


def test_chat_handles_a_missing_finish_reason(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body=json.dumps({"choices": [{"message": {"content": "好"}}]}))

    assert _client().chat(_user()).finish_reason is None


def test_chat_treats_a_content_only_reply_as_a_normal_turn(monkeypatch, no_env_key) -> None:
    """与 generate() 不同：空 content 不算错（那一轮本来就只有 tool_calls）。"""
    _patch_http(monkeypatch, body=_chat_body(content="改完了"))

    turn = _client().chat(_user())

    assert turn.content == "改完了"
    assert turn.tool_calls == []


def test_chat_rejects_a_turn_with_neither_content_nor_tool_calls(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body=_chat_body(content="   "))

    with pytest.raises(llm.DeepSeekResponseError, match="既没有文本内容也没有 tool_calls"):
        _client().chat(_user())


def test_chat_accepts_a_null_content_when_tool_calls_are_present(monkeypatch, no_env_key) -> None:
    _patch_http(
        monkeypatch,
        body=_chat_body(
            content=None, tool_calls=[_tool_call_payload()], finish_reason="tool_calls"
        ),
    )

    turn = _client().chat(_user())

    assert turn.content == ""
    assert len(turn.tool_calls) == 1


# ------------------------------------------------- 场景 I：reasoning_content


def test_chat_keeps_reasoning_content(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body=_chat_body(content="答案是 42", reasoning_content="我先算一下"))

    turn = _client().chat(_user())

    assert turn.reasoning_content == "我先算一下"


def test_chat_reports_no_reasoning_content_for_non_reasoning_models(
    monkeypatch, no_env_key
) -> None:
    _patch_http(monkeypatch, body=_chat_body(content="答案是 42"))

    assert _client().chat(_user()).reasoning_content is None


def test_reasoning_content_survives_a_cross_turn_tool_call_round_trip(
    monkeypatch, no_env_key
) -> None:
    """完整的跨轮回路：user → assistant(content + reasoning + tool_calls) → tool → 第二轮。

    DeepSeek 在 thinking mode + tool calling 下要求后续请求完整回传上一轮
    assistant message 的 reasoning_content，缺了会返回 HTTP 400。所以这里验的
    不是「字段存在」，而是第二轮**实际发出去**的 messages 与第一轮收到的逐字节
    一致：不删除、不改写、不 JSON 重编码、不截断、不转成别的字段。
    """
    reasoning = "reasoning-123"
    raw_arguments = '{ "query" : "httpx" , "limit": 5 }'
    sink: dict[str, Any] = {}
    _patch_http(
        monkeypatch,
        body=_chat_body(
            content="我先查一下项目里哪里用了 httpx",
            reasoning_content=reasoning,
            tool_calls=[_tool_call_payload("call-1", "search_code", raw_arguments)],
            finish_reason="tool_calls",
        ),
        sink=sink,
    )
    client = _client()

    first = client.chat(_user("项目里哪里用了 httpx？"))

    assert first.reasoning_content == reasoning
    assert first.content == "我先查一下项目里哪里用了 httpx"
    assert first.tool_calls[0].arguments == raw_arguments

    # 第二轮：第一轮的 assistant 轮回填 + 工具结果
    messages = [
        ChatMessage(role="user", content="项目里哪里用了 httpx？"),
        first.to_message(),
        ChatMessage(role="tool", content="src/pkg/client.py:12", tool_call_id="call-1"),
    ]
    _patch_http(monkeypatch, body=_chat_body(content="改完了"), sink=sink)
    client.chat(messages)

    sent = sink["payload"]["messages"]

    assert sent[1] == {
        "role": "assistant",
        "content": "我先查一下项目里哪里用了 httpx",
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search_code", "arguments": raw_arguments},
            }
        ],
    }

    # 逐字节：几个具体的失真方式都要单独排除
    assert sent[1]["reasoning_content"] == reasoning  # 未被改写 / 编码
    assert len(sent[1]["reasoning_content"]) == len(reasoning)  # 未被截断
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == raw_arguments  # 未被重编码
    assert sent[1]["tool_calls"][0]["id"] == "call-1"
    assert sent[1]["tool_calls"][0]["function"]["name"] == "search_code"
    assert sent[1]["content"] == "我先查一下项目里哪里用了 httpx"
    # 没有转成别的字段：思维链既没并进正文，也没跑到 tool 消息里
    assert reasoning not in sent[1]["content"]
    assert sent[2] == {
        "role": "tool",
        "content": "src/pkg/client.py:12",
        "tool_call_id": "call-1",
    }
    assert "reasoning_content" not in sent[2]


def test_no_reasoning_content_field_when_the_model_provided_none(monkeypatch, no_env_key) -> None:
    """非推理模型：第二轮不能凭空多出一个 reasoning_content。"""
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="改完了"), sink=sink)
    client = _client()

    turn = client.chat(_user())

    assert turn.reasoning_content is None

    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)
    client.chat([*_user(), turn.to_message()])

    assistant = sink["payload"]["messages"][1]

    assert assistant == {"role": "assistant", "content": "改完了"}
    assert "reasoning_content" not in assistant


def test_reasoning_content_is_never_written_to_logs(monkeypatch, no_env_key, caplog) -> None:
    """思维链只在请求体里回传，不进日志。"""
    sentinel = "SENTINEL_REASONING_TRACE"
    _patch_http(monkeypatch, body=_chat_body(content="好", reasoning_content=sentinel))
    client = _client()

    with caplog.at_level(logging.DEBUG):
        turn = client.chat(_user())
        client.chat([*_user(), turn.to_message()])

    assert sentinel not in caplog.text


# ------------------------------------------- 场景 E：arguments 是 JSON 字符串


def test_chat_preserves_arguments_as_a_raw_json_string(monkeypatch, no_env_key) -> None:
    """arguments 原样保留字符串：解析再序列化会改变字节，下一轮就对不上了。"""
    raw = '{ "query" : "httpx" , "limit": 5 }'
    _patch_http(monkeypatch, body=_chat_body(tool_calls=[_tool_call_payload(arguments=raw)]))

    turn = _client().chat(_user())

    assert turn.tool_calls[0].arguments == raw
    assert turn.tool_calls[0].parsed_arguments().arguments == {"query": "httpx", "limit": 5}


def test_chat_treats_missing_arguments_as_no_arguments(monkeypatch, no_env_key) -> None:
    _patch_http(
        monkeypatch,
        body=_chat_body(tool_calls=[_tool_call_payload("call-1", "inspect_project", "")]),
    )

    call = _client().chat(_user()).tool_calls[0]

    assert call.arguments == ""
    assert call.parsed_arguments().ok is True
    assert call.parsed_arguments().arguments == {}


def test_chat_reserializes_arguments_given_as_an_object(monkeypatch, no_env_key) -> None:
    """个别 OpenAI 兼容实现直接给对象；重新序列化而不是丢掉这个调用。"""
    payload = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "search_code", "arguments": {"query": "httpx"}},
    }
    _patch_http(monkeypatch, body=_chat_body(tool_calls=[payload]))

    call = _client().chat(_user()).tool_calls[0]

    assert json.loads(call.arguments) == {"query": "httpx"}


# ---------------------------------------- 场景 F：非法 JSON arguments 的安全处理


def test_chat_carries_broken_arguments_without_raising(monkeypatch, no_env_key) -> None:
    """模型给了一段坏 JSON 是可恢复的状况：客户端照常返回这一轮，不炸整轮对话。"""
    _patch_http(
        monkeypatch,
        body=_chat_body(
            tool_calls=[_tool_call_payload(arguments='{"query": "httpx"')],
            finish_reason="tool_calls",
        ),
    )

    turn = _client().chat(_user())

    assert turn.tool_calls[0].arguments == '{"query": "httpx"'
    parsed = turn.tool_calls[0].parsed_arguments()
    assert parsed.ok is False
    assert parsed.arguments == {}
    assert parsed.error


def test_chat_does_not_place_non_serializable_arguments_in_the_turn() -> None:
    """无法序列化的 arguments 属于服务端违约：报错，但不静默吞掉这个调用。

    直接对 _coerce_arguments 下单测：json.loads 只会产出 JSON 原生类型，
    所以这条分支走不到「从 HTTP 响应进来」的路径上，只能单独验证。
    """
    with pytest.raises(llm.DeepSeekResponseError, match="tool arguments"):
        deepseek._coerce_arguments(object())

    assert deepseek._coerce_arguments(None) == ""


def test_chat_accepts_arguments_that_are_not_valid_json(monkeypatch, no_env_key) -> None:
    raw = "not json at all"
    _patch_http(monkeypatch, body=_chat_body(tool_calls=[_tool_call_payload(arguments=raw)]))

    turn = _client().chat(_user())

    assert turn.tool_calls[0].arguments == raw  # 原样搬运，不预判
    assert turn.tool_calls[0].parsed_arguments().ok is False


# -------------------------------------------------- 场景 D：tool_calls 的形状校验


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [
        ({"id": "call-1"}, "tool_calls 不是数组"),
        (["not an object"], r"tool_calls\[0\] 不是对象"),
        ([{"id": "call-1", "type": "function"}], r"tool_calls\[0\] 缺少 function"),
        ([{"type": "function", "function": {"name": "search_code"}}], "缺少 id 或 function.name"),
        (
            [{"id": "call-1", "type": "function", "function": {"arguments": "{}"}}],
            "缺少 id 或 function.name",
        ),
        (
            [{"id": "  ", "type": "function", "function": {"name": "search_code"}}],
            "缺少 id 或 function.name",
        ),
    ],
)
def test_chat_rejects_malformed_tool_calls(
    monkeypatch, no_env_key, tool_calls: Any, expected: str
) -> None:
    """缺 id / name 的调用报错而不是跳过。

    跳过等于把模型的一个动作静默吞掉，调用方会看到「模型什么都没要求做」——
    那比报错难排查得多。

    每种畸形都断言**具体**原因，而不是只匹配 "tool_calls" 这个大类：只匹配大类
    的话，「静默跳过畸形调用」的实现会因为随后触发「既没有文本内容也没有
    tool_calls」而照样通过，断言看着管这件事、其实没管。
    """
    _patch_http(monkeypatch, body=_chat_body(tool_calls=tool_calls, finish_reason="tool_calls"))

    with pytest.raises(llm.DeepSeekResponseError, match=expected):
        _client().chat(_user())


def test_chat_returns_no_tool_calls_when_the_field_is_absent(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body=_chat_body(content="好"))

    assert _client().chat(_user()).tool_calls == []


def test_chat_rejects_a_null_message(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body=json.dumps({"choices": [{"message": None}]}))

    with pytest.raises(llm.DeepSeekResponseError, match="message"):
        _client().chat(_user())


def test_chat_rejects_a_non_json_response(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body="<html>502 Bad Gateway</html>")

    with pytest.raises(llm.DeepSeekResponseError, match="不是有效 JSON"):
        _client().chat(_user())


def test_chat_rejects_an_empty_response(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, body="")

    with pytest.raises(llm.DeepSeekResponseError, match="空响应"):
        _client().chat(_user())


# ------------------------------------- 场景 J：tools / tool_choice 进入请求体


def test_chat_converts_tool_specs_into_deepseek_tools(monkeypatch, no_env_key) -> None:
    """适配器负责 ToolSpec → DeepSeek tools[]：调用方只交 ToolSpec。

    断言的是**最终 payload**：转换位置从调用方搬进了适配器，请求体本身必须与
    旧版本逐键一致（同样的 {"type": "function", "function": {...}} 形状）。
    """
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)
    specs = build_default_registry().specs()

    _client().chat(_user(), tools=specs, tool_choice="auto")

    assert sink["payload"]["tools"] == [tool_spec_to_deepseek_function(spec) for spec in specs]
    assert [tool["function"]["name"] for tool in sink["payload"]["tools"]] == [
        "inspect_api",
        "inspect_project",
        "read_file",
        "search_code",
    ]
    assert sink["payload"]["tool_choice"] == "auto"
    assert json.loads(json.dumps(sink["payload"])) == sink["payload"]  # 请求体可序列化


def test_chat_sends_named_tool_choice(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client().chat(
        _user(),
        tools=[build_default_registry().specs()[0]],
        tool_choice={"type": "function", "function": {"name": "inspect_api"}},
    )

    assert sink["payload"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "inspect_api"},
    }


def test_chat_omits_tools_when_none_are_provided(monkeypatch, no_env_key) -> None:
    """「给一个空 tools 数组」与「完全不带 tools」语义不同，后者才是本轮不提供工具。"""
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client().chat(_user(), tools=[])
    assert "tools" not in sink["payload"]
    assert "tool_choice" not in sink["payload"]

    _client().chat(_user(), tools=None, tool_choice="auto")
    assert "tools" not in sink["payload"]
    # tool_choice 单独出现没有意义，不能光带它
    assert "tool_choice" not in sink["payload"]


def test_chat_only_sends_real_tool_schemas(monkeypatch, no_env_key) -> None:
    """下发给 API 的 tools 里不能夹带策略标记或凭据。"""
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client().chat(_user(), tools=build_default_registry().specs())

    dumped = json.dumps(sink["payload"]["tools"], ensure_ascii=False)
    assert "read_only" not in dumped
    for needle in ("api_key", "token", "secret", "password"):
        assert needle not in dumped.lower()


def test_chat_does_not_mutate_the_caller_s_tool_specs(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)
    specs = build_default_registry().specs()
    before = json.dumps([spec.model_dump() for spec in specs], ensure_ascii=False, sort_keys=True)

    _client().chat(_user(), tools=specs)

    # 转换会深拷贝 parameters：适配器改不到调用方手里的 ToolSpec
    assert (
        json.dumps([spec.model_dump() for spec in specs], ensure_ascii=False, sort_keys=True)
        == before
    )


# ------------------------------------------- temperature / timeout / json_mode


def test_chat_uses_the_instance_temperature_by_default(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client().chat(_user())

    assert sink["payload"]["temperature"] == 0.0
    assert sink["payload"]["model"] == "deepseek-flash"


def test_chat_temperature_can_be_overridden_per_call(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client(temperature=0.7).chat(_user(), temperature=0.2)

    assert sink["payload"]["temperature"] == 0.2


def test_chat_timeout_can_be_overridden_per_call(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client(timeout=60.0).chat(_user(), timeout=3.5)

    assert sink["timeout"] == 3.5


def test_chat_json_mode_and_response_format(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client().chat(_user())
    assert "response_format" not in sink["payload"]

    _client(json_mode=True).chat(_user())
    assert sink["payload"]["response_format"] == {"type": "json_object"}

    # 显式传入优先于 json_mode
    _client(json_mode=True).chat(_user(), response_format={"type": "text"})
    assert sink["payload"]["response_format"] == {"type": "text"}


def test_chat_uses_the_configured_base_url(monkeypatch, no_env_key) -> None:
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, body=_chat_body(content="好"), sink=sink)

    _client(base_url="https://proxy.example.com/v1").chat(_user())

    assert sink["url"] == "https://proxy.example.com/v1/chat/completions"
    assert sink["headers"]["Authorization"] == "Bearer test-key"
    assert sink["headers"]["Content-Type"] == "application/json"


# ------------------------------------------------- 场景 K：异常 / 返回值不泄漏 Key


def test_chat_api_key_never_leaks_into_errors(monkeypatch, no_env_key) -> None:
    _patch_http(
        monkeypatch,
        status=401,
        body=json.dumps({"error": {"message": f"bad token {SECRET} for your account"}}),
    )

    with pytest.raises(llm.DeepSeekAPIError) as excinfo:
        llm.DeepSeekLLMClient(api_key=SECRET).chat(_user())

    assert SECRET not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_chat_api_key_never_leaks_into_the_returned_turn(monkeypatch, no_env_key) -> None:
    """服务端把 Key 原样塞回响应里，也不能让 Key 进到返回值。"""
    _patch_http(
        monkeypatch,
        body=_chat_body(
            content=f"你的 Key 是 {SECRET}",
            tool_calls=[_tool_call_payload("call-1", "search_code", f'{{"q": "{SECRET}"}}')],
            finish_reason=f"tool_calls {SECRET}",
            reasoning_content=f"用户 Key={SECRET}",
        ),
    )

    turn = llm.DeepSeekLLMClient(api_key=SECRET).chat(_user())
    dumped = json.dumps(turn.model_dump(), ensure_ascii=False)

    assert SECRET not in dumped
    assert "***" in turn.content
    assert "***" in turn.tool_calls[0].arguments
    assert "***" in (turn.reasoning_content or "")
    assert "***" in (turn.finish_reason or "")


def test_chat_api_key_never_leaks_into_the_request_log_of_a_failure(
    monkeypatch, no_env_key
) -> None:
    """连接失败时只有异常类型名，不带 URL / header / Key。"""
    _patch_http(monkeypatch, error=urllib.error.URLError(f"refused, key={SECRET}"))

    with pytest.raises(llm.DeepSeekLLMError) as excinfo:
        llm.DeepSeekLLMClient(api_key=SECRET).chat(_user())

    assert SECRET not in str(excinfo.value)
    assert "URLError" in str(excinfo.value)


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_chat_error_status_codes(monkeypatch, no_env_key, status: int) -> None:
    _patch_http(monkeypatch, status=status, body=json.dumps({"error": {"message": "nope"}}))

    with pytest.raises(llm.DeepSeekAPIError) as excinfo:
        _client().chat(_user())

    assert excinfo.value.status == status


def test_chat_timeout_raises(monkeypatch, no_env_key) -> None:
    _patch_http(monkeypatch, error=TimeoutError("timed out"))

    with pytest.raises(llm.DeepSeekTimeoutError, match="超时"):
        _client().chat(_user())


# --------------------------------- 场景 M/N：不执行工具、不发生真实工具调用


def test_chat_does_not_execute_tools(monkeypatch, no_env_key, tmp_path) -> None:
    """参数里写的是删除命令与敏感路径，客户端只把它们当字符串搬运。"""
    hostile = json.dumps({"path": str(tmp_path / "secret.txt"), "cmd": "rm -rf /"})
    sink: dict[str, Any] = {}
    _patch_http(
        monkeypatch,
        body=_chat_body(
            tool_calls=[_tool_call_payload("call-1", "read_file", hostile)],
            finish_reason="tool_calls",
        ),
        sink=sink,
    )

    turn = _client().chat(_user())

    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == hostile  # 逐字节搬运
    assert turn.tool_calls[0].parsed_arguments().arguments["cmd"] == "rm -rf /"


def test_chat_only_talks_to_deepseek(monkeypatch, no_env_key) -> None:
    """唯一的出网口是 _http_post，且目标只能是 DeepSeek 的 /chat/completions。

    哨兵：把 urllib.request.urlopen 换成炸弹。若客户端绕过 _http_post 自己发请求
    （比如去"执行"模型给的工具目标地址），这个用例会直接炸掉。
    """
    seen: list[str] = []

    def bomb(*args, **kwargs):
        raise AssertionError("chat() 不允许直接使用 urllib.request.urlopen")

    monkeypatch.setattr(deepseek.urllib.request, "urlopen", bomb)

    def fake_post(url, payload, headers, timeout):
        seen.append(url)
        return 200, _chat_body(
            tool_calls=[
                _tool_call_payload(
                    "call-1",
                    "read_file",
                    json.dumps(
                        {"url": "http://127.0.0.1:9999/admin", "path": "file:///etc/passwd"}
                    ),
                )
            ],
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(deepseek, "_http_post", fake_post)

    _client().chat(_user("把 http://127.0.0.1:9999/admin 抓下来"))

    assert seen == ["https://api.deepseek.com/chat/completions"]
    assert "127.0.0.1" not in seen[0]
    assert "9999" not in seen[0]


def test_chat_does_not_read_a_file_named_in_tool_arguments(
    monkeypatch, no_env_key, tmp_path
) -> None:
    """参数给的是一个**真实存在**的文件路径，客户端也不会把它打开。

    用一个真实文件而不是 /etc/shadow 之类的假路径：只有文件真的在那，
    「没读它」才是一个可证伪的断言。
    """
    sentinel = "SENTINEL_FILE_CONTENT_MUST_NOT_LEAK"
    target = tmp_path / "secret.txt"
    target.write_text(sentinel, encoding="utf-8")
    _patch_http(
        monkeypatch,
        body=_chat_body(
            tool_calls=[_tool_call_payload("c", "read_file", json.dumps({"path": str(target)}))]
        ),
    )

    turn = _client().chat(_user())
    dumped = json.dumps(turn.model_dump(), ensure_ascii=False)

    assert sentinel not in dumped  # 文件内容没有跟着参数一起被搬出来
    assert target.read_text(encoding="utf-8") == sentinel  # 文件没被动过
    assert json.loads(turn.tool_calls[0].arguments)["path"] == str(target)  # 只是字符串


# ---------------------------------------- 依赖方向：chat() 不认识 ToolRegistry


def test_client_module_imports_only_allowlisted_modules() -> None:
    imported = _imported_modules(Path(deepseek.__file__).resolve())

    extra = imported - ALLOWED_CLIENT_IMPORTS
    assert not extra, f"llm/deepseek.py 出现了白名单之外的 import：{sorted(extra)}"


def test_client_module_knows_only_the_tool_spec_data_contract() -> None:
    """适配器可以依赖纯数据型的 ToolSpec，但不能依赖工具的执行侧。

    chat() 收到的 tools 是 provider-neutral 的 ToolSpec，所以
    integration_agent.tools.models 是**必需**的依赖（适配器的职责就是把它转成
    DeepSeek wire format）。需要挡住的是另一半：ToolRegistry（工具的查找与执行
    入口）、AgentLoop / ToolUsingPlanner（调用方）——适配器一旦认识它们，
    "谁负责转 schema" 这条边界就又糊回去了。
    """
    imported = _imported_modules(Path(deepseek.__file__).resolve())

    assert "integration_agent.tools.models" in imported  # 转换的输入类型
    assert "integration_agent.tools" not in imported  # 包级：会带出 registry 与具体工具
    assert "integration_agent.tools.registry" not in imported
    assert "integration_agent.agent.agent_loop" not in imported
    assert "integration_agent.agent.tool_planner" not in imported
    for module in imported:
        if module.startswith("integration_agent.tools."):
            assert module == "integration_agent.tools.models", module
        assert not module.startswith("integration_agent.agent."), module
        assert not module.startswith("integration_agent.repair."), module


def test_client_module_has_no_execution_capability() -> None:
    imported = _imported_modules(Path(deepseek.__file__).resolve())

    for forbidden in ("subprocess", "socket", "shutil", "pathlib", "tempfile", "importlib"):
        assert forbidden not in imported, f"llm/deepseek.py 引入了能力模块 {forbidden}"


# -------------------------------------------------- 场景 15：ToolCallingClient


def test_deepseek_client_conforms_to_tool_calling_client(no_env_key) -> None:
    assert isinstance(_client(), ToolCallingClient)


def test_fake_client_and_real_client_are_interchangeable(monkeypatch, no_env_key) -> None:
    """两者都满足 ToolCallingClient：Agent Loop 可以在离线与真实之间切换。"""
    _patch_http(monkeypatch, body=_chat_body(content="好"))
    clients: list[ToolCallingClient] = [
        _client(),
        FakeToolCallingClient(AssistantTurn(content="好")),
    ]

    for client in clients:
        turn = client.chat(_user())

        assert isinstance(turn, AssistantTurn)


def test_the_network_guard_actually_fires(monkeypatch, no_env_key) -> None:
    """哨兵自检：不打桩 _http_post 时会炸，而不是真的发请求。

    没有这个用例，「所有用例都离线」这句话就无从验证——哨兵可能根本没接上。
    """
    with pytest.raises(AssertionError, match="不允许真实网络请求"):
        _client().chat(_user())


# ------------------------------------------------ 场景 L：generate() 无回归


def test_generate_still_uses_the_plain_single_prompt_payload(monkeypatch, no_env_key) -> None:
    """抽出 _send() 之后 generate() 的请求体一个字节都没变。"""
    sink: dict[str, Any] = {}
    _patch_http(monkeypatch, sink=sink)

    assert _client().generate("修复 import 错误") == "你好世界"

    assert sink["url"] == "https://api.deepseek.com/chat/completions"
    assert sink["payload"] == {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "修复 import 错误"}],
        "temperature": 0.0,
    }
    assert sink["headers"]["Authorization"] == "Bearer test-key"
    assert sink["timeout"] == 60.0
