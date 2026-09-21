"""integration_agent.llm 的消息模型、ToolSpec 转换与测试用假客户端。

场景分组：
    1. ChatMessage / AssistantTurn 校验
    2. ToolSpec → DeepSeek function schema（转换现在属于适配器 llm/deepseek.py）
    3. tool arguments 的不可信解析
    4. FakeToolCallingClient 与 ToolCallingClient 协议
    5. 安全边界（AST 白名单 + 不接触仓库 / 文件系统 / 网络）

跨模块的依赖方向（llm 不依赖 agent / repair / api_server）由
tests/test_llm_architecture.py 单独把关，本文件只管单个模块内部。

这些用例全部离线、不读环境变量、不需要 API Key。
"""

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from integration_agent.llm import (
    AssistantTurn,
    ChatMessage,
    FakeToolCallingClient,
    ToolCallingClient,
    ToolCallRequest,
    client,
    models,
    parse_tool_arguments,
    tool_spec_to_deepseek_function,
)
from integration_agent.tools import build_default_registry
from integration_agent.tools.models import ToolSpec

SECRET_SENTINEL = "SENTINEL_SECRET_MUST_NOT_LEAK"

# llm/models.py 允许 import 的全部模块。做成白名单而不是黑名单：新增一个
# import 必须是有意识的决定，而不是"没被黑名单列到"就悄悄放行。
# 特别地，这张表里没有 tools.models：数据模型连 ToolSpec 都不需要认识。
ALLOWED_MODELS_IMPORTS = {
    "json",
    "typing",
    "pydantic",
}

# llm/client.py 允许 import 的全部模块。ToolSpec 是**纯数据**声明
# （"要提供哪些工具"本来就与 provider 无关），所以它可以出现在契约层；
# tools.registry 不行 —— 契约层不认识 ToolRegistry，也没有执行工具的入口。
ALLOWED_CLIENT_IMPORTS = {
    "collections.abc",
    "typing",
    "integration_agent.llm.models",
    "integration_agent.tools.models",
}

DEFAULT_TOOL_NAMES = ("inspect_api", "inspect_project", "read_file", "search_code")

# 真正算泄漏的形状。刻意不含裸的 "env" / "token" / "secret" 三个字母：
# 工具描述里提醒模型「不要读 .env」是正常的说明文字，不是泄漏。
_CREDENTIAL_NEEDLES = ("api_key", "apikey", "authorization", "bearer", "password", "sk-")


def _leaked_credentials(text: str) -> list[str]:
    lowered = text.lower()
    return [needle for needle in _CREDENTIAL_NEEDLES if needle in lowered]


def _tool_call(
    suffix: str = "1", name: str = "search_code", arguments: str = ""
) -> ToolCallRequest:
    return ToolCallRequest(id=f"call-{suffix}", name=name, arguments=arguments)


def _imported_modules(source_file: Path) -> set[str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


# ---------------------------------------------------------------------------
# 场景 1：ChatMessage / AssistantTurn 校验
# ---------------------------------------------------------------------------


def test_system_and_user_messages() -> None:
    system = ChatMessage(role="system", content="你是集成助手")
    user = ChatMessage(role="user", content="把 httpx 换成官方 SDK")

    assert system.content == "你是集成助手"
    assert user.role == "user"
    assert system.tool_calls == []
    assert system.tool_call_id is None


def test_assistant_may_carry_content_only() -> None:
    message = ChatMessage(role="assistant", content="完成")

    assert message.content == "完成"
    assert message.tool_calls == []


def test_assistant_may_carry_tool_calls_only() -> None:
    message = ChatMessage(role="assistant", tool_calls=[_tool_call()])

    # 只发起工具调用、不带正文，是正常形态而不是缺字段
    assert message.content == ""
    assert len(message.tool_calls) == 1


def test_assistant_may_carry_both() -> None:
    message = ChatMessage(role="assistant", content="我先查一下", tool_calls=[_tool_call()])

    assert message.content == "我先查一下"
    assert message.tool_calls[0].name == "search_code"


def test_assistant_with_neither_content_nor_tool_calls_is_rejected() -> None:
    with pytest.raises(ValidationError, match="content 或 tool_calls"):
        ChatMessage(role="assistant")


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        ChatMessage(role="tool", content="结果")

    message = ChatMessage(role="tool", content="结果", tool_call_id="call-1")
    assert message.tool_call_id == "call-1"


@pytest.mark.parametrize("role", ["system", "user"])
def test_system_and_user_require_nonblank_content(role: str) -> None:
    with pytest.raises(ValidationError, match="非空 content"):
        ChatMessage(role=role, content="   ")


@pytest.mark.parametrize("role", ["system", "user", "tool"])
def test_only_assistant_may_carry_tool_calls(role: str) -> None:
    kwargs: dict[str, Any] = {"role": role, "content": "x"}
    if role == "tool":
        kwargs["tool_call_id"] = "call-1"

    with pytest.raises(ValidationError, match="tool_calls"):
        ChatMessage(**kwargs, tool_calls=[_tool_call()])


@pytest.mark.parametrize("role", ["system", "user", "tool"])
def test_only_assistant_may_carry_reasoning_content(role: str) -> None:
    kwargs: dict[str, Any] = {"role": role, "content": "x"}
    if role == "tool":
        kwargs["tool_call_id"] = "call-1"

    with pytest.raises(ValidationError, match="reasoning_content"):
        ChatMessage(**kwargs, reasoning_content="思维链")


def test_assistant_message_may_carry_reasoning_content() -> None:
    message = ChatMessage(role="assistant", content="x", reasoning_content="思维链")

    assert message.reasoning_content == "思维链"


def test_tool_call_id_is_reserved_for_the_tool_role() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        ChatMessage(role="user", content="x", tool_call_id="call-1")


@pytest.mark.parametrize("role", ["tool", "assistant", "user", "nonsense"])
def test_unknown_or_blank_call_ids_are_rejected(role: str) -> None:
    if role == "nonsense":
        with pytest.raises(ValidationError):
            ChatMessage(role=role, content="x")  # type: ignore[arg-type]
        return
    if role != "tool":
        return
    with pytest.raises(ValidationError, match="tool_call_id"):
        ChatMessage(role="tool", content="x", tool_call_id="   ")


@pytest.mark.parametrize("missing", ["id", "name"])
def test_tool_call_request_requires_id_and_name(missing: str) -> None:
    kwargs: dict[str, str] = {"id": "call-1", "name": "search_code"}
    kwargs[missing] = "  "

    with pytest.raises(ValidationError, match="不能为空"):
        ToolCallRequest(**kwargs)


def test_assistant_turn_defaults() -> None:
    turn = AssistantTurn()

    assert turn.content == ""
    assert turn.tool_calls == []
    assert turn.finish_reason is None
    assert turn.reasoning_content is None


def test_assistant_turn_keeps_every_field() -> None:
    turn = AssistantTurn(
        content="",
        tool_calls=[_tool_call()],
        finish_reason="tool_calls",
        reasoning_content="我想想……",
    )

    assert turn.finish_reason == "tool_calls"
    assert turn.reasoning_content == "我想想……"
    assert turn.tool_calls[0].id == "call-1"


def test_to_message_round_trips_tool_calls() -> None:
    turn = AssistantTurn(
        content="",
        tool_calls=[_tool_call("1"), _tool_call("2", name="read_file", arguments='{"path": "a"}')],
        finish_reason="tool_calls",
    )

    message = turn.to_message()

    # tool_calls 必须原样回填：下一轮的 tool 消息靠 tool_call_id 与它对应
    assert message.role == "assistant"
    assert message.tool_calls == turn.tool_calls
    assert message.content == ""


def test_to_message_carries_reasoning_content() -> None:
    turn = AssistantTurn(
        content="",
        reasoning_content="reasoning-123",
        tool_calls=[_tool_call()],
        finish_reason="tool_calls",
    )

    message = turn.to_message()

    # thinking mode + tool calling：思维链必须跟着这一轮一起回填，
    # 否则后续请求会被 API 以 HTTP 400 拒绝
    assert message.reasoning_content == "reasoning-123"
    assert message.tool_calls == turn.tool_calls


def test_to_message_omits_reasoning_content_when_there_is_none() -> None:
    turn = AssistantTurn(content="完成")

    message = turn.to_message()

    # 非推理模型没有这个字段：不能凭空生成一个空串。
    # 「请求体里到底有没有这个键」由序列化那一层决定，见
    # tests/test_deepseek_client.py::test_no_reasoning_content_field_when_...
    assert turn.reasoning_content is None
    assert message.reasoning_content is None


def test_reasoning_content_is_carried_byte_for_byte() -> None:
    """不重新编码、不截断、不修剪：换个字节下一轮就对不上了。"""
    raw = '  第一步：{"a": 1}\n第二步：<b>&amp;</b>\t收尾空格  '

    message = AssistantTurn(content="x", reasoning_content=raw).to_message()

    assert message.reasoning_content == raw
    assert len(message.reasoning_content) == len(raw)


def test_reasoning_content_is_not_merged_into_content() -> None:
    message = AssistantTurn(content="结论", reasoning_content=SECRET_SENTINEL).to_message()

    # 转成别的字段（比如并进 content）同样会让 API 认不出来
    assert message.content == "结论"
    assert SECRET_SENTINEL not in message.content
    assert message.reasoning_content == SECRET_SENTINEL


def test_to_message_of_an_empty_turn_is_rejected() -> None:
    # 空的一轮既没有正文也没有动作，发回给 API 没有意义
    with pytest.raises(ValidationError, match="content 或 tool_calls"):
        AssistantTurn().to_message()


# ---------------------------------------------------------------------------
# 场景 2：ToolSpec → DeepSeek function schema
# ---------------------------------------------------------------------------


def test_tool_spec_converts_to_function_schema() -> None:
    spec = ToolSpec(
        name="search_code",
        description="在项目里搜索",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    assert tool_spec_to_deepseek_function(spec) == {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "在项目里搜索",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def test_conversion_drops_read_only() -> None:
    spec = ToolSpec(name="read_file", description="读文件", read_only=False)

    function = tool_spec_to_deepseek_function(spec)["function"]

    # read_only 是我们这边的策略标记，API 不认识它
    assert "read_only" not in function
    assert set(function) == {"name", "description", "parameters"}


def test_conversion_is_json_serializable_and_stable() -> None:
    spec = ToolSpec(name="t", description="d")

    first = tool_spec_to_deepseek_function(spec)
    second = tool_spec_to_deepseek_function(spec)

    assert json.loads(json.dumps(first, ensure_ascii=False)) == first
    assert first == second
    # 键顺序也稳定，两次序列化逐字节相同
    assert json.dumps(first) == json.dumps(second)


def test_conversion_fills_in_a_complete_object_schema() -> None:
    # ToolSpec 允许 parameters 为空 dict，但 API 侧期望一个可读的 object schema
    assert tool_spec_to_deepseek_function(ToolSpec(name="inspect_project", description="d"))[
        "function"
    ]["parameters"] == {"type": "object", "properties": {}}


def test_conversion_does_not_alias_the_spec_parameters() -> None:
    spec = ToolSpec(name="t", description="d", parameters={"type": "object", "properties": {}})

    converted = tool_spec_to_deepseek_function(spec)
    converted["function"]["parameters"]["properties"]["injected"] = {"type": "string"}
    converted["function"]["parameters"]["required"] = ["injected"]

    assert spec.parameters == {"type": "object", "properties": {}}


def test_conversion_of_every_default_tool_is_clean() -> None:
    specs = build_default_registry().specs()

    assert tuple(spec.name for spec in specs) == DEFAULT_TOOL_NAMES
    for spec in specs:
        converted = tool_spec_to_deepseek_function(spec)
        dumped = json.dumps(converted, ensure_ascii=False)

        assert converted["type"] == "function"
        assert converted["function"]["name"] == spec.name
        assert converted["function"]["description"] == spec.description
        assert converted["function"]["parameters"]["type"] == "object"
        # 不下发策略标记，也不夹带任何凭据
        assert "read_only" not in dumped
        assert not _leaked_credentials(dumped), f"{spec.name} 的 schema 里出现了凭据字段"


def test_the_credential_check_can_actually_fail() -> None:
    """阴性对照：上面的检查抓得住真凭据，不是一句永远为真的断言。

    工具描述里正常出现「.env」这类词是允许的（那是给模型的提醒），所以这里
    只查真正的凭据形状，而不是「出现 env 三个字母就报错」。
    """
    leaked = tool_spec_to_deepseek_function(
        ToolSpec(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"api_key": {"type": "string", "default": "sk-live-123"}},
            },
        )
    )

    assert _leaked_credentials(json.dumps(leaked))


def test_tool_descriptions_may_still_warn_about_credentials() -> None:
    # read_file 的说明里提醒模型不要读 .env —— 这是描述，不是泄漏
    description = next(
        spec.description for spec in build_default_registry().specs() if spec.name == "read_file"
    )

    assert ".env" in description
    assert not _leaked_credentials(
        json.dumps(
            tool_spec_to_deepseek_function(ToolSpec(name="read_file", description=description)),
            ensure_ascii=False,
        )
    )


def test_conversion_carries_no_absolute_paths() -> None:
    for spec in build_default_registry().specs():
        parameters = json.dumps(tool_spec_to_deepseek_function(spec)["function"]["parameters"])

        # 工具 schema 里不该出现任何绝对路径：模型只需要知道参数长什么样
        assert ":/" not in parameters.replace("https://", "")
        assert "\\\\" not in parameters
        assert "/server/" not in parameters
        assert "/home/" not in parameters


# ---------------------------------------------------------------------------
# 场景 3：tool arguments 的不可信解析
# ---------------------------------------------------------------------------


def test_valid_arguments_are_parsed() -> None:
    result = parse_tool_arguments('{"query": "httpx", "max_results": 5}')

    assert result.ok is True
    assert result.arguments == {"query": "httpx", "max_results": 5}
    assert result.error is None


def test_tool_call_request_exposes_the_same_parse() -> None:
    call = _tool_call(arguments='{"query": "httpx"}')

    assert call.parsed_arguments().arguments == {"query": "httpx"}


@pytest.mark.parametrize("raw", ["", "   ", "\n"])
def test_empty_arguments_mean_no_arguments(raw: str) -> None:
    # 零参数工具的正常形态，不是错误
    result = parse_tool_arguments(raw)

    assert result.ok is True
    assert result.arguments == {}


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "{'single': 'quotes'}",
        '{"query": "httpx"',
        '{"a": 1,}',
        '{"a": }',
        '{"query": "httpx"}}',
    ],
)
def test_invalid_json_returns_a_structured_error(raw: str) -> None:
    result = parse_tool_arguments(raw)

    assert result.ok is False
    assert result.arguments == {}
    assert result.error
    assert isinstance(result.error, str)


@pytest.mark.parametrize("raw", ["[]", "3", '"text"', "null", "true", '[{"a": 1}]'])
def test_non_object_arguments_are_rejected(raw: str) -> None:
    result = parse_tool_arguments(raw)

    assert result.ok is False
    assert "JSON 对象" in (result.error or "")


def test_parse_errors_do_not_echo_the_payload() -> None:
    """坏 JSON 的报错不把原文搬出来。

    arguments 是服务端给的不可信文本。把原文拼进错误文本，就等于开了一条
    「服务端 → 错误信息 → 日志 / LLM 上下文」的通道；不搬它，这条通道从结构上
    就不存在，不需要靠脱敏去堵。
    """
    result = parse_tool_arguments('{"api_key": "' + SECRET_SENTINEL + '", bad}')

    assert result.ok is False
    assert SECRET_SENTINEL not in (result.error or "")


@pytest.mark.parametrize(
    "raw",
    ["", "  ", "{}", "[]", "null", "0", "not json", '{"a": [1, 2]}', "\x00", "🦄"],
)
def test_parsing_never_raises(raw: str) -> None:
    result = parse_tool_arguments(raw)

    assert isinstance(result.ok, bool)
    if not result.ok:
        assert result.arguments == {}


def test_parsed_arguments_are_plain_data() -> None:
    """解析结果只是数据：没有任何可执行对象、路径或文件句柄。"""
    result = parse_tool_arguments('{"path": "../../etc/passwd", "cmd": "rm -rf /"}')

    assert result.ok is True
    assert result.arguments["cmd"] == "rm -rf /"  # 只是一个字符串值，没有被执行
    assert all(
        isinstance(value, (str, int, float, bool, list, dict, type(None)))
        for value in result.arguments.values()
    )


# ---------------------------------------------------------------------------
# 场景 4：FakeToolCallingClient 与协议
# ---------------------------------------------------------------------------


def test_fake_client_walks_through_a_tool_call_then_a_final_answer() -> None:
    """下一阶段 Agent Loop 要用的最小回路：先要工具，再给答案。"""
    fake = FakeToolCallingClient(
        [
            AssistantTurn(
                tool_calls=[
                    ToolCallRequest(id="call-1", name="search_code", arguments='{"query": "httpx"}')
                ],
                finish_reason="tool_calls",
            ),
            AssistantTurn(content="httpx 出现在 src/pkg/client.py", finish_reason="stop"),
        ]
    )
    messages = [ChatMessage(role="user", content="项目里哪里用了 httpx？")]

    first = fake.chat(messages, tools=[ToolSpec(name="search_code", description="搜索")])

    assert first.finish_reason == "tool_calls"
    assert first.tool_calls[0].parsed_arguments().arguments == {"query": "httpx"}

    messages.append(first.to_message())
    messages.append(ChatMessage(role="tool", content="src/pkg/client.py:12", tool_call_id="call-1"))

    second = fake.chat(messages)

    assert second.finish_reason == "stop"
    assert second.tool_calls == []
    assert "httpx" in second.content


def test_fake_client_records_requests_and_tools() -> None:
    fake = FakeToolCallingClient(AssistantTurn(content="好"))
    tools = [ToolSpec(name="search_code", description="搜索")]
    messages = [ChatMessage(role="user", content="你好")]

    fake.chat(messages, tools=tools, tool_choice="auto")

    assert fake.calls == 1
    assert fake.requests[0] == messages
    # 记录的是调用方给的 ToolSpec 原样：假客户端也不做任何 provider 转换
    assert fake.tools[0] == tools
    assert all(isinstance(spec, ToolSpec) for spec in fake.tools[0])


def test_fake_client_returns_the_same_turn_when_only_one_is_given() -> None:
    fake = FakeToolCallingClient(AssistantTurn(content="固定回答"))

    assert fake.chat([]).content == "固定回答"
    assert fake.chat([]).content == "固定回答"
    assert fake.calls == 2


def test_fake_client_raises_when_called_more_than_expected() -> None:
    fake = FakeToolCallingClient([AssistantTurn(content="一"), AssistantTurn(content="二")])

    assert fake.chat([]).content == "一"
    assert fake.chat([]).content == "二"
    with pytest.raises(AssertionError, match="超过预期"):
        fake.chat([])


def test_default_fake_client_is_empty() -> None:
    with pytest.raises(AssertionError, match="超过预期"):
        FakeToolCallingClient().chat([])


def test_fake_client_satisfies_the_protocol() -> None:
    assert isinstance(FakeToolCallingClient(AssistantTurn(content="x")), ToolCallingClient)


def test_fake_client_does_not_execute_tools() -> None:
    """假客户端只回放预设结果：参数里写了什么命令都不会被执行。"""
    fake = FakeToolCallingClient(
        AssistantTurn(tool_calls=[_tool_call(arguments='{"cmd": "rm -rf /", "path": "../../etc"}')])
    )

    turn = fake.chat([ChatMessage(role="user", content="x")])

    assert turn.tool_calls[0].arguments == '{"cmd": "rm -rf /", "path": "../../etc"}'


# ---------------------------------------------------------------------------
# 场景 5：安全边界
# ---------------------------------------------------------------------------


def test_models_module_imports_only_allowlisted_modules() -> None:
    imported = _imported_modules(Path(models.__file__).resolve())

    extra = imported - ALLOWED_MODELS_IMPORTS
    assert not extra, f"llm/models.py 出现了白名单之外的 import：{sorted(extra)}"


def test_models_module_has_no_project_dependency_at_all() -> None:
    """数据模型是纯的：连 ToolSpec 都不认识，更不会碰 tools 包。"""
    imported = _imported_modules(Path(models.__file__).resolve())

    assert not {item for item in imported if item.startswith("integration_agent")}


def test_client_module_imports_only_allowlisted_modules() -> None:
    imported = _imported_modules(Path(client.__file__).resolve())

    extra = imported - ALLOWED_CLIENT_IMPORTS
    assert not extra, f"llm/client.py 出现了白名单之外的 import：{sorted(extra)}"


def test_client_module_does_not_know_the_tool_registry() -> None:
    imported = _imported_modules(Path(client.__file__).resolve())

    # 契约层只认识 ToolSpec（数据契约），不认识 ToolRegistry（执行入口）
    assert "integration_agent.tools.registry" not in imported
    assert "integration_agent.tools" not in imported
    assert "integration_agent.tools.models" in imported


@pytest.mark.parametrize("module", [models, client])
def test_contract_modules_have_no_execution_or_network_capability(module) -> None:
    imported = _imported_modules(Path(module.__file__).resolve())

    for forbidden in (
        "socket",
        "subprocess",
        "os",
        "shutil",
        "pathlib",
        "urllib.request",
        "http.client",
        "requests",
        "httpx",
        "asyncio",
        "tempfile",
        "pickle",
    ):
        assert forbidden not in imported, f"{Path(module.__file__).name} 引入了能力模块 {forbidden}"


def test_conversion_reads_only_the_spec_fields() -> None:
    """转换函数只搬运三个字段，不读 ToolSpec 之外的任何东西（没有 IO 可言）。"""
    spec = ToolSpec(name="t", description="d", parameters={"type": "object", "properties": {}})

    converted = tool_spec_to_deepseek_function(spec)

    assert converted["function"]["name"] == spec.name
    assert converted["function"]["description"] == spec.description
    assert converted["function"]["parameters"] == spec.parameters
