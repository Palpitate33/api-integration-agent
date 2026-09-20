"""Agent Loop 的行为与安全边界测试（E2 Commit 5）。

**本文件不调用真实 DeepSeek，也不发任何网络请求。** 所有用例都由
FakeToolCallingClient 驱动（E2 Commit 4 的既有假客户端，不另造一个），
工具用两种：记录调用的 spy 工具（隔离回路本身），以及真实注册表
（证明回路能驱动真的只读工具）。

覆盖用户要求的 25 个场景：
    1.  直接 final，无工具调用
    2.  一次 tool call → final
    3.  两轮 tool call → final（inspect_project → search_code → final）
    4.  一轮多个 tool calls
    5.  工具不存在
    6.  非法 JSON arguments
    7.  arguments 不是 object
    8.  tool result 失败
    9.  连续 tool error 达到上限
    10. 重复 tool call
    11. max_tool_calls
    12. max_turns
    13. observation budget
    14. history budget
    15. LLM 抛异常
    16. malformed assistant response
    17. tool_call id 原样保留
    18. reasoning_content 原样保留
    19. assistant tool_calls 原样保留
    20. 不执行未知工具
    21. 不执行第二次重复工具
    22. 工具执行顺序保持模型给出的顺序
    23. AgentLoop 不写文件
    24. AgentLoop 不调用 shell
    25. AgentLoop 不调用网络

以及本实现特有的护栏：不伪造 tool call、失败有界、固定安全错误文案、
状态是显式枚举、循环里没有 while True。
"""

import ast
import hashlib
import json
import logging
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from integration_agent.agent import agent_loop as loop_module
from integration_agent.agent.agent_loop import (
    LLM_CALL_FAILED,
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRunner,
)
from integration_agent.agent.llm import (
    AssistantTurn,
    ChatMessage,
    FakeToolCallingClient,
    ToolCallRequest,
)
from integration_agent.api import APIEndpoint, APIInfo
from integration_agent.repository import scan_repository
from integration_agent.tools import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_default_registry,
)

# 工具 schema / 错误信息里都不该出现的东西
CREDENTIAL_NEEDLES = ("api_key", "apikey", "authorization", "bearer", "password", "sk-")
# 模块允许 import 的东西：只读回路不需要任何 IO 能力
ALLOWED_LOOP_IMPORTS = {
    "hashlib",
    "json",
    "logging",
    "typing",
    "pydantic",
    "integration_agent.agent.llm",
    "integration_agent.tools.models",
    "integration_agent.tools.registry",
}
FORBIDDEN_SHELL_MODULES = ("subprocess", "os", "pty", "commands", "shutil")
FORBIDDEN_NETWORK_MODULES = ("socket", "urllib", "http", "requests", "httpx", "aiohttp")

API_INFO = APIInfo(
    name="Demo Petstore API",
    version="1.0.0",
    base_url="https://petstore.example.com/v1",
    endpoints=[
        APIEndpoint(method="GET", path="/pets", operation_id="listPets", summary="列出所有宠物")
    ],
)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch) -> None:
    """出网哨兵：单元测试里不允许真的建立连接。

    模块 import 白名单已经排除网络库，但那证明的是"源码里没有"，这条证明的是
    "这次运行里没有"——两者互补，缺一不可。
    """

    def bomb(*args, **kwargs):
        raise AssertionError("Agent Loop 的单元测试不允许真实网络请求")

    monkeypatch.setattr(socket.socket, "connect", bomb, raising=False)
    monkeypatch.setattr(socket.socket, "connect_ex", bomb, raising=False)


# ------------------------------------------------------------------ helpers


def _project_tree(root: Path) -> None:
    """造一个最小但完整的 Python 项目。"""
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sample-project"\nversion = "0.1.0"\ndependencies = ["httpx>=0.27"]\n',
        encoding="utf-8",
    )
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "client.py").write_text("def fetch_pets():\n    return []\n", encoding="utf-8")


def _context(root: Path) -> ToolContext:
    project = scan_repository(root)
    return ToolContext(
        project_root=root,
        project=project,
        api=API_INFO,
        known_files=set(project.python_files),
    )


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    """目录内每个文件的大小 / mtime / 内容哈希，用于证明"什么都没被改动"。"""
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return snapshot


class _SpyTool:
    """记录每次调用的只读假工具：让回路的断言与具体工具的行为彻底解耦。

    order_log 是所有 spy 共享的列表，用来证明"执行顺序 = 模型给出的顺序"。
    """

    def __init__(
        self,
        name: str,
        *,
        content: str | None = None,
        error: str | None = None,
        order_log: list[str] | None = None,
    ) -> None:
        self.spec = ToolSpec(name=name, description=f"spy tool {name}")
        self.invocations: list[dict] = []
        self._content = content if content is not None else f"ok:{name}"
        self._error = error
        self._order_log = order_log

    def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append(dict(args))
        if self._order_log is not None:
            self._order_log.append(self.spec.name)
        if self._error is not None:
            return ToolResult(
                call_id="", tool=self.spec.name, ok=False, content="", error=self._error
            )
        # call_id 刻意留空：工具并不知道自己的 id，由回路用模型给的 id 补上
        return ToolResult(call_id="", tool=self.spec.name, ok=True, content=self._content)


def _call(call_id: str, name: str, arguments: str = "{}") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _tool_turn(*calls: ToolCallRequest, reasoning: str | None = None) -> AssistantTurn:
    return AssistantTurn(tool_calls=list(calls), reasoning_content=reasoning)


def _final_turn(content: str = '{"final": {"integration_goal": "done"}}') -> AssistantTurn:
    return AssistantTurn(content=content, finish_reason="stop")


def _run(
    client: FakeToolCallingClient,
    *,
    context: ToolContext,
    registry: ToolRegistry,
    config: AgentLoopConfig | None = None,
    system_prompt: str = "system prompt",
    user_prompt: str = "user prompt",
):
    return AgentLoopRunner(config).run(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context=context,
        registry=registry,
        llm=client,
    )


def _registry(*tools: _SpyTool) -> ToolRegistry:
    return ToolRegistry(list(tools))


def _tool_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [message for message in messages if message.role == "tool"]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _loop_source_tree() -> ast.Module:
    return ast.parse(Path(loop_module.__file__).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 1~4 基本回路


def test_01_a_final_answer_without_any_tool_call_completes(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient(_final_turn("hello"))

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert result.status == "completed"
    assert result.completed is True
    assert result.content == "hello"
    assert result.turns == 1
    assert result.call_count == 0
    assert result.tool_calls == [] and result.tool_results == []
    assert spy.invocations == []
    assert client.calls == 1


def test_02_one_tool_call_then_a_final_answer(tmp_path: Path) -> None:
    spy = _SpyTool("search_code")
    client = FakeToolCallingClient(
        [_tool_turn(_call("call-1", "search_code", '{"query": "httpx"}')), _final_turn()]
    )

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert result.status == "completed"
    assert result.turns == 2  # 两次 LLM 请求
    assert result.call_count == 1  # 一次 tool call
    assert spy.invocations == [{"query": "httpx"}]
    # 第二轮请求里必须已有 assistant 轮 + 对应的 tool 消息
    second = client.requests[1]
    assert [message.role for message in second] == ["system", "user", "assistant", "tool"]


def test_03_two_tool_rounds_drive_the_real_read_only_tools(tmp_path: Path) -> None:
    """Turn 1 = inspect_project，Turn 2 = search_code，Turn 3 = final。"""
    _project_tree(tmp_path)
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "inspect_project", "{}")),
            _tool_turn(_call("c2", "search_code", '{"query": "def fetch_pets"}')),
            _final_turn(),
        ]
    )

    result = _run(client, context=_context(tmp_path), registry=build_default_registry())

    assert result.status == "completed"
    assert client.calls == 3
    assert [item.tool for item in result.tool_results] == ["inspect_project", "search_code"]
    assert all(item.ok for item in result.tool_results)
    # 工具真的读到了这个临时项目，而不是返回了一句套话
    assert "sample-project" in result.tool_results[0].content
    assert "pkg/client.py" in result.tool_results[1].content


def test_04_multiple_tool_calls_in_one_turn(tmp_path: Path) -> None:
    first, second = _SpyTool("first"), _SpyTool("second")
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "first"), _call("c2", "second")),
            _final_turn(),
        ]
    )

    result = _run(client, context=_context(tmp_path), registry=_registry(first, second))

    assert result.status == "completed"
    assert result.turns == 2  # 1 次请求带 2 个 tool call = 1 turn + 2 tool calls
    assert result.call_count == 2
    assert [item.tool for item in result.tool_results] == ["first", "second"]
    assert len(_tool_messages(client.requests[1])) == 2


# ------------------------------------------------------------------ 5~8 失败的 tool call


def test_05_unknown_tool_is_reported_as_a_failed_result(tmp_path: Path) -> None:
    spy = _SpyTool("known")
    client = FakeToolCallingClient([_tool_turn(_call("c1", "ghost_tool", "{}")), _final_turn()])

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert result.status == "completed"  # 单次失败可恢复：错误回给模型让它改正
    assert result.tool_results[0].ok is False
    assert "未注册" in (result.tool_results[0].error or "")
    assert spy.invocations == []
    assert any("ghost_tool" in warning for warning in result.warnings)
    # 失败原因必须回填进对话，否则模型无从知道发生了什么
    assert "未注册" in _tool_messages(client.requests[1])[0].content


def test_06_invalid_json_arguments_are_rejected_without_executing(tmp_path: Path) -> None:
    spy = _SpyTool("read_file")
    # 恶意/坏 JSON：里面塞了一个凭据形状的哨兵，用来验证错误路径不回显原文
    raw = '{"path": "pkg/client.py", "token": "sk-leak-me-123", '
    client = FakeToolCallingClient([_tool_turn(_call("c1", "read_file", raw)), _final_turn()])

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert spy.invocations == []
    assert result.tool_results[0].ok is False
    assert "invalid tool arguments" in (result.tool_results[0].error or "")
    assert "sk-leak-me-123" not in (result.tool_results[0].error or "")
    assert "sk-leak-me-123" not in " ".join(result.warnings)
    # assistant 消息必须原样回填模型给的 arguments（这正是不能拿它当错误信息的原因）
    assert client.requests[1][2].tool_calls[0].arguments == raw


def test_07_arguments_that_are_not_an_object_are_rejected(tmp_path: Path) -> None:
    spy = _SpyTool("read_file")
    client = FakeToolCallingClient(
        [_tool_turn(_call("c1", "read_file", "[1, 2, 3]")), _final_turn()]
    )

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert spy.invocations == []
    assert result.tool_results[0].ok is False
    assert "顶层必须是 JSON 对象" in (result.tool_results[0].error or "")


def test_08_a_failing_tool_result_is_recorded_and_survives_the_turn(tmp_path: Path) -> None:
    broken = _SpyTool("read_file", error="文件不存在：pkg/nope.py")
    healthy = _SpyTool("search_code")
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "read_file"), _call("c2", "search_code")),
            _final_turn(),
        ]
    )

    result = _run(client, context=_context(tmp_path), registry=_registry(broken, healthy))

    assert result.status == "completed"
    # 第二个成功的结果没有被第一个失败吞掉
    assert [item.ok for item in result.tool_results] == [False, True]
    assert healthy.invocations == [{}]
    assert result.warnings == ["工具 read_file 执行失败：文件不存在：pkg/nope.py"]


# ------------------------------------------------------------------ 9~12 有界停止


def test_09_consecutive_tool_errors_stop_the_loop(tmp_path: Path) -> None:
    broken = _SpyTool("read_file", error="读取失败")
    # 两次调用的参数必须不同：完全相同的那次会被重复调用检测先拦下，
    # 测到的就不是"连续工具错误"这条路径了
    client = FakeToolCallingClient(
        [
            _tool_turn(
                _call("c1", "read_file", '{"path": "pkg/a.py"}'),
                _call("c2", "read_file", '{"path": "pkg/b.py"}'),
            )
        ]
    )

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(broken),
        config=AgentLoopConfig(max_consecutive_tool_errors=2),
    )

    assert result.status == "tool_error"
    assert result.call_count == 2
    assert client.calls == 1  # 停止后不再调用 LLM
    assert broken.invocations == [{"path": "pkg/a.py"}, {"path": "pkg/b.py"}]


def test_09b_a_success_resets_the_consecutive_error_counter(tmp_path: Path) -> None:
    broken = _SpyTool("broken", error="失败")
    healthy = _SpyTool("healthy")
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "broken", '{"n": 1}')),
            _tool_turn(_call("c2", "healthy")),
            _tool_turn(_call("c3", "broken", '{"n": 2}')),
            _final_turn(),
        ]
    )

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(broken, healthy),
        config=AgentLoopConfig(max_consecutive_tool_errors=2),
    )

    # 失败 → 成功（清零）→ 失败，从未连续两次
    assert result.status == "completed"
    assert [item.ok for item in result.tool_results] == [False, True, False]


def test_10_a_repeated_call_stops_the_loop(tmp_path: Path) -> None:
    spy = _SpyTool("search_code")
    # 两次调用同名同参数、id 不同：指纹只看工具名 + 规范化参数
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "search_code", '{"query": "httpx", "max_results": 5}')),
            _tool_turn(_call("c2", "search_code", '{"max_results": 5, "query": "httpx"}')),
        ]
    )

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert result.status == "repeated_call"
    assert result.tool_results[-1].ok is False
    assert "重复调用" in (result.tool_results[-1].error or "")
    assert any("重复调用" in warning for warning in result.warnings)


def test_11_max_tool_calls_bounds_the_loop(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient([_tool_turn(_call("c1", "spy"), _call("c2", "spy", '{"n": 2}'))])

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(spy),
        config=AgentLoopConfig(max_tool_calls=1),
    )

    assert result.status == "max_tool_calls"
    assert result.call_count == 1
    assert len(spy.invocations) == 1  # 超限的那次没有被执行
    assert client.calls == 1


def test_12_max_turns_bounds_the_loop(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    # 两轮用不同参数：否则会被重复调用检测先拦下，测不到"轮次上限"这条路径
    client = FakeToolCallingClient(
        [_tool_turn(_call("c1", "spy", '{"n": 1}')), _tool_turn(_call("c2", "spy", '{"n": 2}'))]
    )

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(spy),
        config=AgentLoopConfig(max_turns=2),
    )

    assert result.status == "max_turns"
    assert result.turns == 2
    assert client.calls == 2  # 恰好用完轮次，绝不越界多调一次


# ------------------------------------------------------------------ 13~14 预算


def test_13_observation_budget_stops_the_loop_without_pretending(tmp_path: Path) -> None:
    big = "x" * 500
    spy = _SpyTool("search_code", content=big)
    client = FakeToolCallingClient([_tool_turn(_call("c1", "search_code"))])

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(spy),
        config=AgentLoopConfig(max_observation_chars=100),
    )

    assert result.status == "observation_limit"
    assert result.observation_chars == 500
    # 关键：结果既没被截断，也没被标成"完整过的"
    assert result.tool_results[0].content == big
    assert result.tool_results[0].truncated is False
    assert any("超过上限" in warning for warning in result.warnings)


def test_14_history_budget_stops_before_the_first_llm_call(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient(_final_turn())

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(spy),
        config=AgentLoopConfig(max_history_chars=1),
    )

    assert result.status == "observation_limit"
    assert client.calls == 0  # 宁可一开始就停，也不静默丢掉历史
    assert result.tool_calls == []
    assert any("消息历史" in warning for warning in result.warnings)


# ------------------------------------------------------------------ 15~16 非法输入


def test_15_llm_exception_becomes_llm_error_without_leaking(tmp_path: Path, caplog) -> None:
    secret = "sk-super-secret-123"

    class _BoomClient:
        def chat(self, messages, *, tools=None, tool_choice=None):
            raise RuntimeError(f"401 Unauthorized: Bearer {secret}")

    with caplog.at_level(logging.ERROR, logger="integration_agent.agent.agent_loop"):
        result = AgentLoopRunner().run(
            system_prompt="s",
            user_prompt="u",
            context=_context(tmp_path),
            registry=_registry(_SpyTool("spy")),
            llm=_BoomClient(),
        )

    assert result.status == "llm_error"
    assert result.error == LLM_CALL_FAILED
    assert secret not in result.model_dump_json()
    assert "Traceback" not in result.model_dump_json()
    # 完整异常只进服务端日志
    assert any(record.exc_info for record in caplog.records)


def test_16_malformed_assistant_turn_is_invalid_response(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient(AssistantTurn())  # 既无正文也无 tool_calls

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert result.status == "invalid_response"
    assert result.final_message is None
    assert spy.invocations == []


def test_16b_a_tool_call_without_an_id_is_never_rebuilt(tmp_path: Path) -> None:
    """缺 id 的调用整轮判废——绝不补齐、绝不猜测。"""

    class _MalformedTurn:
        content = "text"
        reasoning_content = None
        tool_calls = [SimpleNamespace(id="   ", name="spy", arguments="{}")]

        def to_message(self):  # pragma: no cover - 走到这里就说明实现伪造了消息
            raise AssertionError("不该用不合法的 tool call 去构造 assistant 消息")

    class _Client:
        def chat(self, messages, *, tools=None, tool_choice=None):
            return _MalformedTurn()

    result = AgentLoopRunner().run(
        system_prompt="s",
        user_prompt="u",
        context=_context(tmp_path),
        registry=_registry(_SpyTool("spy")),
        llm=_Client(),
    )

    assert result.status == "invalid_response"
    assert result.tool_calls == []


def test_16c_a_turn_that_cannot_be_replayed_is_invalid_response(tmp_path: Path) -> None:
    """to_message() 回填失败时判废，而不是拿半成品去凑一条 assistant 消息。"""

    class _UnreplayableTurn:
        content = "text"
        reasoning_content = None
        tool_calls = [SimpleNamespace(id="c1", name="spy", arguments="{}")]

        def to_message(self):
            # provider 回了一轮无法回填的消息：pydantic 在构造 assistant 消息时拒绝
            return ChatMessage.model_validate({"role": "assistant", "tool_calls": [{"id": ""}]})

    class _Client:
        def chat(self, messages, *, tools=None, tool_choice=None):
            return _UnreplayableTurn()

    result = AgentLoopRunner().run(
        system_prompt="s",
        user_prompt="u",
        context=_context(tmp_path),
        registry=_registry(_SpyTool("spy")),
        llm=_Client(),
    )

    assert result.status == "invalid_response"


# ------------------------------------------------------------------ 17~19 逐字节保真


def test_17_tool_call_id_is_preserved_byte_for_byte(tmp_path: Path) -> None:
    spy = _SpyTool("search_code")
    weird_id = "call_XyZ-001.α"
    client = FakeToolCallingClient([_tool_turn(_call(weird_id, "search_code")), _final_turn()])

    _run(client, context=_context(tmp_path), registry=_registry(spy))

    sent = client.requests[1]
    assert sent[2].tool_calls[0].id == weird_id
    tool_message = _tool_messages(sent)[0]
    assert tool_message.tool_call_id == weird_id
    # tool message 的 id 必须等于模型给的那个，不能由程序另生成
    assert tool_message.tool_call_id == sent[2].tool_calls[0].id


def test_18_reasoning_content_is_preserved_across_the_loop(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "spy"), reasoning="reasoning-123"),
            _final_turn(),
        ]
    )

    _run(client, context=_context(tmp_path), registry=_registry(spy))

    assistant_message = client.requests[1][2]
    assert assistant_message.role == "assistant"
    assert assistant_message.reasoning_content == "reasoning-123"
    # 第一轮请求里不该凭空出现思维链（它来自模型的响应，不是我们编的）
    assert all(message.reasoning_content is None for message in client.requests[0])


def test_19_assistant_tool_calls_are_replayed_verbatim(tmp_path: Path) -> None:
    spy = _SpyTool("search_code")
    raw = '{  "query" :"httpx",\n  "max_results":   3 }'
    client = FakeToolCallingClient([_tool_turn(_call("c1", "search_code", raw)), _final_turn()])

    _run(client, context=_context(tmp_path), registry=_registry(spy))

    replayed = client.requests[1][2].tool_calls[0]
    assert replayed.arguments == raw  # 逐字节一致，不重新序列化
    assert replayed.id == "c1"
    assert replayed.name == "search_code"


# ------------------------------------------------------------------ 20~22 执行纪律


def test_20_an_unknown_tool_is_never_executed(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient(
        [_tool_turn(_call("c1", "ghost")), _tool_turn(_call("c2", "spy")), _final_turn()]
    )

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert result.status == "completed"
    assert spy.invocations == [{}]  # 只有第二次（合法的那次）被执行
    assert result.tool_results[0].tool == "ghost"  # 未注册的名字只出现在结果里


def test_21_a_repeated_call_is_not_executed_a_second_time(tmp_path: Path) -> None:
    spy = _SpyTool("search_code")
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "search_code", '{"query": "httpx"}')),
            _tool_turn(_call("c2", "search_code", '{"query": "httpx"}')),
        ]
    )

    result = _run(client, context=_context(tmp_path), registry=_registry(spy))

    assert result.status == "repeated_call"
    assert len(spy.invocations) == 1  # 第二次没有真的跑
    assert result.call_count == 2  # 但它确实被"处理"过，并留下了失败结果


def test_22_tool_calls_run_in_the_order_the_model_gave(tmp_path: Path) -> None:
    order: list[str] = []
    tools = [
        _SpyTool("alpha", order_log=order),
        _SpyTool("beta", order_log=order),
        _SpyTool("gamma", order_log=order),
    ]
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "gamma"), _call("c2", "alpha"), _call("c3", "beta")),
            _final_turn(),
        ]
    )

    result = _run(client, context=_context(tmp_path), registry=ToolRegistry(tools))

    # 不排序、不去重、不并行——模型给的顺序就是执行顺序
    assert order == ["gamma", "alpha", "beta"]
    assert [item.tool for item in result.tool_results] == ["gamma", "alpha", "beta"]


# ------------------------------------------------------------------ 23~25 安全边界


def test_23_agent_loop_does_not_write_files(tmp_path: Path) -> None:
    _project_tree(tmp_path)
    before = _snapshot(tmp_path)
    client = FakeToolCallingClient(
        [
            _tool_turn(_call("c1", "inspect_project")),
            _tool_turn(_call("c2", "search_code", '{"query": "def fetch_pets"}')),
            _tool_turn(_call("c3", "read_file", '{"path": "pkg/client.py"}')),
            _final_turn(),
        ]
    )

    result = _run(client, context=_context(tmp_path), registry=build_default_registry())

    assert result.status == "completed"
    assert _snapshot(tmp_path) == before


def test_23b_agent_loop_module_has_no_write_capability() -> None:
    tree = _loop_source_tree()
    calls = [
        node.func
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert all(call.id != "open" for call in calls)

    forbidden_attributes = {
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
        "rename",
        "replace",
        "rmdir",
        "remove",
        "system",
        "popen",
        "run",
        "call",
        "check_output",
        "eval",
        "exec",
    }
    used_attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert used_attributes.isdisjoint(forbidden_attributes)


def test_24_agent_loop_does_not_call_shell() -> None:
    imports = _imported_modules(Path(loop_module.__file__))
    assert imports & set(FORBIDDEN_SHELL_MODULES) == set()
    assert imports <= ALLOWED_LOOP_IMPORTS


def test_25_agent_loop_does_not_use_network_tools() -> None:
    imports = _imported_modules(Path(loop_module.__file__))
    assert imports & set(FORBIDDEN_NETWORK_MODULES) == set()

    source = Path(loop_module.__file__).read_text(encoding="utf-8")
    assert "socket" not in _code_identifiers(source)


def _code_identifiers(source: str) -> set[str]:
    """源码里的**标识符**（不含字符串与 docstring）。"""
    identifiers: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    return identifiers


# ------------------------------------------------------------------ 结构性护栏


def test_the_loop_has_no_unbounded_while_loop() -> None:
    tree = _loop_source_tree()
    while_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.While)]
    assert [node for node in while_nodes if _is_truthy_constant(node.test)] == []


def _is_truthy_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value)


def test_agent_loop_runner_satisfies_the_protocol() -> None:
    assert isinstance(AgentLoopRunner(), AgentLoop)


def test_configuration_bounds_must_be_positive() -> None:
    from pydantic import ValidationError

    for field in (
        "max_tool_calls",
        "max_turns",
        "max_observation_chars",
        "max_consecutive_tool_errors",
        "max_history_chars",
    ):
        with pytest.raises(ValidationError):
            AgentLoopConfig(**{field: 0})
        with pytest.raises(ValidationError):
            AgentLoopConfig(**{field: -1})


def test_default_configuration_matches_the_specified_budgets() -> None:
    config = AgentLoopConfig()
    # 刻意写死：从实现里读默认值，等于把"上限被悄悄放宽"也一起放过
    assert config.max_tool_calls == 8
    assert config.max_turns == 6
    assert config.max_observation_chars == 40_000
    assert config.max_consecutive_tool_errors == 2
    assert config.max_history_chars == 60_000


def test_blank_prompts_are_rejected_before_any_llm_call(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient(_final_turn())

    result = _run(client, context=_context(tmp_path), registry=_registry(spy), system_prompt="   ")

    assert result.status == "error"
    assert client.calls == 0


def test_tool_resources_are_not_in_the_context(tmp_path: Path) -> None:
    """ToolContext 只带只读事实：不带凭据、不带 LLM、不带注册表。"""
    context = _context(tmp_path)
    dumped = context.model_dump_json()
    assert set(context.model_dump()) == {"project_root", "project", "api", "known_files"}
    for needle in CREDENTIAL_NEEDLES:
        assert needle not in dumped.lower()


def test_tool_schemas_sent_to_the_model_are_stable_and_credential_free(tmp_path: Path) -> None:
    spy = _SpyTool("spy")
    client = FakeToolCallingClient(_final_turn())

    _run(client, context=_context(tmp_path), registry=_registry(spy))

    sent_tools = client.tools[0]
    assert [item["function"]["name"] for item in sent_tools] == ["spy"]
    rendered = json.dumps(sent_tools, ensure_ascii=False, sort_keys=True)
    for needle in CREDENTIAL_NEEDLES:
        assert needle not in rendered.lower()
    assert str(tmp_path) not in rendered  # 绝不把本机绝对路径写进工具定义


def test_an_unexpected_tool_crash_becomes_a_failed_result(tmp_path: Path, caplog) -> None:
    class _BoomTool:
        spec = ToolSpec(name="boom", description="会炸的工具")

        def invoke(self, args: dict, ctx: ToolContext):
            raise RuntimeError("内部错误：/home/secret/path.py")

    client = FakeToolCallingClient([_tool_turn(_call("c1", "boom")), _final_turn()])

    with caplog.at_level(logging.ERROR, logger="integration_agent.agent.agent_loop"):
        result = _run(client, context=_context(tmp_path), registry=ToolRegistry([_BoomTool()]))

    assert result.status == "completed"
    assert result.tool_results[0].ok is False
    assert "/home/secret/path.py" not in result.model_dump_json()
    assert any(record.exc_info for record in caplog.records)


def test_a_tool_returning_a_non_result_is_rejected(tmp_path: Path) -> None:
    class _BadTool:
        spec = ToolSpec(name="bad", description="返回类型不对的工具")

        def invoke(self, args: dict, ctx: ToolContext):
            return "not a ToolResult"

    client = FakeToolCallingClient([_tool_turn(_call("c1", "bad")), _final_turn()])

    result = _run(client, context=_context(tmp_path), registry=ToolRegistry([_BadTool()]))

    assert result.tool_results[0].ok is False
    assert "不符合契约" in (result.tool_results[0].error or "")


def test_every_processed_tool_call_has_exactly_one_result(tmp_path: Path) -> None:
    first, second = _SpyTool("first"), _SpyTool("second", error="失败")
    client = FakeToolCallingClient(
        [_tool_turn(_call("c1", "first"), _call("c2", "second"), _call("c3", "ghost"))]
    )

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(first, second),
        config=AgentLoopConfig(max_consecutive_tool_errors=2),
    )

    assert result.status == "tool_error"
    assert len(result.tool_calls) == len(result.tool_results) == result.call_count == 3
    assert [item.tool for item in result.tool_results] == ["first", "second", "ghost"]
    assert [item.call_id for item in result.tool_results] == ["c1", "c2", "c3"]


def test_a_missing_final_answer_is_never_invented(tmp_path: Path) -> None:
    """回路跑到上限时，final_message 保持为 None —— 绝不编一个"最终回答"。"""
    spy = _SpyTool("spy")
    client = FakeToolCallingClient([_tool_turn(_call("c1", "spy"))])

    result = _run(
        client,
        context=_context(tmp_path),
        registry=_registry(spy),
        config=AgentLoopConfig(max_turns=1),
    )

    assert result.status == "max_turns"
    assert result.final_message is None
    assert result.content == ""
