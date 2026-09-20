"""Agent Tool 基础契约单元测试（E2 Commit 1）。

覆盖要求场景：
    1. ToolSpec 正常构造 / JSON round-trip
    2. ToolSpec 空 name / 空 description 拒绝
    3. ToolSpec parameters 必须是 JSON-schema 风格的 object
    4. ToolCall 正常构造 / 默认 args / JSON round-trip
    5. ToolCall 非 dict args 拒绝
    6. ToolResult 成功 / 失败 / truncated / error
    7. ToolResult chars 计量与边界
    8. ToolResult 不接受 traceback
    9. ToolResult JSON round-trip
    10. ToolContext 正常构造
    11. ToolContext 结构上不含凭据 / 执行能力
    12. AgentTool Protocol 与 runtime_checkable
    13. ToolRegistry 注册一个 / 多个
    14. ToolRegistry get 已知 / 未知
    15. ToolRegistry specs 稳定顺序
    16. ToolRegistry describe 稳定输出
    17. ToolRegistry 重复 name 拒绝
    18. ToolRegistry 空 name 拒绝
    19. ToolRegistry 拒绝非法工具对象
    20. ToolRegistry 不执行、不吞异常

全部使用 FakeTool 与内存构造的 ProjectStructure / APIInfo：
不调用 DeepSeek、不访问真实项目文件、不发网络请求。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from integration_agent import tools
from integration_agent.api import APIInfo
from integration_agent.repository import ProjectStructure
from integration_agent.tools import (
    AgentTool,
    ToolContext,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
    ToolSpec,
)

FAKE_ROOT = "/nonexistent/fake_repo"  # 只作为字符串载体，不会触碰磁盘


# ------------------------------------------------------------------ helpers


def _project() -> ProjectStructure:
    return ProjectStructure(
        root=FAKE_ROOT,
        name="fake-project",
        is_python_project=True,
        manifest_files=["pyproject.toml"],
        source_dirs=["fake_pkg"],
        test_dirs=["tests"],
        dependencies=["httpx>=0.27"],
        python_files=["fake_pkg/__init__.py", "fake_pkg/client.py"],
    )


def _api() -> APIInfo:
    return APIInfo(name="Fake API", version="1.0.0", base_url="https://api.example.com")


def _ctx() -> ToolContext:
    return ToolContext(
        project_root=Path(FAKE_ROOT),
        project=_project(),
        api=_api(),
        known_files={"fake_pkg/__init__.py", "fake_pkg/client.py"},
    )


def _spec(name: str = "fake_tool", **overrides) -> ToolSpec:
    payload = {"name": name, "description": "假工具，不做任何真实操作"}
    payload.update(overrides)
    return ToolSpec(**payload)


class FakeTool:
    """满足 AgentTool Protocol 的假工具：不读文件、不联网、不执行 shell。"""

    def __init__(self, name: str = "fake_tool", *, result: ToolResult | None = None) -> None:
        self.spec = _spec(name)
        self._result = result
        self.calls: list[dict] = []

    def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        if self._result is not None:
            return self._result
        return ToolResult(call_id="", tool=self.spec.name, ok=True, content=f"ok:{self.spec.name}")


# ------------------------------------------------------- 场景 1：ToolSpec


def test_tool_spec_constructs() -> None:
    spec = ToolSpec(
        name="search_code",
        description="按关键词搜索仓库中的代码片段",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert spec.name == "search_code"
    assert spec.parameters["type"] == "object"
    assert spec.read_only is True  # v1 默认只读


def test_tool_spec_defaults_are_read_only_and_parameterless() -> None:
    spec = ToolSpec(name="noop", description="无参数工具")
    assert spec.parameters == {}
    assert spec.read_only is True


def test_tool_spec_read_only_can_be_marked_false() -> None:
    """字段存在，留给未来的非只读工具；v1 不注册任何这样的工具。"""
    spec = ToolSpec(name="writer", description="写文件", read_only=False)
    assert spec.read_only is False


def test_tool_spec_json_roundtrip() -> None:
    spec = ToolSpec(
        name="inspect_api",
        description="查询 API 端点",
        parameters={"type": "object", "properties": {"endpoint": {"type": "string"}}},
    )
    assert ToolSpec.model_validate_json(spec.model_dump_json()) == spec


# --------------------------------------------- 场景 2：ToolSpec 非空校验


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_tool_spec_rejects_blank_name(name: str) -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name=name, description="说明")


@pytest.mark.parametrize("description", ["", "   ", "\n"])
def test_tool_spec_rejects_blank_description(description: str) -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="tool", description=description)


def test_tool_spec_strips_surrounding_whitespace() -> None:
    """工具名会成为注册表的键，首尾空白会让"查不到工具"变得难以排查。"""
    spec = ToolSpec(name="  search_code  ", description="  搜索  ")
    assert spec.name == "search_code"
    assert spec.description == "搜索"


# ------------------------------------ 场景 3：ToolSpec parameters 校验


def test_tool_spec_rejects_non_object_schema_type() -> None:
    with pytest.raises(ValidationError, match="object"):
        ToolSpec(name="tool", description="说明", parameters={"type": "array"})


def test_tool_spec_accepts_schema_without_type() -> None:
    """JSON Schema 允许省略 type，此时不强制补齐。"""
    spec = ToolSpec(name="tool", description="说明", parameters={"properties": {}})
    assert "type" not in spec.parameters


def test_tool_spec_rejects_non_serializable_parameters() -> None:
    """parameters 会被 json.dumps 渲染进 prompt，非序列化值必须在这里就挡住。"""
    with pytest.raises(ValidationError, match="JSON"):
        ToolSpec(name="tool", description="说明", parameters={"type": "object", "v": {1, 2}})


# ------------------------------------------------------- 场景 4：ToolCall


def test_tool_call_constructs() -> None:
    call = tools.ToolCall(tool="search_code", args={"query": "httpx"}, call_id="c1")
    assert call.tool == "search_code"
    assert call.args == {"query": "httpx"}
    assert call.call_id == "c1"


def test_tool_call_args_defaults_to_empty_dict() -> None:
    call = tools.ToolCall(tool="inspect_project")
    assert call.args == {}
    assert call.call_id == ""


def test_tool_call_json_roundtrip() -> None:
    call = tools.ToolCall(tool="read_file", args={"path": "fake_pkg/client.py"}, call_id="c2")
    assert tools.ToolCall.model_validate_json(call.model_dump_json()) == call


@pytest.mark.parametrize("args", [["query"], "query", 1, None, [{"a": 1}]])
def test_tool_call_rejects_non_dict_args(args) -> None:
    """args 来自 LLM，类型必须在这里就收敛，工具实现才能安心按 dict 取值。"""
    with pytest.raises(ValidationError):
        tools.ToolCall(tool="search_code", args=args)


@pytest.mark.parametrize("tool", ["", "   "])
def test_tool_call_rejects_blank_tool_name(tool: str) -> None:
    with pytest.raises(ValidationError):
        tools.ToolCall(tool=tool)


# ----------------------------------------------------- 场景 6-9：ToolResult


def test_tool_result_success() -> None:
    result = ToolResult(call_id="c1", tool="search_code", ok=True, content="3 matches")
    assert result.ok is True
    assert result.error is None
    assert result.truncated is False
    assert result.content == "3 matches"


def test_tool_result_failure() -> None:
    result = ToolResult(
        call_id="c1",
        tool="read_file",
        ok=False,
        content="",
        error="path 包含越界片段 '..'",
    )
    assert result.ok is False
    assert result.error == "path 包含越界片段 '..'"
    assert result.content == ""  # 失败时 content 仍是字符串，不是 None


def test_tool_result_truncated_flag() -> None:
    result = ToolResult(call_id="c1", tool="read_file", ok=True, content="x" * 10, truncated=True)
    assert result.truncated is True


def test_tool_result_error_defaults_to_none() -> None:
    result = ToolResult(call_id="c1", tool="t", ok=True, content="ok")
    assert result.error is None


def test_tool_result_json_roundtrip() -> None:
    result = ToolResult(
        call_id="c1",
        tool="inspect_api",
        ok=True,
        content='{"endpoints": []}',
        truncated=True,
        error=None,
    )
    restored = ToolResult.model_validate_json(result.model_dump_json())
    assert restored == result


# ------------------------------------------------ 场景 7：ToolResult chars


def test_tool_result_chars_follows_content_length() -> None:
    """未显式给 chars 时取实际长度：静默的 0 会让观察预算失去刹车作用。"""
    result = ToolResult(call_id="c1", tool="t", ok=True, content="hello")
    assert result.chars == 5


def test_tool_result_chars_is_zero_for_empty_content() -> None:
    result = ToolResult(call_id="c1", tool="t", ok=False, content="", error="boom")
    assert result.chars == 0


def test_tool_result_explicit_chars_is_not_overwritten() -> None:
    result = ToolResult(call_id="c1", tool="t", ok=True, content="hello", chars=99)
    assert result.chars == 99


@pytest.mark.parametrize("chars", [-1, -100])
def test_tool_result_rejects_negative_chars(chars: int) -> None:
    with pytest.raises(ValidationError):
        ToolResult(call_id="c1", tool="t", ok=True, content="hello", chars=chars)


# -------------------------------------------- 场景 8：ToolResult 无 traceback


def test_tool_result_rejects_traceback_in_error() -> None:
    """traceback 里带着本地绝对路径与源码行，绝不能进 Agent 上下文。

    工具必须自己把异常翻译成一句话，完整堆栈只写服务端日志——
    与 Pipeline / RepairLoop 的既有约定一致。
    """
    leaky = (
        "Traceback (most recent call last):\n"
        '  File "D:\\repo\\src\\tool.py", line 12, in invoke\n'
        "ValueError: boom"
    )
    with pytest.raises(ValidationError, match="traceback"):
        ToolResult(call_id="c1", tool="t", ok=False, content="", error=leaky)


def test_tool_result_accepts_plain_error_message() -> None:
    result = ToolResult(call_id="c1", tool="t", ok=False, content="", error="query 不能为空")
    assert result.error == "query 不能为空"


def test_tool_result_does_not_auto_capture_traceback() -> None:
    """模型没有任何字段能自动装载堆栈：error 只能由调用方显式给出。"""
    assert set(ToolResult.model_fields) == {
        "call_id",
        "tool",
        "ok",
        "content",
        "truncated",
        "error",
        "chars",
    }


# ----------------------------------------------------- 场景 10：ToolContext


def test_tool_context_constructs() -> None:
    ctx = _ctx()
    assert ctx.project_root == Path(FAKE_ROOT)
    assert ctx.project.name == "fake-project"
    assert ctx.api.name == "Fake API"
    assert ctx.known_files == {"fake_pkg/__init__.py", "fake_pkg/client.py"}


def test_tool_context_json_roundtrip() -> None:
    ctx = _ctx()
    assert ToolContext.model_validate_json(ctx.model_dump_json()) == ctx


# ------------------------------- 场景 11：ToolContext 无凭据 / 无执行能力


def test_tool_context_has_exactly_the_readonly_fields() -> None:
    """ToolContext 是"只读环境"的结构性保证：字段集合本身就是安全边界。

    这不是"暂时没用上"，而是"不许有"——一旦有人往这里加 executor / api_key，
    这条用例会立刻失败，逼他重新想清楚为什么工具需要那种能力。
    """
    assert set(ToolContext.model_fields) == {
        "project_root",
        "project",
        "api",
        "known_files",
    }


def test_tool_context_field_names_carry_no_credentials_or_abilities() -> None:
    forbidden = ("key", "token", "secret", "password", "credential", "env", "shell", "exec")
    for name in ToolContext.model_fields:
        assert not any(word in name.lower() for word in forbidden), f"可疑字段：{name}"


def test_tool_context_serialization_contains_no_secret() -> None:
    payload = _ctx().model_dump_json()
    assert "sk-" not in payload
    assert "DEEPSEEK_API_KEY" not in payload


# --------------------------------------- 场景 12：AgentTool Protocol


def test_fake_tool_satisfies_protocol() -> None:
    assert isinstance(FakeTool(), AgentTool)


def test_object_without_invoke_does_not_satisfy_protocol() -> None:
    class NotATool:
        spec = _spec("nope")

    assert not isinstance(NotATool(), AgentTool)


def test_object_without_spec_does_not_satisfy_protocol() -> None:
    class NotATool:
        def invoke(self, args, ctx):
            return None

    assert not isinstance(NotATool(), AgentTool)


def test_tools_layer_does_not_import_upward_modules() -> None:
    """工具层不得反向依赖 agent / repair / pipeline / api_server。

    用 AST 解析源码，而不是查 sys.modules——跑全量测试时 repair 早就被别的
    用例导进来了，那种检查会因执行顺序而假通过。这里只看 tools 自己的 import。

    这条用例把架构约束从注释钉成了断言：一旦有人在工具里 import DeepSeekPlanner
    或 repair，它会立刻失败。
    """
    import ast

    tools_dir = Path(tools.__file__).resolve().parent
    forbidden = (
        "integration_agent.agent",
        "integration_agent.repair",
        "integration_agent.pipeline",
        "integration_agent.api_server",
        "integration_agent.generation",
        "integration_agent.validation",
    )

    for source_file in sorted(tools_dir.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            else:
                continue
            for name in imported:
                assert not name.startswith(forbidden), f"{source_file.name} 反向依赖了 {name}"


# ------------------------------------------------ 场景 13-14：注册与查询


def test_registry_registers_single_tool() -> None:
    tool = FakeTool("search_code")
    registry = ToolRegistry([tool])
    assert registry.get("search_code") is tool


def test_registry_registers_multiple_tools() -> None:
    first, second = FakeTool("search_code"), FakeTool("read_file")
    registry = ToolRegistry([first, second])
    assert registry.get("search_code") is first
    assert registry.get("read_file") is second


def test_registry_get_returns_none_for_unknown_tool() -> None:
    registry = ToolRegistry([FakeTool("search_code")])
    assert registry.get("no_such_tool") is None


def test_registry_accepts_empty_tool_list() -> None:
    registry = ToolRegistry([])
    assert registry.specs() == []
    assert registry.describe() == ""
    assert registry.get("anything") is None


def test_registry_is_not_executing_tools_on_construction() -> None:
    """注册不是执行：构造 registry 不应触发任何 invoke。"""
    tool = FakeTool("search_code")
    ToolRegistry([tool])
    assert tool.calls == []


# ------------------------------------------------ 场景 15：specs 稳定顺序


def test_registry_specs_are_sorted_by_name() -> None:
    registry = ToolRegistry(
        [FakeTool("read_file"), FakeTool("inspect_api"), FakeTool("search_code")]
    )
    assert [spec.name for spec in registry.specs()] == ["inspect_api", "read_file", "search_code"]


def test_registry_specs_order_is_independent_of_registration_order() -> None:
    """调用方可能传进来一个 set；顺序抖动会让"同样输入同样 prompt"不成立。"""
    names = ["search_code", "inspect_api", "read_file"]
    forward = ToolRegistry([FakeTool(name) for name in names])
    backward = ToolRegistry([FakeTool(name) for name in reversed(names)])
    assert forward.specs() == backward.specs()


def test_registry_specs_are_repeatable() -> None:
    registry = ToolRegistry([FakeTool("b"), FakeTool("a")])
    assert registry.specs() == registry.specs()


# ---------------------------------------------- 场景 16：describe 稳定输出


def test_registry_describe_renders_name_description_and_parameters() -> None:
    tool = FakeTool("search_code")
    tool.spec = ToolSpec(
        name="search_code",
        description="按关键词搜索代码",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    text = ToolRegistry([tool]).describe()

    assert "search_code" in text
    assert "按关键词搜索代码" in text
    assert '"query"' in text


def test_registry_describe_is_deterministic_across_registration_order() -> None:
    names = ["search_code", "inspect_api"]
    forward = ToolRegistry([FakeTool(name) for name in names]).describe()
    backward = ToolRegistry([FakeTool(name) for name in reversed(names)]).describe()
    assert forward == backward


def test_registry_describe_is_byte_stable_regardless_of_key_order() -> None:
    """sort_keys 保证同一份 schema 的不同 dict 插入顺序渲染结果完全一致。"""

    def registry_with(parameters: dict) -> ToolRegistry:
        tool = FakeTool("t")
        tool.spec = ToolSpec(name="t", description="说明", parameters=parameters)
        return ToolRegistry([tool])

    left = registry_with({"type": "object", "properties": {"a": {"type": "string"}}})
    right = registry_with({"properties": {"a": {"type": "string"}}, "type": "object"})
    assert left.describe() == right.describe()


# -------------------------------------------- 场景 17-19：构造期拒绝


def test_registry_rejects_duplicate_tool_name() -> None:
    with pytest.raises(ToolRegistryError, match="重复"):
        ToolRegistry([FakeTool("search_code"), FakeTool("search_code")])


def test_registry_rejects_blank_tool_name() -> None:
    """用 model_construct 绕过 ToolSpec 自身的校验，模拟"没走正规构造"的工具。

    注册表这道防线存在，正是为了拦住不经过 ToolSpec 校验的对象；
    否则一个空名字会变成字典里的空键，"查不到工具"会非常难排查。
    """
    tool = FakeTool("placeholder")
    tool.spec = ToolSpec.model_construct(name="   ", description="说明", parameters={})
    with pytest.raises(ToolRegistryError, match="不能为空"):
        ToolRegistry([tool])


def test_registry_rejects_tool_without_tool_spec() -> None:
    class BadTool:
        spec = {"name": "not_a_model"}  # 不是 ToolSpec

        def invoke(self, args, ctx):
            return None

    with pytest.raises(ToolRegistryError, match="ToolSpec"):
        ToolRegistry([BadTool()])


def test_registry_rejects_tool_without_invoke() -> None:
    class BadTool:
        spec = _spec("no_invoke")

    with pytest.raises(ToolRegistryError, match="invoke"):
        ToolRegistry([BadTool()])


def test_registry_error_is_a_value_error() -> None:
    """沿用项目既有错误风格，不额外引入异常层次。"""
    assert issubclass(ToolRegistryError, ValueError)


# ------------------------------------------ 场景 20：不执行、不吞异常


def test_registry_does_not_swallow_tool_exceptions() -> None:
    """注册表不夹在调用链中间做 try/except。

    工具内部异常必须原样冒给调用方，由运行时决定如何处理——
    在注册表层吞掉只会让真实故障消失得无影无踪。
    """

    class ExplodingTool:
        spec = _spec("boom")

        def invoke(self, args: dict, ctx: ToolContext) -> ToolResult:
            raise RuntimeError("tool exploded")

    registry = ToolRegistry([ExplodingTool()])
    tool = registry.get("boom")
    assert tool is not None
    with pytest.raises(RuntimeError, match="tool exploded"):
        tool.invoke({}, _ctx())


def test_registry_returns_the_registered_object_unchanged() -> None:
    tool = FakeTool("search_code")
    assert ToolRegistry([tool]).get("search_code") is tool
