"""inspect_api 的行为、输出预算与安全边界。

场景分组：
    1. 基础   —— summary / endpoints / method / path 的语义
    2. 输出   —— 长度上限、truncated 与 chars、稳定性、不泄漏 schema
    3. 安全   —— 不联网、不写盘、不改 ctx.api、错误不泄漏 traceback 与绝对路径
    4. 注册   —— ToolSpec 合法性与默认注册表

测试对象只有 ctx.api 这一个数据来源：inspect_api 不解析 OpenAPI、不读文件、
不联网，所以这里的 APIInfo 直接构造——它是要被测代码的输入契约，不是要被
重新实现的解析逻辑。
"""

import ast
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from integration_agent.api import (
    APIEndpoint,
    APIInfo,
    APIParameter,
    APIRequestBody,
    APIResponse,
)
from integration_agent.repository import ProjectStructure
from integration_agent.tools import (
    AgentTool,
    InspectAPITool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    api_tools,
    build_default_registry,
)
from integration_agent.tools.api_tools import (
    MAX_ENDPOINT_LINES,
    MAX_ENDPOINTS_CHARS,
    MAX_FIELD_CHARS,
    MAX_SUMMARY_CHARS,
)

TOOL_NAME = "inspect_api"
DEFAULT_TOOL_NAMES = ("inspect_api", "inspect_project", "read_file", "search_code")

# 一个一眼能认出来的假项目根。inspect_api 不该碰它，更不该把它回给模型。
FAKE_ROOT = Path("/server/private/root")

SCHEMA_SENTINEL = "SCHEMA_SENTINEL_MUST_NOT_LEAK"
SECRET_SENTINEL = "SENTINEL_CREDENTIAL_MUST_NOT_LEAK"

# 上限的字面期望值。其余用例引用模块常量更好读，这一组刻意写死：
# 只从实现里读上限的话，把 MAX_ENDPOINTS_CHARS 放宽到一百万这类改动会让所有
# 断言静默跟着放宽，测试就失去了看门的作用。改这些数字必须是有意识的决定。
EXPECTED_LIMITS = {
    "MAX_SUMMARY_CHARS": 1200,
    "MAX_ENDPOINT_LINES": 40,
    "MAX_ENDPOINTS_CHARS": 4000,
    "MAX_FIELD_CHARS": 120,
}
ACTUAL_LIMITS = {
    "MAX_SUMMARY_CHARS": MAX_SUMMARY_CHARS,
    "MAX_ENDPOINT_LINES": MAX_ENDPOINT_LINES,
    "MAX_ENDPOINTS_CHARS": MAX_ENDPOINTS_CHARS,
    "MAX_FIELD_CHARS": MAX_FIELD_CHARS,
}

# api_tools.py 允许 import 的全部模块。做成白名单而不是黑名单：新增一个 import
# 必须是有意识的决定，而不是"没被黑名单列到"就悄悄放行。
ALLOWED_IMPORTS = {
    "logging",
    "typing",
    "urllib.parse",
    "integration_agent.api",
    "integration_agent.tools.models",
    "integration_agent.tools.registry",
}


def _param(name: str, location: str, *, required: bool = False) -> APIParameter:
    return APIParameter(name=name, location=location, required=required)


def _endpoint(
    method: str,
    path: str,
    *,
    operation_id: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    parameters: list[APIParameter] | None = None,
    request_body: APIRequestBody | None = None,
    responses: list[APIResponse] | None = None,
) -> APIEndpoint:
    return APIEndpoint(
        method=method,
        path=path,
        operation_id=operation_id,
        summary=summary,
        tags=tags or [],
        parameters=parameters or [],
        request_body=request_body,
        responses=responses or [],
    )


def _api(
    endpoints: list[APIEndpoint],
    *,
    name: str = "Petstore",
    version: str = "1.0.0",
    base_url: str | None = "https://api.example.com/v1",
) -> APIInfo:
    return APIInfo(
        name=name,
        version=version,
        base_url=base_url,
        description="示例 API",
        endpoints=endpoints,
    )


def _petstore() -> APIInfo:
    """一份典型的小 API：4 个端点，其中一个路径上挂了两个方法。"""
    return _api(
        [
            _endpoint("GET", "/pets", operation_id="listPets", summary="List all pets"),
            _endpoint("POST", "/pets", operation_id="createPets", summary="Create a pet"),
            _endpoint("GET", "/pets/{petId}", operation_id="getPetById", summary="Info for a pet"),
            _endpoint("DELETE", "/pets/{petId}", operation_id="deletePet"),
        ]
    )


def _ctx(api: APIInfo, *, root: Path = FAKE_ROOT) -> ToolContext:
    return ToolContext(
        project_root=root,
        project=ProjectStructure(root=str(root), name="demo", is_python_project=True),
        api=api,
        known_files=set(),
    )


def _invoke(args: dict[str, Any], api: APIInfo | None = None, *, root: Path = FAKE_ROOT):
    return InspectAPITool().invoke(args, _ctx(api or _petstore(), root=root))


def _listed(result) -> list[str]:
    """取端点索引里的条目行（"- " 开头的那些）。"""
    return [line for line in result.content.splitlines() if line.startswith("- ")]


def _assert_sanitized(text: str) -> None:
    """错误文本里不允许出现 traceback 或服务器绝对路径。"""
    assert "Traceback (most recent call last)" not in text
    assert 'File "' not in text
    assert str(FAKE_ROOT) not in text


def _assert_refused(result, *, leak: str | None = None) -> None:
    """断言这是一次**明确的拒绝**，而不是撞上兜底 catch-all 的意外失败。

    只断言 ok is False 是不够的：兜底分支也会返回 ok=False，于是"围栏被拆掉"
    和"围栏正常拒绝"在断言层面长得一模一样。抛出 _ArgRejected 的路径不会经过
    logger.exception，所以"工具内部错误"这五个字是区分二者的标记。
    """
    assert result.ok is False
    assert result.content == ""
    assert result.chars == 0
    assert result.error
    assert "工具内部错误" not in result.error, f"安全围栏疑似失效，落到了兜底分支：{result.error}"
    _assert_sanitized(result.error)
    if leak is not None:
        assert leak not in result.content
        assert leak not in result.error


def _many_endpoints(count: int, *, summary_len: int = 0) -> list[APIEndpoint]:
    return [
        _endpoint(
            "GET",
            f"/resource/{index:03d}",
            operation_id=f"get{index:03d}",
            summary="x" * summary_len if summary_len else None,
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# 场景 1：基础语义
# ---------------------------------------------------------------------------


def test_default_mode_is_summary() -> None:
    result = _invoke({})

    assert result.ok is True
    assert result.content.splitlines()[0].startswith("name:")


def test_summary_reports_the_five_required_fields() -> None:
    result = _invoke({"mode": "summary"})

    assert result.ok is True
    assert result.content.splitlines() == [
        "name: Petstore",
        "version: 1.0.0",
        "base_url: https://api.example.com/v1",
        "endpoints: 4",
        "methods: DELETE, GET, POST",
    ]


def test_summary_methods_are_deduplicated_and_sorted() -> None:
    result = _invoke({"mode": "summary"})

    assert "methods: DELETE, GET, POST" in result.content


def test_summary_handles_an_api_without_endpoints() -> None:
    result = _invoke({"mode": "summary"}, _api([]))

    assert result.ok is True
    assert "endpoints: 0" in result.content
    assert "methods: (none)" in result.content


def test_summary_renders_none_for_absent_optional_fields() -> None:
    result = _invoke({"mode": "summary"}, _api([], base_url=None))

    assert result.ok is True
    assert "base_url: (none)" in result.content


def test_endpoints_mode_lists_every_endpoint_sorted_by_path_then_method() -> None:
    result = _invoke({"mode": "endpoints"})

    assert result.ok is True
    # 按 (path, method) 排序：同路径的不同方法会挨在一起，与文档里的书写顺序无关
    assert _listed(result) == [
        "- GET /pets (listPets) List all pets",
        "- POST /pets (createPets) Create a pet",
        "- DELETE /pets/{petId} (deletePet)",
        "- GET /pets/{petId} (getPetById) Info for a pet",
    ]


def test_endpoints_mode_on_an_empty_api_succeeds_without_filters() -> None:
    result = _invoke({"mode": "endpoints"}, _api([]))

    # 没给过滤条件就只是"这个 API 没有端点"，不是问错了
    assert result.ok is True
    assert "0 个" in result.content
    assert result.truncated is False


def test_method_filter_is_case_insensitive() -> None:
    baseline = _invoke({"mode": "endpoints", "method": "GET"})
    expected = _listed(baseline)

    for raw in ("GET", "get", "Get", "  gEt  "):
        result = _invoke({"mode": "endpoints", "method": raw})
        assert result.ok is True, f"method={raw!r} 意外失败：{result.error}"
        assert _listed(result) == expected, f"method={raw!r} 的结果与 GET 不一致"
    assert len(expected) == 2


def test_path_filter_selects_matching_endpoint() -> None:
    result = _invoke({"mode": "endpoints", "path": "/pets"})

    assert result.ok is True
    assert _listed(result) == [
        "- GET /pets (listPets) List all pets",
        "- POST /pets (createPets) Create a pet",
    ]


def test_path_filter_is_trailing_slash_insensitive() -> None:
    strict = _invoke({"mode": "endpoints", "path": "/pets"})
    lenient = _invoke({"mode": "endpoints", "path": "/pets/"})

    assert lenient.ok is True
    assert lenient.content == strict.content


def test_path_filter_reports_the_apis_own_path_not_the_models_input() -> None:
    result = _invoke({"mode": "endpoints", "path": "/pets/"})

    # 回显的是 APIInfo 里的写法，模型给什么都改不了这一点
    assert "- GET /pets/" not in result.content
    assert "- GET /pets " in result.content


def test_method_and_path_together_return_the_single_matching_endpoint() -> None:
    result = _invoke({"mode": "endpoints", "method": "get", "path": "/pets/{petId}"})

    assert result.ok is True
    assert result.content == "\n".join(
        [
            "端点详情:",
            "GET /pets/{petId}",
            "operation_id: getPetById",
            "summary: Info for a pet",
        ]
    )


def test_single_endpoint_detail_covers_parameters_body_and_responses() -> None:
    endpoint = _endpoint(
        "POST",
        "/pets",
        operation_id="createPets",
        summary="Create a pet",
        tags=["pets", "write"],
        parameters=[_param("x-trace-id", "header"), _param("dry_run", "query", required=True)],
        request_body=APIRequestBody(
            required=True, content_type="application/json", schema_type="object", schema_ref="Pet"
        ),
        responses=[
            APIResponse(status_code="201", content_type="application/json"),
            APIResponse(status_code="default"),
        ],
    )

    result = _invoke({"mode": "endpoints", "method": "POST", "path": "/pets"}, _api([endpoint]))

    assert result.ok is True
    assert result.content.splitlines() == [
        "端点详情:",
        "POST /pets",
        "operation_id: createPets",
        "summary: Create a pet",
        "tags: pets, write",
        "parameters: x-trace-id (header), dry_run (query, required)",
        "request_body: application/json, required, schema: Pet",
        "responses: 201, default",
    ]


def test_single_endpoint_detail_on_a_bare_endpoint_omits_empty_lines() -> None:
    result = _invoke(
        {"mode": "endpoints", "method": "DELETE", "path": "/pets/{petId}"},
        _api([_endpoint("DELETE", "/pets/{petId}")]),
    )

    assert result.ok is True
    assert result.content == "端点详情:\nDELETE /pets/{petId}"


def test_path_filter_alone_does_not_produce_endpoint_detail() -> None:
    # 一个路径上可能有多个方法，"只看路径"拿不到唯一定位，只能给索引
    result = _invoke({"mode": "endpoints", "path": "/pets"})

    assert result.ok is True
    assert "端点详情:" not in result.content
    assert "parameters:" not in result.content


def test_unknown_method_is_a_structured_failure() -> None:
    result = _invoke({"mode": "endpoints", "method": "PATCH"})

    _assert_refused(result)
    assert "PATCH" in result.error
    assert "GET" in result.error  # 顺带告诉模型可用的方法


def test_unknown_path_is_a_structured_failure() -> None:
    result = _invoke({"mode": "endpoints", "path": "/nope"})

    _assert_refused(result)
    assert "/nope" in result.error


def test_unknown_method_and_path_combination_is_a_structured_failure() -> None:
    result = _invoke({"mode": "endpoints", "method": "DELETE", "path": "/pets"})

    _assert_refused(result)


def test_unknown_method_on_an_empty_api_says_so() -> None:
    result = _invoke({"mode": "endpoints", "method": "GET"}, _api([]))

    _assert_refused(result)
    assert "没有任何端点" in result.error


def test_summary_mode_rejects_filters_instead_of_ignoring_them() -> None:
    # 静默忽略比报错更糟：模型会以为自己拿到了过滤后的结果
    _assert_refused(_invoke({"mode": "summary", "method": "GET"}))
    _assert_refused(_invoke({"mode": "summary", "path": "/pets"}))
    _assert_refused(_invoke({"method": "GET"}))


def test_unknown_mode_is_rejected() -> None:
    result = _invoke({"mode": "full"})

    _assert_refused(result)
    assert "mode" in result.error


def test_mode_is_normalized() -> None:
    assert _invoke({"mode": " Summary "}).content == _invoke({"mode": "summary"}).content
    assert _invoke({"mode": "ENDPOINTS"}).content == _invoke({"mode": "endpoints"}).content


def test_blank_or_mistyped_arguments_are_rejected() -> None:
    for args in (
        {"mode": ""},
        {"mode": "   "},
        {"mode": 1},
        {"mode": ["summary"]},
        {"mode": "endpoints", "method": ""},
        {"mode": "endpoints", "method": "   "},
        {"mode": "endpoints", "method": 7},
        {"mode": "endpoints", "method": ["GET"]},
        {"mode": "endpoints", "path": ""},
        {"mode": "endpoints", "path": []},
        {"mode": "endpoints", "path": {"a": 1}},
    ):
        _assert_refused(_invoke(args))


def test_null_arguments_are_treated_as_absent() -> None:
    # 模型经常给可选参数发 null；null 不携带任何过滤意图，为此让它重试没有收益
    assert _invoke({"mode": None}).content == _invoke({"mode": "summary"}).content
    assert (
        _invoke({"mode": "endpoints", "method": None}).content
        == _invoke({"mode": "endpoints"}).content
    )
    assert (
        _invoke({"mode": "endpoints", "path": None}).content
        == _invoke({"mode": "endpoints"}).content
    )


def test_oversized_filters_are_rejected() -> None:
    _assert_refused(_invoke({"mode": "endpoints", "method": "G" * 50}))
    _assert_refused(_invoke({"mode": "endpoints", "path": "/" + "a" * 400}))


def test_extra_arguments_are_ignored() -> None:
    result = _invoke({"mode": "summary", "unexpected": "value"})

    assert result.ok is True


def test_the_tool_never_raises_on_hostile_input() -> None:
    hostile: list[Any] = [
        {},
        {"mode": "endpoints", "method": object()},
        {"mode": {"nested": "dict"}},
        {"path": "/pets", "method": "GET", "mode": "endpoints", "extra": None},
    ]
    for args in hostile:
        result = _invoke(args)
        assert isinstance(result.ok, bool), f"args={args!r} 没有返回结构化结果"


# ---------------------------------------------------------------------------
# 场景 2：输出预算与稳定性
# ---------------------------------------------------------------------------


def test_limits_are_pinned_to_expected_values() -> None:
    assert ACTUAL_LIMITS == EXPECTED_LIMITS


def test_summary_stays_under_its_char_cap() -> None:
    api = _api([], name="n" * 5000, version="v" * 5000)

    result = _invoke({"mode": "summary"}, api)

    assert result.ok is True
    assert len(result.content) <= MAX_SUMMARY_CHARS
    assert result.truncated is True
    assert result.chars == len(result.content)


def test_long_fields_are_clipped_to_the_field_cap() -> None:
    api = _api(
        [_endpoint("GET", "/pets", operation_id="o" * 400, summary="s" * 400)],
        name="n" * 400,
    )

    result = _invoke({"mode": "summary"}, api)
    assert result.truncated is True
    assert "…" in result.content
    # 每个字段都被裁到上限；最长的标签是 "base_url: "，留一点余量
    for line in result.content.splitlines():
        assert len(line) <= MAX_FIELD_CHARS + len("base_url: ") + 1

    # 索引行里的 operation_id 与 summary 同样是"缩短了输出"，也必须报 truncated
    index = _invoke({"mode": "endpoints"}, api)
    assert index.truncated is True
    assert "…" in index.content

    detail = _invoke({"mode": "endpoints", "method": "GET", "path": "/pets"}, api)
    assert detail.truncated is True
    assert "…" in detail.content


def test_index_is_capped_by_line_count_and_says_how_many_were_dropped() -> None:
    api = _api(_many_endpoints(200))

    result = _invoke({"mode": "endpoints"}, api)

    assert result.ok is True
    assert result.truncated is True
    assert len(_listed(result)) == MAX_ENDPOINT_LINES == 40
    # 省略数量必须在标题里：写在列表末尾会被字符上限先切掉
    header = result.content.splitlines()[0]
    assert f"仅列前 {MAX_ENDPOINT_LINES} 个" in header
    assert f"其余 {200 - MAX_ENDPOINT_LINES} 个已省略" in header


def test_output_never_exceeds_the_endpoint_char_cap() -> None:
    api = _api(_many_endpoints(200, summary_len=400))

    for args in (
        {"mode": "endpoints"},
        {"mode": "endpoints", "method": "GET"},
        {"mode": "endpoints", "method": "GET", "path": "/resource/000"},
    ):
        result = _invoke(args, api)
        assert result.ok is True, f"args={args!r} 意外失败：{result.error}"
        assert len(result.content) <= MAX_ENDPOINTS_CHARS, f"args={args!r} 超出字符上限"
        assert result.chars == len(result.content)


def test_truncation_notice_survives_the_char_cap() -> None:
    api = _api(_many_endpoints(200, summary_len=400))

    result = _invoke({"mode": "endpoints"}, api)

    assert result.truncated is True
    assert "输出已被截断" in result.content


def test_chars_is_always_the_content_length() -> None:
    api = _petstore()
    for args in (
        {"mode": "summary"},
        {"mode": "endpoints"},
        {"mode": "endpoints", "method": "GET"},
        {"mode": "endpoints", "method": "GET", "path": "/pets"},
        {"mode": "bad"},
    ):
        result = _invoke(args, api)
        assert result.chars == len(result.content)


def test_not_truncated_for_a_small_api() -> None:
    result = _invoke({"mode": "endpoints"}, _api([_endpoint("GET", "/pets")]))

    assert result.truncated is False
    assert "输出已被截断" not in result.content


def test_repeated_calls_are_identical() -> None:
    api = _petstore()
    for args in (
        {"mode": "summary"},
        {"mode": "endpoints"},
        {"mode": "endpoints", "method": "GET"},
    ):
        first = _invoke(args, api)
        second = _invoke(args, api)
        assert first.content == second.content == _invoke(args, api).content


def test_output_is_independent_of_endpoint_list_order() -> None:
    forward = _petstore()
    backward = _api(list(reversed(forward.endpoints)))

    for args in (
        {"mode": "summary"},
        {"mode": "endpoints"},
        {"mode": "endpoints", "method": "GET"},
        {"mode": "endpoints", "method": "GET", "path": "/pets"},
    ):
        assert _invoke(args, forward).content == _invoke(args, backward).content, (
            f"args={args!r} 的输出依赖了端点列表顺序"
        )


def test_detail_lines_are_capped_when_an_endpoint_has_many_items() -> None:
    endpoint = _endpoint(
        "GET",
        "/pets",
        tags=[f"tag{index}" for index in range(50)],
        parameters=[_param(f"p{index}", "query") for index in range(50)],
        responses=[APIResponse(status_code=str(200 + index)) for index in range(50)],
    )

    result = _invoke({"mode": "endpoints", "method": "GET", "path": "/pets"}, _api([endpoint]))

    assert result.ok is True
    assert result.truncated is True
    assert "其余 35 项" in result.content
    assert "tag49" not in result.content


def test_full_json_schema_is_never_returned() -> None:
    endpoint = _endpoint(
        "POST",
        "/pets",
        operation_id="createPets",
        summary="Create a pet",
        request_body=APIRequestBody(
            content_type="application/json",
            schema_type="object",
            schema_ref="Pet",
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string", "example": SCHEMA_SENTINEL}},
            },
        ),
        responses=[
            APIResponse(
                status_code="200",
                json_schema={"type": "object", "description": SCHEMA_SENTINEL},
            )
        ],
    )
    api = _api([endpoint])

    for args in (
        {"mode": "summary"},
        {"mode": "endpoints"},
        {"mode": "endpoints", "method": "POST"},
        {"mode": "endpoints", "method": "POST", "path": "/pets"},
    ):
        result = _invoke(args, api)
        assert result.ok is True
        assert result.error is None
        assert SCHEMA_SENTINEL not in result.content, f"args={args!r} 泄漏了 json_schema"
        # schema_ref 是组件名（Pet），不是 schema 原文，可以给
        assert "json_schema" not in result.content


def test_endpoint_description_is_not_returned() -> None:
    endpoint = _endpoint("GET", "/pets", summary="List all pets")
    endpoint.description = SCHEMA_SENTINEL
    api = _api([endpoint])

    result = _invoke({"mode": "endpoints", "method": "GET", "path": "/pets"}, api)

    assert SCHEMA_SENTINEL not in result.content


# ---------------------------------------------------------------------------
# 场景 3：安全边界
# ---------------------------------------------------------------------------


def test_inspect_api_makes_no_network_calls(monkeypatch) -> None:
    attempts: list[str] = []

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        attempts.append("network")
        raise AssertionError("工具不应发起网络请求")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    api = _petstore()
    for args in (
        {"mode": "summary"},
        {"mode": "endpoints"},
        {"mode": "endpoints", "method": "GET"},
    ):
        result = _invoke(args, api)
        # ok is True 是必需的：兜底 catch-all 会把"尝试联网"也变成 ok=False，
        # 只断言 ok=False 的话，这条测试对联网行为是瞎的
        assert result.ok is True, f"args={args!r} 意外失败：{result.error}"
    assert attempts == []


def test_inspect_api_does_not_mutate_the_api() -> None:
    api = _petstore()
    before = api.model_dump()
    snapshot = repr(before)

    for args in (
        {"mode": "summary"},
        {"mode": "endpoints"},
        {"mode": "endpoints", "method": "GET"},
        {"mode": "endpoints", "method": "GET", "path": "/pets"},
    ):
        _invoke(args, api)

    assert api.model_dump() == before
    assert repr(api.model_dump()) == snapshot


def test_inspect_api_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "client.py").write_text("x = 1\n", encoding="utf-8")
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    for args in ({"mode": "summary"}, {"mode": "endpoints"}):
        assert _invoke(args, root=tmp_path).ok is True

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before


def test_inspect_api_works_with_an_unreachable_project_root() -> None:
    # 本工具不读项目目录，所以项目根本不存在也不该影响它
    result = _invoke({"mode": "summary"}, root=Path("/definitely/not/here"))

    assert result.ok is True


def test_errors_do_not_leak_the_project_root_or_a_traceback() -> None:
    for args in (
        {"mode": "bad"},
        {"mode": "summary", "method": "GET"},
        {"mode": "endpoints", "method": "PATCH"},
        {"mode": "endpoints", "path": "/nope"},
    ):
        _assert_refused(_invoke(args), leak=str(FAKE_ROOT))


def test_unexpected_exceptions_are_sanitized(monkeypatch) -> None:
    def _boom(_api: APIInfo) -> Any:
        raise RuntimeError(f"内部炸了：{SECRET_SENTINEL} at C:\\server\\secret\\path.py")

    monkeypatch.setattr(api_tools, "_render_summary", _boom)

    result = _invoke({"mode": "summary"})

    assert result.ok is False
    assert result.content == ""
    assert result.chars == 0
    assert result.error
    # 兜底分支会写 logger.exception，但回给模型的只有一句固定文案
    assert SECRET_SENTINEL not in result.error
    assert "RuntimeError" not in result.error
    assert "secret" not in result.error
    _assert_sanitized(result.error)


def test_unexpected_exceptions_are_logged(monkeypatch, caplog) -> None:
    def _boom(_api: APIInfo) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(api_tools, "_render_summary", _boom)

    with caplog.at_level("ERROR"):
        result = _invoke({"mode": "summary"})

    assert result.ok is False
    # 完整堆栈必须留在服务端日志里，否则问题无法排查
    assert any(record.exc_info for record in caplog.records)


def test_base_url_credentials_are_stripped() -> None:
    api = _api([], base_url=f"https://user:{SECRET_SENTINEL}@api.example.com/v1")
    # APIInfo 本身仍然原样保存 URL；去凭据只发生在回给模型的那一份
    assert SECRET_SENTINEL in (api.base_url or "")

    result = _invoke({"mode": "summary"}, api)

    assert result.ok is True
    assert SECRET_SENTINEL not in result.content
    assert "base_url: https://api.example.com/v1" in result.content


def test_base_url_without_credentials_is_untouched() -> None:
    for raw in ("https://api.example.com/v1", "http://localhost:8000", "/v1"):
        result = _invoke({"mode": "summary"}, _api([], base_url=raw))
        assert f"base_url: {raw}" in result.content


def test_api_tools_imports_only_allowlisted_modules() -> None:
    """白名单校验 api_tools.py 的 import：联网 / 子进程 / 文件系统能力一律不得进入。

    做成白名单而不是黑名单：黑名单只能挡住"已知的坏东西"，白名单要求每一个
    新 import 都是一个显式决定。
    """
    source_file = Path(api_tools.__file__).resolve()
    tree = ast.parse(source_file.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)

    assert imported <= ALLOWED_IMPORTS, (
        f"出现了白名单之外的 import：{sorted(imported - ALLOWED_IMPORTS)}"
    )


# ---------------------------------------------------------------------------
# 场景 4：ToolSpec 与默认注册表
# ---------------------------------------------------------------------------


def test_spec_declares_a_read_only_object_schema() -> None:
    spec = InspectAPITool().spec

    assert spec.name == TOOL_NAME
    assert spec.read_only is True
    assert spec.description.strip()
    assert spec.parameters["type"] == "object"
    assert set(spec.parameters["properties"]) == {"mode", "method", "path"}
    assert spec.parameters["properties"]["mode"]["enum"] == ["summary", "endpoints"]
    # 三个参数都可缺省：不带参数调用必须合法（默认 summary）
    assert spec.parameters.get("required", []) == []


def test_spec_parameters_are_json_serializable() -> None:
    spec = InspectAPITool().spec

    assert json.loads(json.dumps(spec.parameters, ensure_ascii=False)) == spec.parameters


def test_spec_rejects_a_parameters_schema_that_is_not_an_object() -> None:
    # ToolSpec 的这道校验由 E2 Commit 1 保证，这里确认本工具没有绕过它
    with pytest.raises(ValueError):
        ToolSpec(name=TOOL_NAME, description="x", parameters={"type": "array"})


def test_tool_satisfies_the_agent_tool_protocol() -> None:
    # AgentTool 是带数据成员的 Protocol，3.10 / 3.11 上只能 isinstance 不能 issubclass
    assert isinstance(InspectAPITool(), AgentTool)
    assert callable(InspectAPITool().invoke)
    assert InspectAPITool().spec.name == TOOL_NAME


def test_describe_is_stable_across_instances() -> None:
    first = ToolRegistry([InspectAPITool()])
    second = ToolRegistry([InspectAPITool()])

    assert first.describe() == second.describe()
    assert first.describe() == first.describe()
    assert TOOL_NAME in first.describe()


def test_registered_in_the_default_registry() -> None:
    registry = build_default_registry()

    tool = registry.get(TOOL_NAME)
    assert tool is not None
    assert isinstance(tool, InspectAPITool)


def test_default_registry_holds_exactly_the_four_tools() -> None:
    registry = build_default_registry()
    names = [spec.name for spec in registry.specs()]

    assert sorted(DEFAULT_TOOL_NAMES) == names
    assert len(names) == 4
    assert len(set(names)) == 4


def test_inspect_api_is_reachable_through_the_registry() -> None:
    tool = build_default_registry().get(TOOL_NAME)
    assert tool is not None

    result = tool.invoke({"mode": "summary"}, _ctx(_petstore()))

    assert result.ok is True
    assert "name: Petstore" in result.content
