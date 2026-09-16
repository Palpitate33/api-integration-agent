"""API Parser 单元测试。"""

import json
from pathlib import Path

import pytest

from integration_agent.api import (
    APIAuth,
    APIInfo,
    OpenAPISpecError,
    parse_openapi,
    parse_openapi_text,
)
from integration_agent.api.parser import HTTP_METHODS

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "examples" / "openapi" / "petstore.yaml"

MINIMAL_JSON = {
    "openapi": "3.0.3",
    "info": {"title": "JSON API", "version": "2.0"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/items": {
            "get": {
                "operationId": "listItems",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}


@pytest.fixture(scope="module")
def api_info() -> APIInfo:
    return parse_openapi(FIXTURE_PATH)


def test_basic_info(api_info: APIInfo) -> None:
    assert api_info.name == "Demo Petstore API"
    assert api_info.version == "1.0.0"
    assert api_info.base_url == "https://petstore.example.com/v1"
    assert api_info.description


def test_global_auth(api_info: APIInfo) -> None:
    assert isinstance(api_info.auth, APIAuth)
    assert api_info.auth.type == "apiKey"
    assert api_info.auth.name == "X-API-Key"
    assert api_info.auth.location == "header"


def test_endpoints_discovered(api_info: APIInfo) -> None:
    keys = {(e.method, e.path) for e in api_info.endpoints}
    assert keys == {("GET", "/pets"), ("POST", "/pets"), ("GET", "/pets/{petId}")}


def test_list_pets_parameters(api_info: APIInfo) -> None:
    endpoint = next(e for e in api_info.endpoints if e.operation_id == "listPets")
    assert endpoint.summary == "List all pets"
    assert endpoint.description
    assert endpoint.tags == ["pets"]
    assert len(endpoint.parameters) == 1
    param = endpoint.parameters[0]
    assert param.name == "limit"
    assert param.location == "query"
    assert param.required is False
    assert param.schema_type == "integer"
    assert param.schema_format == "int32"


def test_list_pets_response_schema(api_info: APIInfo) -> None:
    endpoint = next(e for e in api_info.endpoints if e.operation_id == "listPets")
    response = endpoint.responses[0]
    assert response.status_code == "200"
    assert response.content_type == "application/json"
    assert response.schema_type == "array"
    assert response.json_schema  # 原始 JSON Schema 保留


def test_get_pet_response_schema_ref(api_info: APIInfo) -> None:
    endpoint = next(e for e in api_info.endpoints if e.operation_id == "getPet")
    assert [r.status_code for r in endpoint.responses] == ["200", "404"]
    ok = endpoint.responses[0]
    assert ok.content_type == "application/json"
    assert ok.schema_type is None
    assert ok.schema_ref == "Pet"


def test_path_level_parameter_merge(api_info: APIInfo) -> None:
    endpoint = next(e for e in api_info.endpoints if e.operation_id == "getPet")
    param = endpoint.parameters[0]
    assert param.name == "petId"
    assert param.location == "path"
    assert param.required is True


def test_request_body(api_info: APIInfo) -> None:
    endpoint = next(e for e in api_info.endpoints if e.operation_id == "createPet")
    assert endpoint.request_body is not None
    assert endpoint.request_body.required is True
    assert endpoint.request_body.content_type == "application/json"
    assert endpoint.request_body.schema_ref == "NewPet"


def test_json_text_parsing() -> None:
    api = parse_openapi_text(json.dumps(MINIMAL_JSON))
    assert api.name == "JSON API"
    assert api.version == "2.0"
    assert api.base_url == "https://api.example.com"
    assert len(api.endpoints) == 1
    assert api.endpoints[0].method == "GET"


def test_json_file_loading(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(MINIMAL_JSON), encoding="utf-8")
    api = parse_openapi(path)
    assert api.name == "JSON API"


def test_yaml_text_parsing() -> None:
    text = """
openapi: 3.1.0
info:
  title: YAML API
  version: "1"
paths: {}
"""
    api = parse_openapi_text(text)
    assert api.name == "YAML API"
    assert api.endpoints == []


def test_unquoted_status_code_is_coerced_to_string() -> None:
    text = """
openapi: 3.0.3
info: {title: T, version: "1"}
paths:
  /x:
    get:
      responses:
        200:
          description: ok
"""
    api = parse_openapi_text(text)
    assert api.endpoints[0].responses[0].status_code == "200"


def test_parameter_ref_resolution() -> None:
    text = """
openapi: 3.0.3
info: {title: T, version: "1"}
paths:
  /x:
    get:
      parameters:
        - $ref: "#/components/parameters/Page"
      responses:
        "200": {description: ok}
components:
  parameters:
    Page:
      name: page
      in: query
      schema: {type: integer}
"""
    api = parse_openapi_text(text)
    param = api.endpoints[0].parameters[0]
    assert param.name == "page"
    assert param.schema_type == "integer"


def test_http_methods_order_is_fixed() -> None:
    """P1 回归：HTTP_METHODS 必须是有序序列。

    set 的迭代顺序随字符串哈希种子变化，会让同一份文档在不同进程解析出不同的
    端点顺序（曾实测 PYTHONHASHSEED=4 时 /pets 的 GET/POST 顺序翻转）。
    """
    assert not isinstance(HTTP_METHODS, (set, frozenset))
    assert list(HTTP_METHODS) == [
        "get",
        "put",
        "post",
        "delete",
        "options",
        "head",
        "patch",
        "trace",
    ]


def test_endpoint_order_follows_canonical_method_order() -> None:
    """P1 回归：无论文档中方法声明顺序如何，解析结果顺序都应一致。"""
    text = """
openapi: 3.0.3
info: {title: T, version: "1"}
paths:
  /x:
    trace: {responses: {"200": {description: ok}}}
    patch: {responses: {"200": {description: ok}}}
    get: {responses: {"200": {description: ok}}}
    post: {responses: {"200": {description: ok}}}
    delete: {responses: {"200": {description: ok}}}
"""
    api = parse_openapi_text(text)
    assert [endpoint.method for endpoint in api.endpoints] == [
        "GET",
        "POST",
        "DELETE",
        "PATCH",
        "TRACE",
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "顶层必须是对象"),
        ("just a string", "顶层必须是对象"),
        ('{"swagger": "2.0"}', "缺少 'openapi'"),
        (
            '{"openapi": null, "info": {"title": "t", "version": "1"}, "paths": {}}',
            "'openapi' 字段必须是字符串",
        ),
        (
            '{"openapi": 3.0, "info": {"title": "t", "version": "1"}, "paths": {}}',
            "'openapi' 字段必须是字符串",
        ),
        (
            '{"openapi": "2.0", "info": {"title": "t", "version": "1"}, "paths": {}}',
            "仅支持 OpenAPI 3.x",
        ),
        ('{"openapi": "3.0.3"}', "缺少 'info'"),
        ('{"openapi": "3.0.3", "info": {}}', "缺少 'info.title'"),
        ('{"openapi": "3.0.3", "info": {"title": "t"}}', "缺少 'info.version'"),
        ('{"openapi": "3.0.3", "info": {"title": "t", "version": "1"}}', "缺少 'paths'"),
        (
            '{"openapi": "3.0.3", "info": {"title": "t", "version": "1"}, '
            '"paths": {"/x": {"get": {}}}}',
            "缺少 'responses'",
        ),
    ],
)
def test_invalid_documents_raise(text: str, message: str) -> None:
    with pytest.raises(OpenAPISpecError, match=message):
        parse_openapi_text(text)


def _yaml_spec(info_body: str, openapi_version: str = "3.0.3") -> str:
    """构造一份最小 YAML OpenAPI 文档，用于测试字段类型校验。"""
    return f"openapi: {openapi_version}\ninfo:\n{info_body}\npaths: {{}}\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # YAML 中未加引号的 1.0 / 1 会被解析为 float / int
        (_yaml_spec("  title: T\n  version: 1.0"), "'info.version' 必须是字符串"),
        (_yaml_spec("  title: T\n  version: 1"), "'info.version' 必须是字符串"),
        (_yaml_spec('  title: 123\n  version: "1"'), "'info.title' 必须是字符串"),
        (
            _yaml_spec('  title: T\n  version: "1"', openapi_version="3.0"),
            "'openapi' 字段必须是字符串",
        ),
    ],
)
def test_non_string_scalars_raise_spec_error(text: str, message: str) -> None:
    """P2 回归：非字符串标量必须抛 OpenAPISpecError，不得泄漏 Pydantic ValidationError。

    ValidationError 与本异常同为 ValueError 子类但互不继承，因此若回归，
    pytest.raises(OpenAPISpecError) 会捕获失败。
    """
    with pytest.raises(OpenAPISpecError, match=message):
        parse_openapi_text(text)


def test_number_like_string_version_is_accepted() -> None:
    """加引号的 version 是合法写法，不应被类型校验误伤。"""
    text = 'openapi: 3.0.3\ninfo:\n  title: T\n  version: "1.0"\npaths: {}\n'
    assert parse_openapi_text(text).version == "1.0"


def test_malformed_yaml_raises() -> None:
    with pytest.raises(OpenAPISpecError, match="不是有效的 YAML"):
        parse_openapi_text("openapi: [unclosed")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OpenAPISpecError, match="无法读取"):
        parse_openapi(tmp_path / "nope.yaml")
