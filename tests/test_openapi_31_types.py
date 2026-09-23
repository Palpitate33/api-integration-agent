"""OpenAPI 3.1 联合类型（``type: [string, "null"]``）的回归测试。

背景
----
OpenAPI 3.1 的 Schema Object 就是 JSON Schema 2020-12：``type`` 可以是字符串，
也可以是字符串数组。数组形式下的 ``"null"`` 表示"该值可以为空"。

解析器早期只认字符串，于是这样一份文档会直接在 Pydantic 层炸掉：

    schema_type: list[str] = ...  →  ValidationError: Input should be a valid string

也就是把一份**合法**的 3.1 文档判成了非法文档，抛出的还是 Pydantic 的原始
ValidationError，而不是本模块定义的 OpenAPISpecError。

与 OpenAPI 3.0 的 ``nullable`` 不是一回事
------------------------------------------
3.0 写 ``type: string`` + ``nullable: true``；3.1 写 ``type: [string, "null"]``。
两套语义不能互相翻译：本文件里有专门的测试钉住"``nullable: true`` 不会被当成
联合类型成员"，避免以后有人为了"顺手支持一下 3.0"把两者混在一起。
"""

import pytest

from integration_agent.api import OpenAPISpecError, parse_openapi_text
from integration_agent.api.schema import NULL_TYPE, normalize_schema_types

# ---------------------------------------------------------------- 规范化函数


def test_normalize_keeps_plain_string_type() -> None:
    assert normalize_schema_types("string") == ("string", ["string"])


@pytest.mark.parametrize("declared", ["integer", "number", "boolean", "object", "array", "null"])
def test_normalize_keeps_every_plain_json_schema_type(declared: str) -> None:
    """基础类型一个都不能少：``type`` 是字符串时原样返回。"""
    assert normalize_schema_types(declared) == (declared, [declared])


def test_normalize_reads_a_union() -> None:
    assert normalize_schema_types(["string", NULL_TYPE]) == ("string", ["string", NULL_TYPE])


def test_normalize_takes_the_first_non_null_member_as_primary() -> None:
    """主类型供下游按类型分支，绝不能因为数组里带了 null 就退化成 None。"""
    assert normalize_schema_types([NULL_TYPE, "string"]) == ("string", [NULL_TYPE, "string"])
    assert normalize_schema_types(["array", NULL_TYPE]) == ("array", ["array", NULL_TYPE])


def test_normalize_null_only_union_has_no_primary_type() -> None:
    """只有 null 时不存在主类型，但仍然如实报告声明了什么。"""
    assert normalize_schema_types([NULL_TYPE]) == (None, [NULL_TYPE])


def test_normalize_ignores_non_string_members_instead_of_crashing() -> None:
    """``type`` 是外部输入：成员类型失控时降级，而不是让整份文档解析失败。

    YAML 里写 ``type: [string, null]``（null 不加引号）会被解析成 None——
    这是真实文档里很容易写错的地方。
    """
    assert normalize_schema_types(["string", None, 3]) == ("string", ["string"])


@pytest.mark.parametrize("raw", [None, 3, {}, True])
def test_normalize_degrades_on_unexpected_shapes(raw: object) -> None:
    assert normalize_schema_types(raw) == (None, [])


# ------------------------------------------------------- 完整文档：3.1 联合类型

OPENAPI_31_SPEC = """
openapi: 3.1.0
info: {title: Nullable API, version: "1.0"}
servers: [{url: https://nullable.example.com}]
paths:
  /profiles/{profile_id}:
    get:
      operationId: getProfile
      parameters:
        - name: profile_id
          in: path
          required: true
          schema: {type: string}
        - name: nickname
          in: query
          schema: {type: [string, "null"]}
        - name: age
          in: query
          schema: {type: [integer, "null"]}
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema: {type: [object, "null"]}
    put:
      operationId: replaceProfiles
      requestBody:
        required: true
        content:
          application/json:
            schema: {type: [array, "null"]}
      responses:
        "204": {description: no content}
"""


@pytest.fixture(scope="module")
def api_31():
    """一份合法的 3.1 文档——曾经在这里抛 Pydantic ValidationError。"""
    return parse_openapi_text(OPENAPI_31_SPEC)


def _endpoint(api, operation_id: str):
    return next(item for item in api.endpoints if item.operation_id == operation_id)


def _parameter(endpoint, name: str):
    return next(item for item in endpoint.parameters if item.name == name)


def test_31_document_parses_without_leaking_a_validation_error(api_31) -> None:
    assert api_31.name == "Nullable API"
    assert {e.operation_id for e in api_31.endpoints} == {"getProfile", "replaceProfiles"}


def test_union_parameter_keeps_its_primary_type(api_31) -> None:
    """``type: [string, "null"]`` → 主类型仍然是 string，不是 None。"""
    endpoint = _endpoint(api_31, "getProfile")
    nickname = _parameter(endpoint, "nickname")

    assert nickname.schema_type == "string"
    assert nickname.schema_types == ["string", "null"]


def test_second_union_parameter_is_typed_as_integer(api_31) -> None:
    """再来一个联合类型：``[integer, "null"]`` 必须仍然被当成整数。"""
    age = _parameter(_endpoint(api_31, "getProfile"), "age")

    assert age.schema_type == "integer"
    assert age.schema_types == ["integer", "null"]


def test_plain_string_type_inside_a_31_document_behaves_as_before(api_31) -> None:
    """同一份 3.1 文档里，普通字符串类型的行为不能因为支持联合类型而改变。"""
    profile_id = _parameter(_endpoint(api_31, "getProfile"), "profile_id")

    assert profile_id.location == "path"
    assert profile_id.required is True
    assert profile_id.schema_type == "string"
    assert profile_id.schema_types == ["string"]


def test_union_request_body_keeps_its_primary_type(api_31) -> None:
    body = _endpoint(api_31, "replaceProfiles").request_body

    assert body is not None
    assert body.schema_type == "array"
    assert body.schema_types == ["array", "null"]
    assert body.content_type == "application/json"


def test_union_response_keeps_its_primary_type(api_31) -> None:
    response = _endpoint(api_31, "getProfile").responses[0]

    assert response.status_code == "200"
    assert response.schema_type == "object"
    assert response.schema_types == ["object", "null"]


def test_union_document_does_not_raise_a_raw_validation_error() -> None:
    """把"曾经抛 ValidationError"这件事直接钉住：异常类型必须是本模块自己的。"""
    broken = OPENAPI_31_SPEC.replace("openapi: 3.1.0", "openapi: 2.0.0")
    with pytest.raises(OpenAPISpecError):
        parse_openapi_text(broken)


# --------------------------------------- 3.0 的 nullable 不是 3.1 的联合类型


OPENAPI_30_NULLABLE_SPEC = """
openapi: 3.0.3
info: {title: Nullable 3.0 API, version: "1.0"}
servers: [{url: https://nullable30.example.com}]
paths:
  /profiles:
    get:
      operationId: listProfiles
      parameters:
        - name: nickname
          in: query
          schema: {type: string, nullable: true}
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema: {type: object, nullable: true}
"""


def test_nullable_true_is_not_translated_into_a_type_union() -> None:
    """``nullable: true`` 是 3.0 的写法，解析器不把它翻译成 3.1 的联合类型。

    两套语义混在一起，会让"这个字段可空"和"这个字段是 string 或 null"变得无法区分。
    当前解析器压根不建模 ``nullable``——这个测试就是那条边界的守门人：哪天有人
    顺手做了这层翻译，它会立刻失败，逼着先想清楚语义。
    """
    api = parse_openapi_text(OPENAPI_30_NULLABLE_SPEC)
    endpoint = api.endpoints[0]

    nickname = _parameter(endpoint, "nickname")
    assert nickname.schema_type == "string"
    assert nickname.schema_types == ["string"]
    assert NULL_TYPE not in nickname.schema_types

    response = endpoint.responses[0]
    assert response.schema_type == "object"
    assert response.schema_types == ["object"]


def test_nullable_true_is_preserved_in_the_raw_schema_only() -> None:
    """原始 schema 照原样保留（``nullable`` 还在里面），但结构化字段不据此改写。"""
    api = parse_openapi_text(OPENAPI_30_NULLABLE_SPEC)
    response = api.endpoints[0].responses[0]

    assert response.json_schema == {"type": "object", "nullable": True}
    assert response.schema_types == ["object"]
