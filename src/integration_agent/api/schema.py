"""API Understanding 阶段的核心数据模型。

将 OpenAPI 3.x 文档转换为适合 Agent 推理的紧凑结构化对象：
只保留端点、参数、请求体、响应与认证等关键信息，
避免把整份 OpenAPI 文档直接塞进 LLM 上下文。
"""

from typing import Any

from pydantic import BaseModel, Field

# JSON Schema 中表示"空值"的那个类型名（OpenAPI 3.1 / JSON Schema 2020-12）。
NULL_TYPE = "null"


def normalize_schema_types(raw: Any) -> tuple[str | None, list[str]]:
    """把 schema 的 ``type`` 规范化成 ``(主类型, 声明的完整类型集合)``。

    JSON Schema 2020-12（即 OpenAPI 3.1 的 Schema Object）允许 ``type`` 是字符串
    **或**字符串数组，数组表示联合类型：

        type: string                 → ("string", ["string"])
        type: [string, "null"]       → ("string", ["string", "null"])
        type: [integer, "null"]      → ("integer", ["integer", "null"])
        type: ["null"]               → (None, ["null"])

    主类型取第一个非 ``null`` 的成员：下游（生成器 / Planner / 工具）按它分支，
    因此 ``type: [array, "null"]`` 必须仍然被当成 array 处理，而不是退化成 None。

    与 OpenAPI 3.0 的 ``nullable`` 区分
    -----------------------------------
    3.0 用 ``type: string`` + ``nullable: true`` 表达"string 或 null"，这是**另一套
    语义**。本函数只解析 ``type`` 本身，不读取也不翻译 ``nullable``：把两者混为
    一谈会让 3.1 的联合类型被误当成 3.0 的可空标记（反之亦然）。当前解析器不建模
    ``nullable``，文档版本也未逐条携带到 schema 层。

    Args:
        raw: schema 中 ``type`` 字段的原始取值，可能是字符串、数组或任意其它类型。

    Returns:
        ``(主类型, 类型集合)``；无法识别时返回 ``(None, [])``，不抛异常——
        ``type`` 是外部输入，形状失控时应当降级而不是让整份文档解析失败。
    """
    if isinstance(raw, str):
        return raw, [raw]
    if isinstance(raw, list):
        declared = [item for item in raw if isinstance(item, str)]
        primary = next((item for item in declared if item != NULL_TYPE), None)
        return primary, declared
    return None, []


class APIAuth(BaseModel):
    """API 认证方式（来自 components.securitySchemes 的简化表示）。"""

    type: str  # http / apiKey / oauth2 / openIdConnect
    description: str | None = None
    scheme: str | None = None  # http 认证方案，如 basic、bearer
    bearer_format: str | None = None  # bearer token 格式，如 JWT
    name: str | None = None  # apiKey 的参数名
    location: str | None = None  # apiKey 的位置：header / query / cookie
    flows: dict[str, Any] | None = None  # oauth2 flows 的简化表示


class APIParameter(BaseModel):
    """单个请求参数（path / query / header / cookie）。"""

    name: str
    location: str  # path / query / header / cookie
    required: bool = False
    description: str | None = None
    schema_type: str | None = None  # 参数类型，如 string、integer
    schema_types: list[str] = Field(default_factory=list)  # 声明的完整类型集合（3.1 union）
    schema_format: str | None = None  # 附加格式，如 int32、uuid
    default: Any | None = None
    enum: list[Any] | None = None  # 可选值列表


class APIRequestBody(BaseModel):
    """请求体（requestBody）的简化表示。"""

    required: bool = False
    description: str | None = None
    content_type: str | None = None  # 主要 media type，如 application/json
    schema_type: str | None = None  # schema 顶层类型，如 object、array
    schema_types: list[str] = Field(default_factory=list)  # 声明的完整类型集合（3.1 union）
    schema_ref: str | None = None  # $ref 指向 components/schemas 的名称
    json_schema: dict[str, Any] | None = None  # 原始 JSON Schema（内部 $ref 保持原样）


class APIResponse(BaseModel):
    """单个响应状态的简化表示。"""

    status_code: str  # 如 "200"、"404"、"default"
    description: str | None = None
    content_type: str | None = None
    schema_type: str | None = None
    schema_types: list[str] = Field(default_factory=list)  # 声明的完整类型集合（3.1 union）
    schema_ref: str | None = None
    json_schema: dict[str, Any] | None = None  # 原始 JSON Schema（内部 $ref 保持原样）


class APIEndpoint(BaseModel):
    """单个 API 端点（一个 HTTP method + path）。"""

    method: str  # 大写 HTTP method，如 GET、POST
    path: str
    operation_id: str | None = None
    summary: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    parameters: list[APIParameter] = Field(default_factory=list)
    request_body: APIRequestBody | None = None
    responses: list[APIResponse] = Field(default_factory=list)


class APIInfo(BaseModel):
    """一份 OpenAPI 3.x 文档的结构化表示。"""

    name: str  # info.title
    version: str  # info.version
    base_url: str | None = None  # servers[0].url
    description: str | None = None
    auth: APIAuth | None = None  # 全局认证（根级 security 的第一个方案）
    endpoints: list[APIEndpoint] = Field(default_factory=list)
