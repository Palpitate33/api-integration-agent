"""API Understanding 阶段的核心数据模型。

将 OpenAPI 3.x 文档转换为适合 Agent 推理的紧凑结构化对象：
只保留端点、参数、请求体、响应与认证等关键信息，
避免把整份 OpenAPI 文档直接塞进 LLM 上下文。
"""

from typing import Any

from pydantic import BaseModel, Field


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
    schema_format: str | None = None  # 附加格式，如 int32、uuid
    default: Any | None = None
    enum: list[Any] | None = None  # 可选值列表


class APIRequestBody(BaseModel):
    """请求体（requestBody）的简化表示。"""

    required: bool = False
    description: str | None = None
    content_type: str | None = None  # 主要 media type，如 application/json
    schema_type: str | None = None  # schema 顶层类型，如 object、array
    schema_ref: str | None = None  # $ref 指向 components/schemas 的名称
    json_schema: dict[str, Any] | None = None  # 原始 JSON Schema（内部 $ref 保持原样）


class APIResponse(BaseModel):
    """单个响应状态的简化表示。"""

    status_code: str  # 如 "200"、"404"、"default"
    description: str | None = None
    content_type: str | None = None
    schema_type: str | None = None
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
