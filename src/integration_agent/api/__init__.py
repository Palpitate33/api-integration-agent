"""API Parser 阶段：解析第三方 OpenAPI Spec，提取端点、模型与认证方式。

主要 API：
    parse_openapi(path)       —— 从文件加载并解析（自动识别 JSON / YAML）
    parse_openapi_text(text)  —— 从文本解析（适合从 URL 拉取 spec 的场景）

结构化对象：
    APIInfo / APIEndpoint / APIParameter / APIRequestBody / APIResponse / APIAuth
"""

from integration_agent.api.parser import (
    OpenAPISpecError,
    build_api_info,
    load_spec,
    load_spec_text,
    parse_openapi,
    parse_openapi_text,
)
from integration_agent.api.schema import (
    APIAuth,
    APIEndpoint,
    APIInfo,
    APIParameter,
    APIRequestBody,
    APIResponse,
)

__all__ = [
    "APIAuth",
    "APIEndpoint",
    "APIInfo",
    "APIParameter",
    "APIRequestBody",
    "APIResponse",
    "OpenAPISpecError",
    "build_api_info",
    "load_spec",
    "load_spec_text",
    "parse_openapi",
    "parse_openapi_text",
]
