"""OpenAPI 3.x 文档解析器：将 YAML / JSON 的 OpenAPI 文档转换为结构化对象。

典型用法：
    from integration_agent.api import parse_openapi, parse_openapi_text

    api = parse_openapi("examples/openapi/petstore.yaml")
    api = parse_openapi_text(raw_spec_text)  # 自动识别 JSON / YAML

限制（当前阶段）：
    - 仅支持 OpenAPI 3.x；仅解析文档内（#/...）引用，外部文件引用不支持。
    - 认证只提取根级 security 的第一个方案，忽略端点级 security 覆盖。
"""

import json
import re
from pathlib import Path
from typing import Any

import yaml

from integration_agent.api.schema import (
    APIAuth,
    APIEndpoint,
    APIInfo,
    APIParameter,
    APIRequestBody,
    APIResponse,
    normalize_schema_types,
)

# path 模板里的参数占位符：`/users/{user_id}/posts/{post_id}` → user_id / post_id
_PATH_TEMPLATE_PARAMETER = re.compile(r"\{([^{}]+)\}")

# 必须是有序序列：set 的迭代顺序随字符串哈希种子变化，会导致同一份文档
# 在不同进程解析出不同的端点顺序。此处按 OpenAPI Path Item 的字段顺序排列。
HTTP_METHODS: tuple[str, ...] = (
    "get",
    "put",
    "post",
    "delete",
    "options",
    "head",
    "patch",
    "trace",
)


class OpenAPISpecError(ValueError):
    """OpenAPI 文档缺失、格式错误或不受支持时抛出。"""


def parse_openapi(source: str | Path) -> APIInfo:
    """便捷入口：从文件加载并解析 OpenAPI 文档为结构化 APIInfo。"""
    return build_api_info(load_spec(source))


def parse_openapi_text(text: str, *, source: str = "<text>") -> APIInfo:
    """便捷入口：解析 OpenAPI 文本为结构化 APIInfo（适合从 URL 拉取 spec 的场景）。"""
    return build_api_info(load_spec_text(text, source=source))


def load_spec(source: str | Path) -> dict[str, Any]:
    """从文件加载 OpenAPI 文档，返回原始 dict。

    按文件扩展名识别格式：.json 走 JSON 解析，.yaml/.yml 走 YAML 解析，
    其他扩展名先尝试 JSON，失败后回退到 YAML。
    """
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise OpenAPISpecError(f"无法读取 OpenAPI 文件 {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        fmt = "json"
    elif path.suffix.lower() in (".yaml", ".yml"):
        fmt = "yaml"
    else:
        fmt = None
    return load_spec_text(text, fmt=fmt, source=str(path))


def load_spec_text(text: str, *, fmt: str | None = None, source: str = "<text>") -> dict[str, Any]:
    """解析 YAML / JSON 文本为 dict。

    fmt 可显式指定 "json" 或 "yaml"；缺省时先尝试 JSON，失败后回退到 YAML。
    """
    if fmt == "json":
        document = _parse_json(text, source)
    elif fmt == "yaml":
        document = _parse_yaml(text, source)
    else:
        try:
            document = _parse_json(text, source)
        except OpenAPISpecError:
            document = _parse_yaml(text, source)
    if not isinstance(document, dict):
        raise OpenAPISpecError(
            f"OpenAPI 文档顶层必须是对象（{source} 中为 {type(document).__name__}）"
        )
    return document


def build_api_info(spec: dict[str, Any]) -> APIInfo:
    """校验并转换 OpenAPI 3.x 文档为结构化对象；文档无效时抛出 OpenAPISpecError。"""
    if "openapi" not in spec:
        raise OpenAPISpecError("缺少 'openapi' 字段：不是合法的 OpenAPI 文档")
    openapi_version = spec["openapi"]
    if not isinstance(openapi_version, str):
        raise OpenAPISpecError(
            f"'openapi' 字段必须是字符串，实际为 {type(openapi_version).__name__}："
            f"{openapi_version!r}"
        )
    if not openapi_version.startswith("3."):
        raise OpenAPISpecError(f"不支持的 OpenAPI 版本 '{openapi_version}'：仅支持 OpenAPI 3.x")

    info = spec.get("info")
    if not isinstance(info, dict):
        raise OpenAPISpecError("缺少 'info' 字段：不是合法的 OpenAPI 文档")
    name = _require_string(info, "info.title", "API 名称")
    version = _require_string(info, "info.version", "API 版本")

    paths = spec.get("paths")
    if not isinstance(paths, dict):
        raise OpenAPISpecError("缺少 'paths' 字段")

    endpoints: list[APIEndpoint] = []
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        path_parameters = _extract_parameters(
            path_item.get("parameters"), spec, context=f"path {path}"
        )
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            endpoints.append(
                _build_endpoint(method.upper(), path, operation, path_parameters, spec)
            )

    return APIInfo(
        name=name,
        version=version,
        base_url=_extract_base_url(spec),
        description=info.get("description"),
        auth=_extract_auth(spec),
        endpoints=endpoints,
    )


def _require_string(container: dict[str, Any], field: str, description: str) -> str:
    """校验必需的字符串字段；缺失、为空或类型错误时抛出 OpenAPISpecError。

    YAML 中未加引号的 `version: 1.0` 会被解析为 float，必须在进入 Pydantic 模型前
    拦截，否则会泄漏 ValidationError 而不是本模块定义的 OpenAPISpecError。

    Args:
        container: 字段所在的对象（如 info）。
        field: 带前缀的字段名，用于错误信息（如 "info.title"）。
        description: 字段的中文说明（如 "API 名称"）。
    """
    value = container.get(field.rsplit(".", 1)[-1])
    if value is None or value == "":
        raise OpenAPISpecError(f"缺少 '{field}'（{description}）")
    if not isinstance(value, str):
        raise OpenAPISpecError(f"'{field}' 必须是字符串，实际为 {type(value).__name__}：{value!r}")
    return value


def _build_endpoint(
    method: str,
    path: str,
    operation: dict[str, Any],
    path_parameters: list[APIParameter],
    spec: dict[str, Any],
) -> APIEndpoint:
    """构建单个端点。path 级参数与 operation 级参数合并，后者同名覆盖前者。"""
    responses_raw = operation.get("responses")
    if not isinstance(responses_raw, dict):
        raise OpenAPISpecError(f"{method} {path}: 缺少 'responses' 字段")

    merged: dict[tuple[str, str], APIParameter] = {
        (param.name, param.location): param for param in path_parameters
    }
    operation_parameters = _extract_parameters(
        operation.get("parameters"), spec, context=f"{method} {path}"
    )
    merged.update({(param.name, param.location): param for param in operation_parameters})
    parameters = list(merged.values())
    parameters.extend(_undeclared_path_parameters(path, parameters))

    tags = operation.get("tags")
    return APIEndpoint(
        method=method,
        path=path,
        operation_id=operation.get("operationId"),
        summary=operation.get("summary"),
        description=operation.get("description"),
        tags=tags if isinstance(tags, list) else [],
        parameters=parameters,
        request_body=_build_request_body(
            operation.get("requestBody"), spec, context=f"{method} {path}"
        ),
        responses=[
            _build_response(status, resp, spec, context=f"{method} {path}")
            for status, resp in responses_raw.items()
        ],
    )


def _undeclared_path_parameters(path: str, declared: list[APIParameter]) -> list[APIParameter]:
    """补上 path 模板里出现、但 ``parameters`` 中未声明的路径参数。

    OpenAPI 要求 path 参数必须显式声明，但真实文档经常漏写。漏写时 `{user_id}`
    不会被任何一层替换：生成的客户端把字面量 ``{user_id}`` 发出去，而生成的测试
    又按"这个端点有参数"来构造调用——两边对不上。模板本身是这段契约唯一的事实
    来源，所以以模板为准补齐；没有声明就没有类型可推断，按 string 处理。

    按模板出现顺序追加在已声明参数之后，保证同一份文档的解析结果稳定。
    """
    known = {param.name for param in declared if param.location == "path"}
    return [
        APIParameter(
            name=name,
            location="path",
            required=True,
            description="由 path 模板推导：OpenAPI 未显式声明该路径参数",
            schema_type="string",
            schema_types=["string"],
        )
        for name in dict.fromkeys(_PATH_TEMPLATE_PARAMETER.findall(path))
        if name not in known
    ]


def _extract_parameters(params: Any, spec: dict[str, Any], *, context: str) -> list[APIParameter]:
    """提取参数列表，支持 $ref 引用 components/parameters 中的定义。"""
    if params is None:
        return []
    if not isinstance(params, list):
        raise OpenAPISpecError(f"{context}: 'parameters' 必须是数组")
    result: list[APIParameter] = []
    for entry in params:
        if not isinstance(entry, dict):
            raise OpenAPISpecError(f"{context}: parameters 中的条目必须是对象")
        if "$ref" in entry:
            entry = _resolve_local_ref(spec, entry["$ref"])
            if not isinstance(entry, dict):
                raise OpenAPISpecError(f"{context}: 无法解析参数引用")
        name = entry.get("name")
        location = entry.get("in")
        if not isinstance(name, str) or not isinstance(location, str):
            raise OpenAPISpecError(f"{context}: 参数缺少 'name' 或 'in' 字段")
        schema = entry.get("schema") if isinstance(entry.get("schema"), dict) else {}
        schema_type, schema_types = normalize_schema_types(schema.get("type"))
        result.append(
            APIParameter(
                name=name,
                location=location,
                required=bool(entry.get("required", False)),
                description=entry.get("description"),
                schema_type=schema_type,
                schema_types=schema_types,
                schema_format=schema.get("format"),
                default=schema.get("default"),
                enum=schema.get("enum"),
            )
        )
    return result


def _build_request_body(body: Any, spec: dict[str, Any], *, context: str) -> APIRequestBody | None:
    """构建请求体表示，支持 $ref 引用 components/requestBodies 中的定义。"""
    if body is None:
        return None
    if not isinstance(body, dict):
        raise OpenAPISpecError(f"{context}: 'requestBody' 必须是对象")
    if "$ref" in body:
        body = _resolve_local_ref(spec, body["$ref"])
        if not isinstance(body, dict):
            raise OpenAPISpecError(f"{context}: 无法解析 requestBody 引用")
    content_type, media = _pick_media(body.get("content"))
    schema_type, schema_types, schema_ref, schema = _extract_schema(
        media.get("schema") if media else None
    )
    return APIRequestBody(
        required=bool(body.get("required", False)),
        description=body.get("description"),
        content_type=content_type,
        schema_type=schema_type,
        schema_types=schema_types,
        schema_ref=schema_ref,
        json_schema=schema,
    )


def _build_response(status: Any, resp: Any, spec: dict[str, Any], *, context: str) -> APIResponse:
    """构建响应表示，支持 $ref 引用 components/responses 中的定义。"""
    if not isinstance(resp, dict):
        raise OpenAPISpecError(f"{context}: 响应 {status!r} 必须是对象")
    if "$ref" in resp:
        resp = _resolve_local_ref(spec, resp["$ref"])
        if not isinstance(resp, dict):
            raise OpenAPISpecError(f"{context}: 无法解析响应 {status!r} 的引用")
    content_type, media = _pick_media(resp.get("content"))
    schema_type, schema_types, schema_ref, schema = _extract_schema(
        media.get("schema") if media else None
    )
    return APIResponse(
        status_code=str(status),
        description=resp.get("description"),
        content_type=content_type,
        schema_type=schema_type,
        schema_types=schema_types,
        schema_ref=schema_ref,
        json_schema=schema,
    )


def _pick_media(content: Any) -> tuple[str | None, dict | None]:
    """选择最主要的 media type：优先 application/json，否则取第一个。"""
    if not isinstance(content, dict) or not content:
        return None, None
    if "application/json" in content:
        media = content["application/json"]
        return "application/json", media if isinstance(media, dict) else None
    first_type, media = next(iter(content.items()))
    return first_type, media if isinstance(media, dict) else None


def _extract_schema(schema: Any) -> tuple[str | None, list[str], str | None, dict | None]:
    """提取 schema 的简化信息：返回 (顶层类型, 类型集合, $ref 名称, 原始 schema)。"""
    if not isinstance(schema, dict):
        return None, [], None, None
    ref = schema.get("$ref")
    ref_name = ref.rsplit("/", 1)[-1] if isinstance(ref, str) and ref else None
    schema_type, schema_types = normalize_schema_types(schema.get("type"))
    return schema_type, schema_types, ref_name, schema


def _extract_auth(spec: dict[str, Any]) -> APIAuth | None:
    """提取全局认证：取根级 security 第一个要求对象中的第一个方案。"""
    security = spec.get("security")
    if not isinstance(security, list) or not security:
        return None
    first_requirement = security[0]
    if not isinstance(first_requirement, dict) or not first_requirement:
        return None
    scheme_name = next(iter(first_requirement))
    schemes = spec.get("components", {}).get("securitySchemes")
    if not isinstance(schemes, dict):
        return None
    scheme = schemes.get(scheme_name)
    if isinstance(scheme, dict) and "$ref" in scheme:
        scheme = _resolve_local_ref(spec, scheme["$ref"])
    if not isinstance(scheme, dict):
        return None
    return APIAuth(
        type=scheme.get("type", "unknown"),
        description=scheme.get("description"),
        scheme=scheme.get("scheme"),
        bearer_format=scheme.get("bearerFormat"),
        name=scheme.get("name"),
        location=scheme.get("in"),
        flows=scheme.get("flows"),
    )


def _extract_base_url(spec: dict[str, Any]) -> str | None:
    """提取 servers[0].url 作为 base URL。"""
    servers = spec.get("servers")
    if not isinstance(servers, list) or not servers or not isinstance(servers[0], dict):
        return None
    url = servers[0].get("url")
    return url if isinstance(url, str) else None


def _resolve_local_ref(spec: dict[str, Any], ref: str) -> Any:
    """解析文档内引用（'#/components/...' 形式）；外部引用或解析失败返回 None。"""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    current: Any = spec
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _parse_json(text: str, source: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenAPISpecError(f"{source} 不是有效的 JSON：{exc}") from exc


def _parse_yaml(text: str, source: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OpenAPISpecError(f"{source} 不是有效的 YAML：{exc}") from exc
