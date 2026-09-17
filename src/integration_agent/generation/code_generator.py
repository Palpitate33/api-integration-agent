"""Deterministic Code Generator：把 IntegrationPlan 转换为 GeneratedArtifacts。

本阶段是纯生成组件：
    - 不写入磁盘、不修改仓库、不执行命令、不发起网络请求、不读取 secret。
    - 不接 LLM：所有输出由模板规则确定，同样输入必定同样输出
      （无 random / 时间戳 / UUID / 无序集合）。
    - 只消费 IntegrationPlan 携带的结构化信息，不重新解析 OpenAPI、
      不重新扫描仓库。

生成范围（全部来自 plan 的结构化字段）：
    - files_to_create   → 完整文件内容（client / models / exceptions / config / __init__）
    - files_to_modify   → 结构化修改建议（插入片段 + 插入点，不重写整个文件）
    - dependencies      → DependencyChange（already_installed 的不重复添加）
    - endpoints         → 每个 PlannedEndpoint 一个方法（method / path / operationId / parameters）
    - authentication    → 环境变量读取 + 请求头注入，绝不硬编码 secret
    - error_handling    → 状态码到异常的映射；retryable=True 才生成 429/5xx 重试与退避
    - testing_strategy  → 测试文件骨架（只生成代码，不执行测试）
"""

import re
from typing import Protocol, runtime_checkable

from integration_agent.agent import (
    AuthenticationPlan,
    ErrorHandlingRule,
    FileModification,
    IntegrationPlan,
    PlannedEndpoint,
    TestSpec,
)
from integration_agent.api import APIEndpoint, APIResponse
from integration_agent.generation.models import (
    DependencyChange,
    GeneratedArtifacts,
    GeneratedFile,
)

# OpenAPI schema 类型 → Python 类型提示
SCHEMA_TYPE_HINTS = {
    "integer": "int",
    "number": "float",
    "string": "str",
    "boolean": "bool",
    "array": "list[Any]",
    "object": "dict[str, Any]",
}

# 错误类别 → 生成代码中的人类可读标签
ERROR_LABELS = {
    "validation": "请求参数错误",
    "authentication": "认证失败",
    "authorization": "权限不足",
    "not_found": "资源不存在",
    "rate_limit": "请求过于频繁（限流）",
    "server_error": "服务端错误",
}


@runtime_checkable
class CodeGenerator(Protocol):
    """Code Generator 接口：确定性实现与未来的 LLMCodeGenerator 共用同一契约。"""

    def generate(self, plan: IntegrationPlan) -> GeneratedArtifacts:
        """基于集成方案产出生成物。"""
        ...


def generate_code(plan: IntegrationPlan) -> GeneratedArtifacts:
    """便捷入口：默认使用 DeterministicCodeGenerator。"""
    return DeterministicCodeGenerator().generate(plan)


class DeterministicCodeGenerator:
    """规则驱动的确定性代码生成器：IntegrationPlan → GeneratedArtifacts。"""

    def generate(self, plan: IntegrationPlan) -> GeneratedArtifacts:
        files: list[GeneratedFile] = []
        files.extend(self._generated_created(plan))
        files.extend(self._generated_modified(plan))
        files.extend(self._generated_tests(plan))
        dependency_changes = self._dependency_changes(plan)
        return GeneratedArtifacts(
            files=files,
            dependency_changes=dependency_changes,
            summary=_summary(plan, files, dependency_changes),
            warnings=_warnings(plan, dependency_changes),
        )

    # ---------------------------------------------------------- files_to_create

    def _generated_created(self, plan: IntegrationPlan) -> list[GeneratedFile]:
        generated: list[GeneratedFile] = []
        for item in plan.files_to_create:
            if item.kind == "client":
                content = self._client_module(plan)
            elif item.kind == "models":
                content = self._models_module(plan)
            elif item.kind == "exceptions":
                content = self._exceptions_module(plan)
            elif item.kind == "config":
                content = self._config_module(plan)
            elif item.kind == "package_init":
                content = self._package_init_module(plan)
            else:
                content = _placeholder_module(item.purpose)
            generated.append(
                GeneratedFile(
                    path=item.path,
                    action="create",
                    content=content,
                    purpose=item.purpose,
                )
            )
        return generated

    def _client_module(self, plan: IntegrationPlan) -> str:
        class_name = _class_name(plan)
        slug = _slug(plan)
        auth = plan.authentication
        retryable_rules = [rule for rule in plan.error_handling.rules if rule.retryable]
        exceptions_module = _import_of_kind(plan, "exceptions")
        models_module = _import_of_kind(plan, "models")
        config_module = _import_of_kind(plan, "config")
        exception_type = plan.error_handling.exception_type
        exception_names = [exception_type] + [
            _exception_class(rule, exception_type) for rule in plan.error_handling.rules
        ]
        model_names = _referenced_models(plan)
        purpose = _purpose_of_kind(plan, "client") or f"封装 {plan.target_api.name} 的 HTTP 调用"
        needs_time = bool(retryable_rules)
        needs_base64 = auth is not None and auth.scheme == "http-basic"

        lines = [
            f'"""{purpose}（由 APIForge 生成）。',
            "",
            "base_url 与认证凭据从环境变量读取（见 config 模块）；不包含任何硬编码凭据。",
            '"""',
            "from __future__ import annotations",
            "",
        ]
        if needs_base64:
            lines.append("import base64")
        if needs_time:
            lines.append("import time")
        lines += [
            "from typing import Any, NoReturn",
            "",
            "import httpx",
            "",
        ]
        if exceptions_module:
            lines.append(f"from {exceptions_module} import (")
            for name in exception_names:
                lines.append(f"    {name},")
            lines.append(")")
        if models_module and model_names:
            lines.append(f"from {models_module} import (")
            for name in model_names:
                lines.append(f"    {name},")
            lines.append(")")
        if config_module:
            parent, module_name = config_module.rsplit(".", 1)
            lines.append(f"from {parent} import {module_name} as config")
        lines += ["", ""]

        lines.append(f"class {class_name}:")
        lines.append(f'    """{plan.target_api.name} 客户端。"""')
        lines.append("")
        if retryable_rules:
            lines += ["    _max_retries = 3", ""]

        init_args = ["self", "*"]
        if config_module:
            init_args.append("base_url: str | None = None")
        else:
            init_args.append(f'base_url: str = "{plan.target_api.base_url or ""}"')
        init_args.extend(_auth_ctor_args(auth))
        init_args.extend(["timeout: float = 10.0", "transport: httpx.BaseTransport | None = None"])
        lines.append(f"    def __init__({', '.join(init_args)}) -> None:")
        lines.append('        """构造客户端。"""')
        if config_module:
            lines.append('        self._base_url = (base_url or config.BASE_URL or "").rstrip("/")')
            lines.append("        if not self._base_url:")
            lines.append(
                '            raise ValueError("缺少 base_url：请通过参数或 '
                f'{slug.upper()}_BASE_URL 环境变量提供")'
            )
        else:
            lines.append('        self._base_url = base_url.rstrip("/")')
        lines.extend(_auth_attr_lines(auth, config_module is not None))
        lines.append("        self._client = httpx.Client(timeout=timeout, transport=transport)")
        lines.append("")

        if auth is not None:
            lines.append("    def _build_headers(self) -> dict[str, str]:")
            lines.append(
                f'        """按 AuthenticationPlan 注入认证凭据（scheme: {auth.scheme}）。"""'
            )
            lines.append("        headers: dict[str, str] = {}")
            lines.extend(_auth_headers_lines(auth))
            lines.append("        return headers")
            lines.append("")

        lines += self._request_lines(retryable_rules)
        if retryable_rules:
            lines += self._retry_helper_lines(retryable_rules)
        lines += self._raise_for_status_lines(plan, auth)
        for planned in plan.endpoints:
            lines.extend(self._endpoint_method_lines(planned, auth))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _request_lines(retryable_rules: list[ErrorHandlingRule]) -> list[str]:
        lines = [
            "    def _request(",
            "        self,",
            "        method: str,",
            "        url: str,",
            "        *,",
            "        params: dict[str, Any] | None = None,",
            "        json: Any = None,",
            "        headers: dict[str, str] | None = None,",
            "        cookies: dict[str, str] | None = None,",
            "    ) -> httpx.Response:",
        ]
        if retryable_rules:
            lines.append(
                '        """发起请求；可重试的错误码按退避策略重试，其余错误映射为异常。"""'
            )
            lines.append("        response = None")
            lines.append("        for attempt in range(self._max_retries + 1):")
            lines.append("            response = self._client.request(")
            lines.append(
                "                method, url, params=params, json=json, "
                "headers=headers, cookies=cookies"
            )
            lines.append("            )")
            lines.append("            if response.is_success:")
            lines.append("                return response")
            lines.append("            if not self._is_retryable(method, response.status_code):")
            lines.append("                break")
            lines.append("            if attempt >= self._max_retries:")
            lines.append("                break")
            lines.append("            time.sleep(self._backoff_delay(response, attempt))")
        else:
            lines.append('        """发起请求并把错误状态码映射为异常。"""')
            lines.append("        response = self._client.request(")
            lines.append(
                "            method, url, params=params, json=json, headers=headers, "
                "cookies=cookies"
            )
            lines.append("        )")
            lines.append("        if response.is_success:")
            lines.append("            return response")
        lines.append("        return self._raise_for_status(response)")
        lines.append("")
        return lines

    @staticmethod
    def _retry_helper_lines(retryable_rules: list[ErrorHandlingRule]) -> list[str]:
        lines = [
            "    def _is_retryable(self, method: str, status_code: int) -> bool:",
            '        """按 ErrorHandlingPlan 判断是否重试：429 一律可重试，5xx 仅幂等方法。"""',
        ]
        specific = [rule.status_code for rule in retryable_rules if rule.status_code.isdigit()]
        has_5xx = any(rule.status_code == "5xx" for rule in retryable_rules)
        if specific:
            codes = "{" + ", ".join(str(code) for code in specific) + "}"
            lines.append(f"        if status_code in {codes}:")
            lines.append("            return True")
        if has_5xx:
            lines.append("        if status_code >= 500:")
            lines.append('            return method not in {"POST", "PATCH"}')
        lines += [
            "        return False",
            "",
            "    def _backoff_delay(self, response: httpx.Response, attempt: int) -> float:",
            '        """优先按 Retry-After 头退避，否则指数退避（2^attempt 秒）。"""',
            '        retry_after = response.headers.get("Retry-After")',
            "        if retry_after is not None and retry_after.isdigit():",
            "            return float(retry_after)",
            "        return 2.0**attempt",
            "",
        ]
        return lines

    @staticmethod
    def _raise_for_status_lines(
        plan: IntegrationPlan, auth: AuthenticationPlan | None
    ) -> list[str]:
        exception_type = plan.error_handling.exception_type
        lines = [
            "    def _raise_for_status(self, response: httpx.Response) -> NoReturn:",
            '        """把错误状态码映射为结构化异常（来自 ErrorHandlingPlan）。"""',
            "        code = response.status_code",
        ]
        auth_hint = ""
        if auth is not None and auth.required_env_vars:
            auth_hint = f"，请检查 {auth.required_env_vars[0]}"
        for rule in plan.error_handling.rules:
            name = _exception_class(rule, exception_type)
            label = ERROR_LABELS.get(rule.category, rule.category)
            if rule.category == "authentication" and auth_hint:
                detail = f"{label}（HTTP {{code}}{auth_hint}）"
            else:
                detail = f"{label}（HTTP {{code}}）"
            if rule.status_code == "5xx":
                lines.append("        if code >= 500:")
            else:
                lines.append(f"        if code == {rule.status_code}:")
            lines.append(f'            raise {name}(f"{detail}: {{response.text}}")')
        lines.append(f'        raise {exception_type}(f"HTTP {{code}}: {{response.text}}")')
        lines.append("")
        return lines

    @staticmethod
    def _endpoint_method_lines(
        planned: PlannedEndpoint, auth: AuthenticationPlan | None
    ) -> list[str]:
        endpoint = planned.endpoint
        name = _endpoint_name(endpoint)
        path_params = [p for p in endpoint.parameters if p.location == "path"]
        query_params = [p for p in endpoint.parameters if p.location == "query"]
        header_params = [p for p in endpoint.parameters if p.location == "header"]
        cookie_params = [p for p in endpoint.parameters if p.location == "cookie"]
        body = endpoint.request_body

        sig = ["self", "*"]
        for param in path_params:
            hint = _type_hint(param.schema_type)
            arg = _py_arg(param.name)
            sig.append(f"{arg}: {hint}" if param.required else f"{arg}: {hint} | None = None")
        for param in query_params + header_params + cookie_params:
            sig.append(f"{_py_arg(param.name)}: {_type_hint(param.schema_type)} | None = None")
        if body is not None:
            if body.schema_ref:
                sig.append(f"payload: {_model_name(body.schema_ref)}")
            else:
                sig.append("payload: dict[str, Any]")

        doc = planned.purpose or endpoint.summary or f"{endpoint.method} {endpoint.path}"
        lines = [
            f"    def {name}({', '.join(sig)}) -> {_return_type(endpoint)}:",
            f'        """{doc}"""',
        ]
        url = f'        url = self._base_url + "{endpoint.path}"'
        for param in path_params:
            url += f'.replace("{{{param.name}}}", str({_py_arg(param.name)}))'
        lines.append(url)

        if query_params:
            lines.append("        params: dict[str, Any] = {}")
            for param in query_params:
                arg = _py_arg(param.name)
                lines.append(f"        if {arg} is not None:")
                lines.append(f'            params["{param.name}"] = {arg}')
        if auth is not None:
            lines.append("        headers = self._build_headers()")
        elif header_params:
            lines.append("        headers: dict[str, str] = {}")
        for param in header_params:
            arg = _py_arg(param.name)
            lines.append(f"        if {arg} is not None:")
            lines.append(f'            headers["{param.name}"] = {arg}')
        if cookie_params:
            lines.append("        cookies: dict[str, str] = {}")
            for param in cookie_params:
                arg = _py_arg(param.name)
                lines.append(f"        if {arg} is not None:")
                lines.append(f'            cookies["{param.name}"] = {arg}')

        call_args: list[str] = []
        if query_params:
            call_args.append("params=params")
        if body is not None:
            if body.schema_ref:
                call_args.append("json=payload.model_dump(exclude_none=True)")
            else:
                call_args.append("json=payload")
        if auth is not None or header_params:
            call_args.append("headers=headers")
        if cookie_params:
            call_args.append("cookies=cookies")
        if call_args:
            lines.append("        response = self._request(")
            lines.append(f'            "{endpoint.method}", url,')
            for arg in call_args:
                lines.append(f"            {arg},")
            lines.append("        )")
        else:
            lines.append(f'        response = self._request("{endpoint.method}", url)')
        lines.append(_return_statement(endpoint))
        lines.append("")
        return lines

    def _models_module(self, plan: IntegrationPlan) -> str:
        names = _referenced_models(plan)
        request_refs = {
            _model_name(p.endpoint.request_body.schema_ref)
            for p in plan.endpoints
            if p.endpoint.request_body is not None and p.endpoint.request_body.schema_ref
        }
        response_refs: set[str] = set()
        for planned in plan.endpoints:
            for response in planned.endpoint.responses:
                if response.schema_ref:
                    response_refs.add(_model_name(response.schema_ref))
                items = _items_ref(response)
                if items:
                    response_refs.add(_model_name(items))
        lines = [
            f'"""{plan.target_api.name} 的数据模型（由 APIForge 生成）。',
            "",
            "字段定义以 OpenAPI components/schemas 为准；当前生成器只生成允许",
            "额外字段的占位模型，字段级模型由 LLMCodeGenerator 阶段补充。",
            '"""',
            "",
        ]
        if names:
            lines += ["from pydantic import BaseModel, ConfigDict", "", ""]
            for name in names:
                role = "模型"
                if name in request_refs and name in response_refs:
                    role = "请求/响应模型"
                elif name in request_refs:
                    role = "请求模型"
                elif name in response_refs:
                    role = "响应模型"
                lines.append(f"class {name}(BaseModel):")
                lines.append(f'    """OpenAPI schema 对应的{role}。"""')
                lines.append('    model_config = ConfigDict(extra="allow")')
                lines.append("")
        else:
            lines.append("# 当前所选端点未引用任何 components schema")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _exceptions_module(plan: IntegrationPlan) -> str:
        exception_type = plan.error_handling.exception_type
        lines = [
            f'"""{plan.target_api.name} 的异常层次（由 APIForge 生成）。',
            "",
            "每个异常对应 ErrorHandlingPlan 中的一条规则；重试行为由客户端实现",
            f"（见 {_class_name(plan)}._is_retryable）。",
            '"""',
            "",
            f"class {exception_type}(Exception):",
            f'    """{plan.target_api.name} 调用错误的基类。"""',
            "",
        ]
        for rule in plan.error_handling.rules:
            name = _exception_class(rule, exception_type)
            label = ERROR_LABELS.get(rule.category, rule.category)
            retry = "可重试" if rule.retryable else "不重试"
            lines.append(f"class {name}({exception_type}):")
            lines.append(f'    """HTTP {rule.status_code}：{label}（{retry}）。"""')
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _config_module(plan: IntegrationPlan) -> str:
        slug = _slug(plan)
        auth = plan.authentication
        lines = [
            f'"""{plan.target_api.name} 配置（由 APIForge 生成）。',
            "",
            "所有配置从环境变量读取；禁止把真实凭据写入代码或提交到版本库。",
            '"""',
            "",
            "import os",
            "",
        ]
        env_base = f"{slug.upper()}_BASE_URL"
        if plan.target_api.base_url:
            lines.append(f'BASE_URL = os.environ.get("{env_base}", "{plan.target_api.base_url}")')
        else:
            lines.append("# OpenAPI 未声明 servers：base_url 必须由使用方显式提供")
            lines.append(f'BASE_URL = os.environ.get("{env_base}")')
        if auth is not None:
            for env_var in auth.required_env_vars:
                var = _auth_config_var(env_var, slug)
                lines.append(f'{var} = os.environ.get("{env_var}")')
            if auth.scheme == "oauth2":
                lines.append("# CLIENT_ID / CLIENT_SECRET 用于 client credentials 流程；")
                lines.append("# access token 的获取与刷新由使用方接入。")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _package_init_module(plan: IntegrationPlan) -> str:
        lines = [
            f'"""{plan.target_api.name} 集成包（由 APIForge 生成）。"""',
            "",
            f"from .{_slug(plan)}_client import {_class_name(plan)}",
            f"from .{_slug(plan)}_exceptions import {plan.error_handling.exception_type}",
            "",
            "__all__ = [",
            f'    "{_class_name(plan)}",',
            f'    "{plan.error_handling.exception_type}",',
            "]",
        ]
        return "\n".join(lines) + "\n"

    # --------------------------------------------------------- files_to_modify

    def _generated_modified(self, plan: IntegrationPlan) -> list[GeneratedFile]:
        return [self._modification(plan, item) for item in plan.files_to_modify]

    @staticmethod
    def _modification(plan: IntegrationPlan, item: FileModification) -> GeneratedFile:
        if item.path.endswith(("pyproject.toml", "requirements.txt")):
            content, point = _manifest_snippet(item)
        elif item.path.endswith("/__init__.py") or item.path == "__init__.py":
            content, point = _init_snippet(plan), "文件末尾（导出区）"
        elif item.path == plan.repository.http_client_module:
            content, point = _transport_snippet(plan), "模块末尾（或 transport/session 构造处）"
        else:
            content, point = _usage_snippet(plan), "模块末尾（或合适的调用位置）"
        return GeneratedFile(
            path=item.path,
            action="modify",
            content=content,
            purpose=item.reason,
            insertion_point=point,
            changes=list(item.changes),
        )

    # ------------------------------------------------------------- dependencies

    @staticmethod
    def _dependency_changes(plan: IntegrationPlan) -> list[DependencyChange]:
        return [
            DependencyChange(name=item.name, version=item.version, action="add", reason=item.reason)
            for item in plan.dependencies
            if not item.already_installed
        ]

    # --------------------------------------------------------- testing strategy

    def _generated_tests(self, plan: IntegrationPlan) -> list[GeneratedFile]:
        files: list[GeneratedFile] = []
        for spec in plan.testing_strategy.unit_tests:
            files.append(
                GeneratedFile(
                    path=spec.path,
                    action="create",
                    content=self._unit_tests(plan, spec),
                    purpose=spec.purpose,
                )
            )
        for spec in plan.testing_strategy.integration_tests:
            files.append(
                GeneratedFile(
                    path=spec.path,
                    action="create",
                    content=self._integration_tests(plan, spec),
                    purpose=spec.purpose,
                )
            )
        for spec in plan.testing_strategy.contract_tests:
            files.append(
                GeneratedFile(
                    path=spec.path,
                    action="create",
                    content=self._contract_tests(plan, spec),
                    purpose=spec.purpose,
                )
            )
        return files

    def _unit_tests(self, plan: IntegrationPlan, spec: TestSpec) -> str:
        class_name = _class_name(plan)
        client_module = _import_of_kind(plan, "client")
        exceptions_module = _import_of_kind(plan, "exceptions")
        models_module = _import_of_kind(plan, "models")
        exception_type = plan.error_handling.exception_type
        rule_classes = [_exception_class(r, exception_type) for r in plan.error_handling.rules]
        model_names = _referenced_models(plan)
        needs_json = any(p.endpoint.request_body is not None for p in plan.endpoints)

        lines = [f'"""{spec.purpose}（由 APIForge 生成）。"""', ""]
        if needs_json:
            lines += ["import json", ""]
        lines += ["import httpx", "", f"from {client_module} import {class_name}", ""]
        if exceptions_module:
            lines.append(f"from {exceptions_module} import (")
            for name in [exception_type] + rule_classes:
                lines.append(f"    {name},")
            lines.append(")")
        if models_module and model_names:
            lines.append(f"from {models_module} import (")
            for name in model_names:
                lines.append(f"    {name},")
            lines.append(")")
        lines += ["", ""]

        lines += [
            "def test_error_code_to_exception_mapping() -> None:",
            '    """错误码到异常的映射"""',
        ]
        for rule_class in rule_classes:
            lines.append(f"    assert issubclass({rule_class}, {exception_type})")
        lines.append("")
        for planned in plan.endpoints:
            lines.extend(self._endpoint_test(planned, class_name))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _endpoint_test(planned: PlannedEndpoint, class_name: str) -> list[str]:
        endpoint = planned.endpoint
        name = _endpoint_name(endpoint)
        call_args, expected_path = _test_call_args(endpoint)
        body = endpoint.request_body

        lines = [
            f"def test_{name}() -> None:",
            f'    """{planned.purpose}"""',
            "",
            "    def handler(request: httpx.Request) -> httpx.Response:",
            f'        assert request.method == "{endpoint.method}"',
            f'        assert request.url.path == "{expected_path}"',
        ]
        for param in endpoint.parameters:
            if param.location == "query":
                lines.append(f'        assert request.url.params["{param.name}"] == "10"')
            elif param.location == "header":
                lines.append(f'        assert request.headers["{param.name}"] == "trace-1"')
        if body is not None:
            lines.append("        body = json.loads(request.content)")
            if body.schema_ref:
                lines.append("        assert body == {}")
            else:
                lines.append('        assert body == {"name": "test"}')
        lines.append("        " + _dummy_response(endpoint))
        lines += [
            "",
            f"    client = {class_name}(",
            '        base_url="https://example.test",',
            "        transport=httpx.MockTransport(handler),",
            "    )",
            _call_line(name, call_args),
        ]
        lines.extend(_result_assert_lines(endpoint, "    "))
        lines.append("")
        return lines

    def _integration_tests(self, plan: IntegrationPlan, spec: TestSpec) -> str:
        class_name = _class_name(plan)
        client_module = _import_of_kind(plan, "client")
        auth = plan.authentication
        retryable_rules = [r for r in plan.error_handling.rules if r.retryable]
        first = plan.endpoints[0]
        call_args, _ = _test_call_args(first.endpoint)
        call_name = _endpoint_name(first.endpoint)
        needs_base64 = auth is not None and auth.scheme == "http-basic"

        lines = [f'"""{spec.purpose}（由 APIForge 生成）。"""', ""]
        if needs_base64:
            lines += ["import base64", ""]
        lines += ["import httpx", ""]
        if retryable_rules:
            lines += ["import pytest", ""]
        lines += [f"from {client_module} import {class_name}", ""]
        if retryable_rules:
            rate_limit = next(
                (
                    _exception_class(r, plan.error_handling.exception_type)
                    for r in retryable_rules
                    if r.category == "rate_limit"
                ),
                _exception_class(retryable_rules[0], plan.error_handling.exception_type),
            )
            exceptions_module = _import_of_kind(plan, "exceptions")
            lines.append(f"from {exceptions_module} import {rate_limit}")
            lines.append("")
        lines += ["", ""]

        lines += [
            "def test_end_to_end_call() -> None:",
            '    """端到端调用"""',
            "",
            "    def handler(request: httpx.Request) -> httpx.Response:",
            f'        assert request.method == "{first.endpoint.method}"',
            f'        assert request.url.path == "{first.endpoint.path}"',
            "        " + _dummy_response(first.endpoint),
            "",
            f"    client = {class_name}(",
            '        base_url="https://example.test",',
            "        transport=httpx.MockTransport(handler),",
            "    )",
            _call_line(call_name, call_args),
        ]
        if auth is not None:
            lines += [
                "",
                "def test_authentication_injection() -> None:",
                '    """认证凭据注入"""',
                "",
                "    def handler(request: httpx.Request) -> httpx.Response:",
            ]
            for assertion in _auth_assert_lines(auth):
                lines.append(f"        {assertion}")
            lines.append("        " + _dummy_response(first.endpoint))
            lines += [
                "",
                f"    client = {class_name}(",
                '        base_url="https://example.test",',
                f"        {_auth_test_args(auth)},",
                "        transport=httpx.MockTransport(handler),",
                "    )",
                _call_line(call_name, call_args),
            ]
        if retryable_rules:
            lines += [
                "",
                "def test_retry_and_backoff() -> None:",
                '    """重试与退避行为"""',
                "",
                "    def handler(request: httpx.Request) -> httpx.Response:",
                '        return httpx.Response(429, headers={"Retry-After": "0"})',
                "",
                f"    client = {class_name}(",
                '        base_url="https://example.test",',
                "        transport=httpx.MockTransport(handler),",
                "    )",
                f"    with pytest.raises({rate_limit}):",
                f"        client.{call_name}()",
            ]
        return "\n".join(lines) + "\n"

    def _contract_tests(self, plan: IntegrationPlan, spec: TestSpec) -> str:
        class_name = _class_name(plan)
        client_module = _import_of_kind(plan, "client")
        models_module = _import_of_kind(plan, "models")
        model_names = _referenced_models(plan)
        lines = [
            f'"""{spec.purpose}（由 APIForge 生成）。"""',
            "",
            "import httpx",
            "",
            f"from {client_module} import {class_name}",
            "",
        ]
        if models_module and model_names:
            lines.append(f"from {models_module} import (")
            for name in model_names:
                lines.append(f"    {name},")
            lines.append(")")
        lines.append("")
        for planned in plan.endpoints:
            endpoint = planned.endpoint
            name = _endpoint_name(endpoint)
            call_args, expected_path = _test_call_args(endpoint)
            lines += [
                f"def test_{name}_response_matches_schema() -> None:",
                f'    """{planned.purpose}"""',
                "",
                "    def handler(request: httpx.Request) -> httpx.Response:",
                f'        assert request.url.path == "{expected_path}"',
                "        " + _dummy_response(endpoint),
                "",
                f"    client = {class_name}(",
                '        base_url="https://example.test",',
                "        transport=httpx.MockTransport(handler),",
                "    )",
                _call_line(name, call_args),
            ]
            lines.extend(_contract_assert_lines(endpoint, "    "))
            lines.append("")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 辅助函数


def _to_snake(name: str) -> str:
    """camelCase / 任意分隔符 → snake_case。"""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return name.strip("_").lower()


def _pascal(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_") if part)


def _slug(plan: IntegrationPlan) -> str:
    """从 client_module 文件名推导 slug（如 demo_petstore_client.py → demo_petstore）。"""
    name = plan.integration_strategy.client_module.rsplit("/", 1)[-1]
    if name.endswith(".py"):
        name = name[: -len(".py")]
    if name.endswith("_client"):
        name = name[: -len("_client")]
    return name


def _class_name(plan: IntegrationPlan) -> str:
    return f"{_pascal(_slug(plan))}Client"


def _file_of_kind(plan: IntegrationPlan, kind: str) -> str | None:
    for item in plan.files_to_create:
        if item.kind == kind:
            return item.path
    return None


def _purpose_of_kind(plan: IntegrationPlan, kind: str) -> str:
    for item in plan.files_to_create:
        if item.kind == kind:
            return item.purpose
    return ""


def _import_of_kind(plan: IntegrationPlan, kind: str) -> str | None:
    path = _file_of_kind(plan, kind)
    return _import_module(path) if path else None


def _import_module(path: str) -> str:
    """文件路径 → 可导入的模块路径；src/lib 容器前缀不属于导入路径。"""
    module = path
    for container in ("src/", "lib/"):
        if module.startswith(container):
            module = module[len(container) :]
            break
    if module.endswith(".py"):
        module = module[: -len(".py")]
    return module.replace("/", ".")


def _model_name(ref: str) -> str:
    """schema $ref 名称 → 合法且稳定的 Python 类名。"""
    return _pascal(_to_snake(ref))


def _type_hint(schema_type: str | None) -> str:
    return SCHEMA_TYPE_HINTS.get(schema_type or "", "Any")


def _py_arg(name: str) -> str:
    return _to_snake(name)


def _exception_class(rule: ErrorHandlingRule, exception_type: str) -> str:
    """由规则类别推导异常类名：DemoPetstoreAPIError + rate_limit → DemoPetstoreRateLimitError。

    server_error 的 Pascal 形式已以 "Error" 结尾（ServerError），必须先去重，
    否则会生成 DemoPetstoreServerErrorError。
    """
    base = exception_type
    if "APIError" in base:
        base = base[: base.index("APIError")]
    elif base.endswith("Error"):
        base = base[: -len("Error")]
    suffix = _pascal(rule.category)
    if suffix.endswith("Error"):
        suffix = suffix[: -len("Error")]
    return f"{base}{suffix}Error"


def _endpoint_name(endpoint: APIEndpoint) -> str:
    """端点 → 方法名：优先 operationId 的 snake_case。"""
    if endpoint.operation_id:
        return _to_snake(endpoint.operation_id)
    token = re.sub(r"[^0-9a-zA-Z]+", "_", endpoint.path.replace("{", "").replace("}", ""))
    return f"{endpoint.method.lower()}_{token.strip('_').lower()}"


def _items_ref(response: APIResponse) -> str | None:
    """从数组响应的原始 schema 中取出 items.$ref 名称（plan 已携带，无需重新解析 OpenAPI）。"""
    schema = response.json_schema or {}
    items = schema.get("items")
    if isinstance(items, dict):
        ref = items.get("$ref")
        if isinstance(ref, str) and ref:
            return ref.rsplit("/", 1)[-1]
    return None


def _referenced_models(plan: IntegrationPlan) -> list[str]:
    """端点引用的全部 schema 名称（排序去重，保证输出稳定）。"""
    names: set[str] = set()
    for planned in plan.endpoints:
        endpoint = planned.endpoint
        if endpoint.request_body is not None and endpoint.request_body.schema_ref:
            names.add(_model_name(endpoint.request_body.schema_ref))
        for response in endpoint.responses:
            if response.schema_ref:
                names.add(_model_name(response.schema_ref))
            items = _items_ref(response)
            if items:
                names.add(_model_name(items))
    return sorted(names)


def _success_response(endpoint: APIEndpoint) -> APIResponse | None:
    return next((r for r in endpoint.responses if r.status_code.startswith("2")), None)


def _return_type(endpoint: APIEndpoint) -> str:
    success = _success_response(endpoint)
    if success is None:
        return "Any"
    if success.schema_ref:
        return _model_name(success.schema_ref)
    items = _items_ref(success)
    if success.schema_type == "array" and items:
        return f"list[{_model_name(items)}]"
    if success.content_type is None:
        return "None"
    return "Any"


def _return_statement(endpoint: APIEndpoint) -> str:
    success = _success_response(endpoint)
    if success is None:
        return "        return response.json()"
    if success.schema_ref:
        return f"        return {_model_name(success.schema_ref)}.model_validate(response.json())"
    items = _items_ref(success)
    if success.schema_type == "array" and items:
        name = _model_name(items)
        return f"        return [{name}.model_validate(item) for item in response.json()]"
    if success.content_type is None:
        return "        return None"
    return "        return response.json()"


def _auth_config_var(env_var: str, slug: str) -> str:
    """环境变量名 → config 模块中的变量名（去掉 <SLUG>_ 前缀）。"""
    prefix = f"{slug.upper()}_"
    if env_var.startswith(prefix):
        return env_var[len(prefix) :]
    return env_var.rsplit("_", 1)[-1]


def _auth_ctor_args(auth: AuthenticationPlan | None) -> list[str]:
    if auth is None:
        return []
    if auth.scheme == "apiKey":
        return ["api_key: str | None = None"]
    if auth.scheme == "http-basic":
        return ["username: str | None = None", "password: str | None = None"]
    if auth.scheme == "http-bearer":
        return ["token: str | None = None"]
    if auth.scheme == "oauth2":
        return ["access_token: str | None = None"]
    return ["credential: str | None = None"]


def _auth_attr_lines(auth: AuthenticationPlan | None, has_config: bool) -> list[str]:
    if auth is None:
        return []
    if auth.scheme == "apiKey":
        if has_config:
            return ["        self._api_key = api_key if api_key is not None else config.API_KEY"]
        return ["        self._api_key = api_key"]
    if auth.scheme == "http-bearer":
        if has_config:
            return ["        self._token = token if token is not None else config.TOKEN"]
        return ["        self._token = token"]
    if auth.scheme == "http-basic":
        if has_config:
            return [
                "        self._username = username if username is not None else config.USERNAME",
                "        self._password = password if password is not None else config.PASSWORD",
            ]
        return ["        self._username = username", "        self._password = password"]
    if auth.scheme == "oauth2":
        return [
            "        # access_token 由使用方按 client credentials 流程获取后传入",
            "        self._access_token = access_token",
        ]
    return ["        self._credential = credential"]


def _auth_headers_lines(auth: AuthenticationPlan) -> list[str]:
    if auth.scheme == "apiKey":
        header = auth.header_name or "X-API-Key"
        return [
            "        if self._api_key is not None:",
            f'            headers["{header}"] = self._api_key',
        ]
    if auth.scheme == "http-bearer":
        return [
            "        if self._token is not None:",
            '            headers["Authorization"] = f"Bearer {self._token}"',
        ]
    if auth.scheme == "http-basic":
        return [
            "        if self._username is not None and self._password is not None:",
            '            credentials = f"{self._username}:{self._password}".encode("utf-8")',
            '            headers["Authorization"] = "Basic " + '
            'base64.b64encode(credentials).decode("ascii")',
        ]
    if auth.scheme == "oauth2":
        return [
            "        if self._access_token is not None:",
            '            headers["Authorization"] = f"Bearer {self._access_token}"',
        ]
    return [
        "        if self._credential is not None:",
        '            headers["Authorization"] = f"Bearer {self._credential}"',
    ]


def _auth_test_args(auth: AuthenticationPlan) -> str:
    if auth.scheme == "apiKey":
        return 'api_key="test-key"'
    if auth.scheme == "http-basic":
        return 'username="test-user", password="test-pass"'
    if auth.scheme == "oauth2":
        return 'access_token="test-token"'
    return 'token="test-token"'


def _auth_assert_lines(auth: AuthenticationPlan) -> list[str]:
    if auth.scheme == "apiKey":
        header = auth.header_name or "X-API-Key"
        return [f'assert request.headers.get("{header}") == "test-key"']
    if auth.scheme == "http-basic":
        return [
            'expected = "Basic " + base64.b64encode(b"test-user:test-pass").decode("ascii")',
            'assert request.headers.get("Authorization") == expected',
        ]
    return ['assert request.headers.get("Authorization") == "Bearer test-token"']


def _test_call_args(endpoint: APIEndpoint) -> tuple[list[str], str]:
    """生成测试中对端点方法的调用参数，并返回参数替换后的期望路径。"""
    args: list[str] = []
    expected_path = endpoint.path
    for index, param in enumerate(endpoint.parameters):
        if param.location == "path":
            value = f"test-{index + 1}"
            args.append(f"{_py_arg(param.name)}={value!r}")
            expected_path = expected_path.replace("{" + param.name + "}", value)
        elif param.location == "query":
            args.append(f"{_py_arg(param.name)}=10")
        elif param.location == "header":
            args.append(f'{_py_arg(param.name)}="trace-1"')
    body = endpoint.request_body
    if body is not None:
        if body.schema_ref:
            args.append(f"payload={_model_name(body.schema_ref)}()")
        else:
            args.append('payload={"name": "test"}')
    return args, expected_path


def _call_line(name: str, call_args: list[str]) -> str:
    if call_args:
        return f"    result = client.{name}({', '.join(call_args)})"
    return f"    result = client.{name}()"


def _dummy_response(endpoint: APIEndpoint) -> str:
    """handler 中返回的代表性响应（不依赖真实网络）。"""
    success = _success_response(endpoint)
    if success is None:
        return "return httpx.Response(200, json={})"
    if success.content_type is None:
        return "return httpx.Response(204)"
    if success.schema_type == "array":
        return 'return httpx.Response(200, json=[{"id": 1}])'
    if success.schema_ref:
        return 'return httpx.Response(200, json={"id": 1})'
    return "return httpx.Response(200, json={})"


def _result_assert_lines(endpoint: APIEndpoint, indent: str) -> list[str]:
    success = _success_response(endpoint)
    if success is not None and success.content_type is None:
        return [f"{indent}assert result is None"]
    if success is None:
        return [f"{indent}assert result is not None"]
    if success.schema_ref:
        return [f"{indent}assert result.id == 1"]
    if success.schema_type == "array":
        if _items_ref(success):
            return [
                f"{indent}assert len(result) == 1",
                f"{indent}assert result[0].id == 1",
            ]
        return [f"{indent}assert isinstance(result, list)"]
    return [f"{indent}assert result is not None"]


def _contract_assert_lines(endpoint: APIEndpoint, indent: str) -> list[str]:
    success = _success_response(endpoint)
    if success is not None and success.content_type is None:
        return [f"{indent}assert result is None"]
    if success is None:
        return [f"{indent}assert result is not None"]
    if success.schema_ref:
        return [f"{indent}assert isinstance(result, {_model_name(success.schema_ref)})"]
    if success.schema_type == "array":
        items = _items_ref(success)
        if items:
            return [
                f"{indent}assert isinstance(result, list)",
                f"{indent}assert isinstance(result[0], {_model_name(items)})",
            ]
        return [f"{indent}assert isinstance(result, list)"]
    return [f"{indent}assert result is not None"]


def _manifest_snippet(item: FileModification) -> tuple[str, str]:
    """依赖清单的修改片段；changes 直接来自 plan.files_to_modify。"""
    if item.path.endswith("pyproject.toml"):
        content = "\n".join(f'"{entry}",' for entry in item.changes) + "\n"
        return content, "[project] 的 dependencies 列表内"
    return "\n".join(item.changes) + "\n", "文件末尾（每行一个依赖）"


def _init_snippet(plan: IntegrationPlan) -> str:
    lines = [
        f"from .{_slug(plan)}_client import {_class_name(plan)}",
        f"from .{_slug(plan)}_exceptions import {plan.error_handling.exception_type}",
    ]
    return "\n".join(lines) + "\n"


def _usage_snippet(plan: IntegrationPlan) -> str:
    module = _import_of_kind(plan, "client") or ""
    lines = [
        f"from {module} import {_class_name(plan)}",
        "",
        f"client = {_class_name(plan)}()  # base_url 与凭据从环境变量读取",
    ]
    return "\n".join(lines) + "\n"


def _transport_snippet(plan: IntegrationPlan) -> str:
    return (
        "# APIForge 集成提示：如需共享本模块的 httpx 配置，可将构造好的\n"
        f"# httpx.Client 实例通过 transport 参数传入 {_class_name(plan)}。\n"
    )


def _placeholder_module(purpose: str) -> str:
    return f'"""{purpose}（由 APIForge 生成）。"""\n\n# 该文件由 Planner 计划，具体内容待补充。\n'


def _is_advisory_modification(item: FileModification) -> bool:
    manifest = item.path.endswith(("pyproject.toml", "requirements.txt"))
    init = item.path.endswith("/__init__.py") or item.path == "__init__.py"
    return not manifest and not init


def _summary(
    plan: IntegrationPlan,
    files: list[GeneratedFile],
    dependency_changes: list[DependencyChange],
) -> str:
    created = sum(1 for item in files if item.action == "create")
    modified = sum(1 for item in files if item.action == "modify")
    test_count = (
        len(plan.testing_strategy.unit_tests)
        + len(plan.testing_strategy.integration_tests)
        + len(plan.testing_strategy.contract_tests)
    )
    return (
        f"为 {plan.target_api.name} v{plan.target_api.version} 生成集成代码："
        f"新建 {created - test_count} 个源文件、{test_count} 个测试文件，"
        f"修改 {modified} 个既有文件，新增 {len(dependency_changes)} 项依赖，"
        f"覆盖 {len(plan.endpoints)} 个端点"
    )


def _warnings(plan: IntegrationPlan, dependency_changes: list[DependencyChange]) -> list[str]:
    warnings: list[str] = []
    if plan.target_api.base_url is None:
        warnings.append(
            f"OpenAPI 未声明 servers：生成的 config 依赖 {_slug(plan).upper()}_BASE_URL 环境变量"
        )
    if len(plan.endpoints) < plan.target_api.total_endpoints:
        warnings.append(
            f"本次仅覆盖 {len(plan.endpoints)}/{plan.target_api.total_endpoints} 个端点，"
            "其余端点未生成方法"
        )
    if dependency_changes:
        names = ", ".join(f"{item.name}{item.version or ''}" for item in dependency_changes)
        warnings.append(f"新增依赖需要在目标项目中安装：{names}")
        has_manifest = any(
            item.path.endswith(("pyproject.toml", "requirements.txt"))
            for item in plan.files_to_modify
        )
        if not has_manifest:
            warnings.append(
                "未找到依赖清单文件（pyproject.toml / requirements.txt），依赖变更需手动应用"
            )
    if plan.authentication is not None:
        warnings.append("认证凭据必须通过环境变量注入，禁止把真实 secret 写入代码或提交到版本库")
    advisory = [item.path for item in plan.files_to_modify if _is_advisory_modification(item)]
    if advisory:
        warnings.append(
            "对既有模块的修改为建议性片段（" + ", ".join(advisory) + "），应用前需要人工确认"
        )
    return warnings
