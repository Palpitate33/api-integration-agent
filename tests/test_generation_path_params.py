"""Path parameter 生成契约的回归测试（P0）。

背景
----
带 path parameter 的端点上，生成物曾经自相矛盾：

    OpenAPI path template       /users/{user_id}
    Generated client method     def get_user(self) -> User:      ← 没有参数
    URL                         "/users/{user_id}"               ← 没被替换
    Generated test              assert request.url.path == "/users/{user_id}"
                                client.get_user()                ← 参数被丢掉

结果是真实 Pipeline 跑出 tests_failed：`test_end_to_end_call` 断言失败、
`test_retry_and_backoff` 抛 TypeError（调用一个需要 path 参数的方法却没传参）。

本文件把这四层之间的契约钉死在 AST 上，而不是靠字符串匹配：调用点传了哪些参数、
客户端方法收哪些参数、URL 里替换了哪些占位符，三者必须逐项对齐；最后再真的把
生成的测试跑一遍。

覆盖：
    Test A  单个 path parameter
    Test B  多个 path parameters
    Test C  path + query
    Test D  path + body
    Test E  真实 Pipeline：GeneratedArtifacts → TestRunner → 全绿
    以及    path 模板声明缺失（parameters 里没写）时的兜底
"""

import ast
import re
from pathlib import Path

import pytest

from integration_agent.agent import IntegrationPlan, plan_integration
from integration_agent.api import APIEndpoint, parse_openapi, parse_openapi_text
from integration_agent.generation import GeneratedArtifacts, GeneratedFile, generate_code
from integration_agent.repository import scan_repository
from integration_agent.validation import DeterministicTestRunner

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
USERS_PATH = EXAMPLES / "openapi" / "users_api.yaml"
DEMO_PROJECT_PATH = EXAMPLES / "demo_project"
CLIENT_CLASS = "DemoUsersClient"
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


# ------------------------------------------------------------------ 生成辅助


@pytest.fixture(scope="module")
def plan() -> IntegrationPlan:
    """真实解析 spec + 真实扫描项目 → IntegrationPlan。"""
    return plan_integration(parse_openapi(USERS_PATH), scan_repository(DEMO_PROJECT_PATH))


@pytest.fixture(scope="module")
def artifacts(plan: IntegrationPlan) -> GeneratedArtifacts:
    return generate_code(plan)


def _endpoint(plan: IntegrationPlan, operation_id: str) -> APIEndpoint:
    return next(
        item.endpoint for item in plan.endpoints if item.endpoint.operation_id == operation_id
    )


def _created(artifacts: GeneratedArtifacts, *, suffix: str, under_tests: bool) -> GeneratedFile:
    return next(
        item
        for item in artifacts.files
        if item.action == "create"
        and item.path.endswith(suffix)
        and item.path.startswith("tests/") is under_tests
    )


def _client_source(artifacts: GeneratedArtifacts) -> str:
    return _created(artifacts, suffix="_client.py", under_tests=False).content


def _unit_test_source(artifacts: GeneratedArtifacts) -> str:
    return _created(artifacts, suffix="_client.py", under_tests=True).content


def _integration_test_source(artifacts: GeneratedArtifacts) -> str:
    return _created(artifacts, suffix="_integration.py", under_tests=True).content


# -------------------------------------------------------------------- AST 辅助


def _client_class(source: str) -> ast.ClassDef:
    """生成模块里的客户端类。类名由 API 名称推导，所以按"有且只有一个类"来取，
    而不是把某个具体名字写死在测试里。"""
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    assert len(classes) == 1, "生成的客户端模块里应当只有一个类"
    return classes[0]


def _method(source: str, method_name: str) -> ast.FunctionDef:
    """客户端类里的某个方法节点；方法不存在时直接失败（StopIteration → 测试报错）。"""
    return next(
        node
        for node in _client_class(source).body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _signature(source: str, method_name: str) -> dict[str, str | None]:
    """方法的关键字参数 → 类型提示。

    生成器用的是 keyword-only 签名（`def f(self, *, user_id: str)`），所以参数一定
    以关键字形式传递——这正是"调用点参数名 == 签名参数名"可以直接比较的前提。
    """
    method = _method(source, method_name)
    return {
        arg.arg: (ast.unparse(arg.annotation) if arg.annotation is not None else None)
        for arg in method.args.kwonlyargs
    }


def _required(source: str, method_name: str) -> set[str]:
    """没有默认值的关键字参数（调用时必须传）。"""
    method = _method(source, method_name)
    return {
        arg.arg
        for arg, default in zip(method.args.kwonlyargs, method.args.kw_defaults)
        if default is None
    }


def _substituted(source: str, method_name: str) -> set[str]:
    """方法体里 `.replace("{name}", ...)` 替换掉的 path 占位符。"""
    found: set[str] = set()
    for node in ast.walk(_method(source, method_name)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "replace" or not node.args:
            continue
        target = node.args[0]
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            if PLACEHOLDER.fullmatch(target.value):
                found.add(target.value[1:-1])
    return found


def _test_functions(source: str) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


def _client_calls(source: str) -> dict[str, ast.Call]:
    """测试函数名 → 该函数里对 `client.<method>(...)` 的调用。"""
    calls: dict[str, ast.Call] = {}
    for name, node in _test_functions(source).items():
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "client"
            ):
                calls[name] = inner
                break
    return calls


def _keyword_names(call: ast.Call) -> set[str]:
    return {keyword.arg for keyword in call.keywords if keyword.arg is not None}


def _path_assertions(node: ast.FunctionDef) -> list[str]:
    """测试函数里 `assert request.url.path == "..."` 断言的路径字面量。"""
    found: list[str] = []
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Compare) or len(inner.comparators) != 1:
            continue
        comparator = inner.comparators[0]
        if ast.unparse(inner.left) == "request.url.path" and isinstance(comparator, ast.Constant):
            if isinstance(comparator.value, str):
                found.append(comparator.value)
    return found


def _query_assertions(node: ast.FunctionDef) -> set[str]:
    """测试函数里 `assert request.url.params["x"] == ...` 断言的查询参数名。"""
    found: set[str] = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Compare) or not isinstance(inner.left, ast.Subscript):
            continue
        if ast.unparse(inner.left.value) != "request.url.params":
            continue
        key = inner.left.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            found.add(key.value)
    return found


def _method_name(endpoint: APIEndpoint) -> str:
    """operationId → 生成的方法名（与该 fixture 的命名规则一致；名字不符会直接报错）。"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", endpoint.operation_id or "").lower()


def _template_parameters(endpoint: APIEndpoint) -> set[str]:
    return set(PLACEHOLDER.findall(endpoint.path))


# ---------------------------------------------------- Test A：单个 path parameter


def test_a_single_path_parameter_reaches_signature_and_call(artifacts) -> None:
    """GET /users/{user_id}：签名里有 user_id，测试里也按关键字传了 user_id。"""
    client = _client_source(artifacts)
    unit = _unit_test_source(artifacts)

    assert _client_class(client).name == CLIENT_CLASS
    assert _signature(client, "get_user") == {"user_id": "str"}
    assert _required(client, "get_user") == {"user_id"}
    assert _substituted(client, "get_user") == {"user_id"}

    call = _client_calls(unit)["test_get_user"]
    assert call.func.attr == "get_user"
    assert _keyword_names(call) == {"user_id"}


def test_a_expected_path_is_the_substituted_path(artifacts) -> None:
    """测试断言的期望路径必须是替换后的路径，而不是 `{user_id}` 模板原文。"""
    node = _test_functions(_unit_test_source(artifacts))["test_get_user"]

    # 单元测试与契约测试各自断言一次；integration 测试断言的是第一个端点
    unit_paths = _path_assertions(node)
    assert unit_paths == ["/users/test-1"]
    assert "{" not in unit_paths[0]


def test_integration_test_asserts_the_substituted_path_too(artifacts) -> None:
    """端到端测试曾经拿模板原文 `{user_id}` 去比真实 URL——这是 P0 的断言失败点。"""
    node = _test_functions(_integration_test_source(artifacts))["test_end_to_end_call"]

    assert _path_assertions(node) == ["/users/test-1"]


# ------------------------------------------------- Test B：多个 path parameters


def test_b_multiple_path_parameters(artifacts) -> None:
    """GET /users/{user_id}/posts/{post_id}：两个参数都在，且都被替换。"""
    client = _client_source(artifacts)
    unit = _unit_test_source(artifacts)

    assert _signature(client, "get_user_post") == {"user_id": "str", "post_id": "int"}
    assert _substituted(client, "get_user_post") == {"user_id", "post_id"}

    call = _client_calls(unit)["test_get_user_post"]
    assert _keyword_names(call) == {"user_id", "post_id"}


def test_b_declared_integer_path_parameter_gets_a_typed_value(artifacts) -> None:
    """`post_id` 声明为 integer：测试既传对名字，也传对类型。"""
    node = _test_functions(_unit_test_source(artifacts))["test_get_user_post"]

    assert _path_assertions(node) == ["/users/test-1/posts/2"]


# ----------------------------------------------------------- Test C：path + query


def test_c_path_and_query_are_not_confused(artifacts) -> None:
    """GET /users/{user_id}/posts?limit=10：user_id 走路径、limit 走查询串。"""
    client = _client_source(artifacts)
    unit = _unit_test_source(artifacts)

    # 签名：path 必填、query 可选——两者在同一个签名里各就各位
    assert _signature(client, "list_user_posts") == {"user_id": "str", "limit": "int | None"}
    assert _required(client, "list_user_posts") == {"user_id"}
    # 只有 user_id 出现在 URL 替换里；limit 绝不能进路径
    assert _substituted(client, "list_user_posts") == {"user_id"}
    assert 'params["limit"] = limit' in client

    call = _client_calls(unit)["test_list_user_posts"]
    assert _keyword_names(call) == {"user_id", "limit"}

    # 查询参数按查询串断言，path 参数不进查询串
    node = _test_functions(unit)["test_list_user_posts"]
    assert _query_assertions(node) == {"limit"}
    assert _path_assertions(node) == ["/users/test-1/posts"]


# ------------------------------------------------------------ Test D：path + body


def test_d_path_and_body(artifacts) -> None:
    """POST /users/{user_id}：path 参数与 request body 同时生成。"""
    client = _client_source(artifacts)
    unit = _unit_test_source(artifacts)

    assert _signature(client, "update_user") == {"user_id": "str", "payload": "User"}
    assert _substituted(client, "update_user") == {"user_id"}

    call = _client_calls(unit)["test_update_user"]
    assert _keyword_names(call) == {"user_id", "payload"}


def test_retry_test_passes_the_same_arguments_as_the_other_call_sites(artifacts) -> None:
    """重试测试曾经调用 `client.get_user()`——漏传必填参数，抛的是 TypeError。"""
    node = _test_functions(_integration_test_source(artifacts))["test_retry_and_backoff"]
    calls = [
        inner
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and isinstance(inner.func.value, ast.Name)
        and inner.func.value.id == "client"
    ]

    assert len(calls) == 1
    assert calls[0].func.attr == "get_user"
    assert _keyword_names(calls[0]) == {"user_id"}


# ------------------------------------------ 四层契约的通用不变式（覆盖全部端点）


def test_every_endpoint_keeps_the_call_contract(plan, artifacts) -> None:
    """对**每个**端点与每个测试调用点：参数名必须与客户端签名逐项一致。"""
    client = _client_source(artifacts)
    checked = 0

    for item in plan.endpoints:
        endpoint = item.endpoint
        name = _method_name(endpoint)
        signature = set(_signature(client, name))
        required = _required(client, name)

        # 模板里的每个占位符都必须被替换，替换用的名字也必须真的是一个参数
        assert _substituted(client, name) == _template_parameters(endpoint)
        assert _template_parameters(endpoint) <= signature

        for source in (_unit_test_source(artifacts), _integration_test_source(artifacts)):
            for test_name, call in _client_calls(source).items():
                if call.func.attr != name:
                    continue
                passed = _keyword_names(call)
                assert passed <= signature, f"{test_name} 传了签名里没有的参数"
                assert required <= passed, f"{test_name} 漏传了必填参数 {required - passed}"
                checked += 1

    assert checked >= len(plan.endpoints)


def test_generated_modules_are_valid_python(artifacts) -> None:
    for item in artifacts.files:
        if item.action == "create" and item.path.endswith(".py") and item.content:
            compile(item.content, item.path, "exec")


# ----------------------------------------------- path 参数未声明时的兜底


UNDECLARED_SPEC = """
openapi: 3.0.3
info: {title: Undeclared Params API, version: "1.0"}
servers: [{url: https://items.example.com}]
paths:
  /items/{id}:
    get:
      operationId: getItem
      responses:
        "200": {description: ok}
  /items/{item_id}/tags/{tag}:
    get:
      operationId: getItemTag
      responses:
        "200": {description: ok}
"""


def test_path_parameters_missing_from_parameters_are_derived_from_template() -> None:
    """模板里写了 `{id}`、parameters 里没声明时按模板补齐（否则 URL 不会被替换）。"""
    api = parse_openapi_text(UNDECLARED_SPEC)
    by_operation = {endpoint.operation_id: endpoint for endpoint in api.endpoints}

    single = {param.name: param for param in by_operation["getItem"].parameters}
    assert set(single) == {"id"}
    assert single["id"].location == "path"
    assert single["id"].required is True

    multiple = {param.name: param for param in by_operation["getItemTag"].parameters}
    assert set(multiple) == {"item_id", "tag"}


def test_declared_path_parameters_are_not_duplicated() -> None:
    """已声明的 path 参数保持原样，不会被模板兜底再补一份。"""
    api = parse_openapi(USERS_PATH)
    plan = plan_integration(api, scan_repository(DEMO_PROJECT_PATH))
    endpoint = _endpoint(plan, "listUserPosts")

    assert [p.name for p in endpoint.parameters if p.location == "path"] == ["user_id"]


def test_undeclared_path_parameter_is_generated_consistently() -> None:
    """兜底补出来的参数同样要贯穿签名、URL 替换与测试调用。"""
    api = parse_openapi_text(UNDECLARED_SPEC)
    artifacts = generate_code(plan_integration(api, scan_repository(DEMO_PROJECT_PATH)))
    client = _created(artifacts, suffix="_client.py", under_tests=False).content

    assert _signature(client, "get_item") == {"id": "str"}
    assert _substituted(client, "get_item") == {"id"}
    assert _signature(client, "get_item_tag") == {"item_id": "str", "tag": "str"}
    assert _substituted(client, "get_item_tag") == {"item_id", "tag"}


# --------------------------------------- Test E：真实 Pipeline（真跑 pytest）


@pytest.fixture(scope="module")
def pipeline_run(artifacts: GeneratedArtifacts):
    """GeneratedArtifacts → TestRunner 的真实路径（真的起 pytest 子进程）。"""
    return DeterministicTestRunner().run(artifacts)


def test_e_real_pipeline_passes_on_a_path_parameter_fixture(pipeline_run) -> None:
    assert pipeline_run.status == "passed", [
        (detail.test_name, detail.message) for detail in pipeline_run.failure_details
    ]
    assert pipeline_run.failed == 0
    assert pipeline_run.errors == 0
    assert pipeline_run.passed > 0


def test_e_the_fixture_really_exercises_path_parameters(plan) -> None:
    """防止 fixture 被改成"其实没有 path 参数"之后测试仍然全绿。"""
    with_templates = [item.endpoint.path for item in plan.endpoints if "{" in item.endpoint.path]

    assert with_templates == [
        "/users/{user_id}",  # getUser
        "/users/{user_id}",  # updateUser
        "/users/{user_id}/posts",
        "/users/{user_id}/posts/{post_id}",
    ]
    # 端到端测试断言的就是第一个端点
    assert plan.endpoints[0].endpoint.path == "/users/{user_id}"
    assert plan.endpoints[0].endpoint.method == "GET"
