"""P0-1 / P1-1 安全回归：OpenAPI 里的普通文本不能变成生成代码里的语句。

威胁模型
--------
Code Generator 必须把 OpenAPI 文本写进 Python 源码（title / summary / description /
path / operationId / 参数名 / servers.url / 认证头名）。朴素的拼接失败方式**不是**
"字符串里有坏字符"，而是**语法结构被改写**：

    f'        \"\"\"{doc}\"\"\"'
    doc = '\"\"\"\\nimport pathlib\\npathlib.Path("pwned").write_text("x")\\n\"\"\"'

生成出来仍是**语法完全合法**的一段 Python，而它确实会执行。所以黑名单在这里是无效
防线：它挡的是字符串，挡不住语法；把 ``os.system`` 换成 ``subprocess`` / ``__import__``
/ 十六进制拼串，每一条都能被绕过，而攻击者本来就可以随便换。

本文件因此**刻意不去断言"文本里没有 os.system"**——那是在测黑名单。这里要证明的是
更强的性质：不管文本长什么样，生成代码的**结构与语义**都不变，文本只是数据。

具体做法（都是"语义级"断言，换一种注入写法同样会被抓到）：
    1. 生成物仍然产出，且每个 .py 都 `compile()` 得过；
    2. 恶意文本的**语句**没有进入生成模块的 AST——用 import 白名单证明
       （payload 里写着 `import pathlib`，白名单里没有它就说明它没变成真 import）；
    3. 真正执行生成的模块，payload 要求的 marker 文件**没有被创建**；
    4. 文本仍然逐字保留（docstring / 字符串字面量的值 === 原始 payload）——
       安全性不是靠丢弃信息换来的；
    5. 完整 pipeline（含 TestRunner 在子进程里 import 生成的包）跑完，marker 仍然不存在。

P1-1 也在这里：无参数、无 body 的端点不能生成 `def health_check(self, *) -> None:`。
"""

import ast
import importlib
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from integration_agent.agent import FileModification, IntegrationPlan, plan_integration
from integration_agent.api import parse_openapi_text
from integration_agent.generation import GeneratedArtifacts, generate_code
from integration_agent.generation.code_generator import _manifest_snippet
from integration_agent.generation.rendering import docstring, string_literal
from integration_agent.repository import scan_repository
from integration_agent.validation import run_tests

# 生成代码里**允许**出现的 import。文本若能改变代码语义，就会在这里露出来。
ALLOWED_GENERATED_IMPORTS = {
    "__future__",
    "base64",
    "httpx",
    "json",
    "os",
    "pytest",
    "time",
    "typing",
    "pydantic",
}


@pytest.fixture
def short_root() -> Iterator[Path]:
    """短路径临时目录。

    payload 会变成**文件名**的一部分（它进了 API title，title 决定模块名），
    而 Windows 的路径上限是 260 字符：用 pytest 自带的 tmp_path 会让生成的文件名
    直接超过上限，落盘时 FileNotFoundError。这是测试环境的限制，不是被测代码的
    问题，所以这里自己拿一个短根目录。
    """
    path = Path(tempfile.mkdtemp(prefix="af-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _docstring_payload(marker: Path) -> str:
    """一段"闭合三引号 + 换行 + 真语句"的 payload：朴素拼接下它会变成可执行代码。"""
    return (
        '"""\n'
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('pwned', encoding='utf-8')\n"
        '"""'
    )


def _inline_payload(marker: Path) -> str:
    """一段"闭合双引号 + 换行 + 真语句"的 payload：打进字符串字面量位置的版本。"""
    return (
        '"\n'
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('pwned', encoding='utf-8')\n"
        '"\n'
    )


def _malicious_spec(marker: Path) -> str:
    """把 payload 撒进**所有**已知的文本插入点，而不是只塞 summary 一处。"""
    doc = _docstring_payload(marker)
    inline = _inline_payload(marker)
    return json.dumps(
        {
            "openapi": "3.0.3",
            "info": {"title": doc, "version": "1.0" + inline},
            "servers": [{"url": "https://evil.test/" + inline}],
            "paths": {
                # path 自身、operationId、summary、description、参数名、响应描述
                "/pets/{petId}" + inline: {
                    "get": {
                        "operationId": "getPet" + "\n" + "import pathlib\n",
                        "summary": doc,
                        "description": doc,
                        "parameters": [
                            {
                                "name": "petId" + inline,
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {"name": inline, "in": "query", "schema": {"type": "string"}},
                            {
                                "name": "X-Trace" + inline,
                                "in": "header",
                                "schema": {"type": "string"},
                            },
                            {"name": inline, "in": "cookie", "schema": {"type": "string"}},
                        ],
                        "responses": {"200": {"description": doc}},
                    }
                },
                "/health": {
                    "get": {
                        "operationId": "healthCheck",
                        "responses": {"200": {"description": "ok"}},
                    }
                },
            },
            "components": {
                "securitySchemes": {
                    "key": {"type": "apiKey", "in": "header", "name": "X-Key" + inline}
                }
            },
            "security": [{"key": []}],
        }
    )


def _healthy_spec() -> str:
    """只有无参数端点的 spec（P1-1：`*` 不能孤立出现）。"""
    return json.dumps(
        {
            "openapi": "3.0.3",
            "info": {"title": "Health API", "version": "1.0"},
            "servers": [{"url": "https://health.test"}],
            "paths": {
                "/health": {
                    "get": {
                        "operationId": "healthCheck",
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        }
    )


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "target-project"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )
    package = root / "app"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text('"""App."""\n', encoding="utf-8")
    return root


def _structure_spec(marker: Path | None) -> str:
    """与 `_malicious_spec` 同构、但只有描述文本不同的 spec（结构对比用）。

    operationId / path / 参数名全部固定，所以两份 spec 生成的**标识符完全一致**，
    差别只在字符串常量的值上。
    """
    doc = "PLACEHOLDER" if marker is None else _docstring_payload(marker)
    return json.dumps(
        {
            "openapi": "3.0.3",
            "info": {"title": "Pets API", "version": "1.0"},
            "servers": [{"url": "https://pets.test"}],
            "paths": {
                "/pets/{petId}": {
                    "get": {
                        "operationId": "getPet",
                        "summary": doc,
                        "description": doc,
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {"200": {"description": doc}},
                    }
                }
            },
        }
    )


def _build(tmp_path: Path, spec_text: str) -> tuple[IntegrationPlan, GeneratedArtifacts]:
    plan = plan_integration(parse_openapi_text(spec_text), scan_repository(_project(tmp_path)))
    return plan, generate_code(plan)


def _client_source(tmp_path: Path, spec_text: str) -> str:
    """生成物里客户端模块的源码。"""
    _, artifacts = _build(tmp_path, spec_text)
    return next(
        source for path, source in _generated_modules(artifacts) if path.endswith("_client.py")
    )


def _generated_modules(artifacts: GeneratedArtifacts) -> list[tuple[str, str]]:
    """(path, source) —— 只取 create 的 .py 文件。"""
    return [
        (item.path, item.content)
        for item in artifacts.files
        if item.action == "create" and item.path.endswith(".py")
    ]


def _imported_modules(source: str) -> set[str]:
    """源码里真实出现的顶层 import 模块名（AST 级，不是子串搜索）。"""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    return imported


def _front_word(source: str) -> str:
    """把源码变回"人话"：去掉空白与注释，用于判断语句有没有被注入进来。"""
    return "".join(source.split())


# ====================================================== P0-1：恶意文本不改变语义


def test_malicious_openapi_still_produces_artifacts(tmp_path: Path) -> None:
    """生成必须仍然成功：安全修复不能退化成"遇到特殊字符就拒绝"。"""
    marker = tmp_path / "pwned"
    _, artifacts = _build(tmp_path, _malicious_spec(marker))

    assert artifacts.files
    assert _generated_modules(artifacts), "恶意 OpenAPI 下仍然必须产出 Python 文件"


def test_every_generated_python_file_compiles(tmp_path: Path) -> None:
    """每个生成文件都必须是语法有效的 Python。"""
    marker = tmp_path / "pwned"
    _, artifacts = _build(tmp_path, _malicious_spec(marker))

    for path, source in _generated_modules(artifacts):
        compile(source, path, "exec")


def test_malicious_text_does_not_become_an_import(tmp_path: Path) -> None:
    """payload 里写着 `import pathlib`——它必须没有变成生成模块的真 import。

    这是**语义级**断言：不管 payload 换成 `subprocess`、`__import__` 还是拼串，
    只要它变成了语句，模块的 import 集合就会偏离白名单。
    """
    marker = tmp_path / "pwned"
    _, artifacts = _build(tmp_path, _malicious_spec(marker))
    modules = _generated_modules(artifacts)
    # 生成代码自己所在的包（app / src.app / …）当然会被 import，它不是"外来"的
    owned = {path.split("/")[0] for path, _ in modules}
    allowed = ALLOWED_GENERATED_IMPORTS | owned

    for path, source in modules:
        stray = _imported_modules(source) - allowed
        assert not stray, f"{path} 出现了预期之外的 import：{sorted(stray)}"


def test_malicious_text_stays_inside_string_constants(tmp_path: Path) -> None:
    """文本逐字保留在字符串常量里——安全不是靠丢弃信息换来的。

    payload 在文档字符串位置（title / summary / description / 响应描述）出现，
    所以在生成模块的**字符串常量**里必须能原样找到它。找不到就说明实现是靠
    "删掉危险文本"过关的，那不是边界控制，是信息损失。
    """
    marker = tmp_path / "pwned"
    payload = _docstring_payload(marker)
    _, artifacts = _build(tmp_path, _malicious_spec(marker))

    found = False
    for _, source in _generated_modules(artifacts):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found = found or payload in node.value
    assert found, "payload 没有出现在任何字符串常量里，说明文本被丢掉了而不是被转义"


def test_importing_generated_package_does_not_run_payload(
    short_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把生成物当普通 Python 包 import，payload 要求的 marker 必须不存在。

    这里走的是**使用者的路径**，不是我们挑模块 exec：模块之间互相 import
    （`from app.xxx_config import ...`），所以必须整棵树落盘、把根目录挂上
    sys.path 之后按真实模块名导入——任何一条被注入的顶层语句都会在这时执行。
    """
    marker = short_root / "m"
    _, artifacts = _build(short_root, _malicious_spec(marker))

    for item in artifacts.files:
        if item.action != "create":
            continue
        target = short_root / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(short_root))
    saved = dict(sys.modules)
    try:
        imported = 0
        for path, _ in _generated_modules(artifacts):
            if path.startswith("tests/"):
                continue  # 测试文件不是包的一部分
            importlib.import_module(path[: -len(".py")].replace("/", "."))
            imported += 1
    finally:
        for name in [name for name in sys.modules if name not in saved]:
            del sys.modules[name]

    assert imported > 0
    assert not marker.exists(), "恶意 OpenAPI 文本在生成代码里执行了"


def test_pipeline_with_malicious_spec_does_not_execute_payload(tmp_path: Path) -> None:
    """完整 pipeline：TestRunner 在子进程里 import 生成的包并跑生成的测试。

    这是最强的一条证据——生成物被真的当作 Python 包加载、测试被真的收集并执行，
    而 marker 仍然不存在。所以这里刻意**不**断言 `status == "passed"`：
    恶意 path / 参数名会让生成的请求-断言往返对不上，那 5 条失败是预期的；
    要证明的是"没人执行 payload"，不是"恶意 spec 也能跑绿"。
    """
    marker = tmp_path / "pwned"
    _, artifacts = _build(tmp_path, _malicious_spec(marker))

    result = run_tests(artifacts)

    assert result.passed + result.failed > 0, f"测试没有被执行：\n{result.stdout}"
    assert result.errors == 0, f"生成物在 import 阶段就炸了：\n{result.stdout}"
    assert not marker.exists(), "pipeline 执行了恶意 OpenAPI 文本"


# 渲染层本身的往返用例：正常文本、会闭合字面量的文本、控制字符、非 ASCII、代理项。
ROUND_TRIP_PAYLOADS = [
    "普通摘要",
    '"""',
    "'''",
    '"',
    "\\",
    "a\nb",
    "\x00\x1b\v\f",
    "中文 😀 surrogates",
    "@@ -1 +1 @@\n-import os\n+pass\n",
    "ᐊᐃᓄᐃᑦ \u2028 \u2029",
    "\ud800 lone surrogate",
]


@pytest.mark.parametrize("payload", ROUND_TRIP_PAYLOADS)
def test_rendering_round_trips_untrusted_text(payload: str) -> None:
    """渲染出来的字面量必须"编译得过 + 取回原文"，两者缺一不可。

    只测"编译得过"会放过"内容被改掉"的实现（删几个字符当然也编译得过）；
    只测"取回原文"会放过"语法被改写"的实现。所以两条一起断言。

    唯一的例外是孤立代理项：它在 Python 源码里**没有任何写法**可表示
    （``compile()`` 会抛 ``UnicodeEncodeError: surrogates not allowed``），
    渲染器把它换成 U+FFFD，于是它被逐字比较时必然不等——这是有意的取舍，
    单独放宽（见下一条用例），不是"实现漏了"。
    """
    expected = payload if "\ud800" not in payload else payload.replace("\ud800", "�")
    for rendered in (string_literal(payload), docstring(payload, "")):
        compile(rendered, "<literal>", "exec")
        assert ast.literal_eval(rendered) == expected


def test_rendering_replaces_unrepresentable_surrogates() -> None:
    """孤立代理项必须被换成可表示的字符，而不是产出一个一 import 就炸的文件。

    留一颗"看起来语法正确、实际无法编译"的雷比替换字符危险得多：前者会在
    pipeline 里以"生成成功但 import 失败"的形式出现，定位成本远高于一个可见的 ``�``。
    """
    rendered = string_literal("\ud800")
    compile(rendered, "<literal>", "exec")
    assert ast.literal_eval(rendered) == "�"


def test_malicious_text_does_not_change_generated_structure(tmp_path: Path) -> None:
    """**本文件的核心断言**：恶意文本不能改变生成代码的结构。

    做法是把生成源码解析成 AST，抹掉所有字符串常量的**值**，再逐节点比较。
    两份 spec 除了描述文本外逐字节相同，所以标识符（类名 / 方法名 / 形参名）
    与控制流应当完全一致；一旦文本取得了语法地位（多出一个 import、多出一条
    语句、多一个实参），抹掉常量后仍然会露出来，AST 就不再相等。

    这比"源码里不包含 os.system"强得多：它不关心文本里**写的是什么**，
    只检查文本有没有变成**代码**。换成任何 payload 都成立。
    """
    clean = _client_source(tmp_path, _structure_spec(None))
    dirty = _client_source(tmp_path, _structure_spec(tmp_path / "pwned"))

    assert clean != dirty, "恶意文本没有进入生成源码，这条测试就没在测东西"
    assert _blank_literals(clean) == _blank_literals(dirty)


def _blank_literals(source: str) -> str:
    """把源码里的字符串常量值抹成空串，只留语法结构。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.dump(tree, annotate_fields=False)


# ============================================================ P1-1：无参数 endpoint


def test_endpoint_without_parameters_has_no_bare_star(tmp_path: Path) -> None:
    """`def health_check(self, *) -> None:` 是 SyntaxError，不能生成。"""
    _, artifacts = _build(tmp_path, _healthy_spec())
    client = next(
        source for path, source in _generated_modules(artifacts) if path.endswith("_client.py")
    )

    assert "def health_check(self) -> None:" in client
    assert "self, *)" not in client
    assert "self,*)" not in client


def test_endpoint_with_parameters_keeps_keyword_only(tmp_path: Path) -> None:
    """有关键字参数时 `*` 必须保留：这是既有契约，不能为了修 P1-1 丢掉它。"""
    client = _client_source(tmp_path, _structure_spec(None))

    assert "def get_pet(self, *, pet_id: str) -> None:" in client


def test_full_pipeline_from_openapi_to_test_runner(tmp_path: Path) -> None:
    """OpenAPI → Planner → Generator → GeneratedArtifacts → TestRunner 全链路。

    无参数端点是 P1-1 的触发条件：修好之前，生成的 client 连 import 都过不去，
    整个 pipeline 会停在 TestRunner 的收集期错误上。
    """
    plan, artifacts = _build(tmp_path, _healthy_spec())

    assert plan.endpoints, "Planner 必须解析出端点"
    client = next(
        source for path, source in _generated_modules(artifacts) if path.endswith("_client.py")
    )
    compile(client, "client.py", "exec")

    result = run_tests(artifacts)

    assert result.status == "passed", f"{result.status}: {result.stdout}\n{result.stderr}"
    assert result.passed > 0, "pipeline 必须真的执行了生成的测试"


def test_zero_test_result_is_never_treated_as_success(tmp_path: Path) -> None:
    """P0-2 在生成侧的同一条不变量：没有跑过测试就不算通过。"""
    _, artifacts = _build(tmp_path, _healthy_spec())
    result = run_tests(artifacts)

    if result.status == "passed":
        assert result.passed > 0
        assert result.failed == 0
        assert result.errors == 0


# ================================================ Follow-up：sink × payload 矩阵
#
# 上面几条用的是两个"复合 payload"（把闭合引号 + 换行 + 真语句拼在一起）。那能证明
# 已知的写法不管用，但证明不了"每个 sink 单独面对每一种形状的文本都成立"。这里按
# **字段 × payload** 铺开：一个字段一次只被一种形状的文本污染，逐个 compile。
#
# 覆盖的是"会进入真实仓库的 Python 文本"，包含两类：
#   - create 的完整模块；
#   - **modify 片段**（`__init__.py` 导出区、使用示例、模板文件）——它们会被贴进
#     真实文件，同样是待编译的 Python。
# 把片段纳进来不是凑数：`_init_snippet` 的漏网点只在片段里出现，而当时所有用例都
# 只看 create 的完整文件，所以它一直是绿的。

# OpenAPI 文本里最小的一组"字符串终结者 / 结构改写者"形状。
EVIL_PAYLOADS = [
    '"""',
    "\\",
    "\n",
    "'",
    '"',
    "#",
    "\r\n",
    "\u2028",
]

SINK_FIELDS = ("title", "summary", "description", "path", "operationId", "parameter")


def _spec_with(field: str, payload: str) -> str:
    """合法 spec，但把 payload 注入指定的那**一个**字段。"""
    path = "/pets/{petId}"
    parameter = {
        "name": "petId",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    }
    operation: dict = {
        "operationId": "getPet",
        "summary": "摘要",
        "description": "描述",
        "parameters": [parameter],
        "responses": {"200": {"description": "ok"}},
    }
    info = {"title": "Pets API", "version": "1.0"}
    if field == "title":
        info["title"] = "Pets " + payload
    elif field == "summary":
        operation["summary"] = payload
    elif field == "description":
        operation["description"] = payload
    elif field == "path":
        path = "/pets/{petId}" + payload
    elif field == "operationId":
        operation["operationId"] = "getPet" + payload
    elif field == "parameter":
        parameter["name"] = "petId" + payload
    else:  # pragma: no cover - 参数表写错时立刻暴露
        raise AssertionError(f"未知字段：{field}")
    return json.dumps(
        {
            "openapi": "3.0.3",
            "info": info,
            "servers": [{"url": "https://pets.test"}],
            "paths": {path: {"get": operation}},
        }
    )


def _generated_python(artifacts: GeneratedArtifacts) -> list[tuple[str, str]]:
    """**所有**会进入真实仓库的 Python 文本：完整模块 + modify 片段。"""
    return [(item.path, item.content) for item in artifacts.files if item.path.endswith(".py")]


@pytest.mark.parametrize("payload", EVIL_PAYLOADS)
@pytest.mark.parametrize("field", SINK_FIELDS)
def test_hostile_field_never_breaks_generated_python_syntax(
    tmp_path: Path, field: str, payload: str
) -> None:
    """6 个字段 × 8 种 payload：每一次都必须**全部** compile 通过。

    这里不判断生成物长得好不好看，只判断它是不是合法的 Python——即
    "hostile OpenAPI text cannot break generated Python syntax"。
    """
    _, artifacts = _build(tmp_path, _spec_with(field, payload))
    sources = _generated_python(artifacts)

    assert sources, f"{field}={payload!r} 下没有产出任何 Python 文本，这条用例就空了"
    for path, source in sources:
        compile(source, path, "exec")


# ============================================ Follow-up：新发现的 sink（逐个钉住）


def _plan_for(tmp_path: Path, spec_text: str):
    return plan_integration(parse_openapi_text(spec_text), scan_repository(_project(tmp_path)))


# `ErrorHandlingPlan.exception_type` 是**自由文本**字段（LLM 计划），却出现在
# `from ... import <这里>` 的标识符位置。plan 由模型构造，所以这条 payload 直接进
# plan——它模拟的正是"计划里这个字段不是干净标识符"的情形，与 deterministic
# planner 恰好会把它 slug 化无关。
EXCEPTION_TYPE_PAYLOAD = "APIError\nimport pathlib\npathlib.Path('pwned').write_text('x')\n"


def test_exception_type_cannot_smuggle_statements_into_the_package_init(
    tmp_path: Path,
) -> None:
    """`__init__.py` 导出区只有两条 import 语句，多一条都算注入成功。

    这是审计发现的漏网点：同一个 `exception_type` 在其余 6 处都过了 identifier
    收敛，只有这一处直接拼进了 import 行。所以断言必须是**语义级**的（AST 里
    只允许出现两条 ImportFrom），而不是"字符串里没有 import"——后者是黑名单。
    """
    plan = _plan_for(tmp_path, _healthy_spec())
    plan = plan.model_copy(
        update={
            "error_handling": plan.error_handling.model_copy(
                update={"exception_type": EXCEPTION_TYPE_PAYLOAD}
            )
        }
    )

    fragment = next(
        item.content
        for item in generate_code(plan).files
        if item.path.endswith("__init__.py") and item.action == "modify"
    )
    tree = ast.parse(fragment)
    kinds = [type(node).__name__ for node in ast.walk(tree)]

    compile(fragment, "__init__.py", "exec")
    assert "Import" not in kinds, f"exception_type 变成了真 import 语句：\n{fragment}"
    assert kinds.count("ImportFrom") == 2, fragment
    exported = tree.body[1].names[0].name
    assert exported.isidentifier(), f"导出的名字不是合法标识符：{exported!r}"


def test_usage_snippet_without_a_client_file_is_still_valid_python(tmp_path: Path) -> None:
    """计划里没有 client 文件时，不能产出 `from  import X` 这种坏代码。

    它是片段，但同样会被贴进真实文件；`compile()` 认不认是唯一的判据。
    """
    plan = _plan_for(tmp_path, _healthy_spec())
    plan = plan.model_copy(
        update={
            "files_to_create": [item for item in plan.files_to_create if item.kind != "client"],
            "files_to_modify": [
                FileModification(path="app/api.py", reason="接入客户端", changes=["新增调用"])
            ],
        }
    )

    fragment = next(item.content for item in generate_code(plan).files if item.path == "app/api.py")

    compile(fragment, "app/api.py", "exec")
    assert "from  import" not in fragment


def test_requirements_manifest_entries_cannot_add_lines() -> None:
    """requirements.txt 是**逐行**格式：条目里的换行等于注入额外的依赖行。

    与 pyproject 分支不同，这里没有"转义后照抄"的选项——清单行的语义由整行决定，
    转义只会得到一个同样畸形的依赖名。所以断言是"注入内容整条消失"。

    范围说明：这里管的是**结构**（一行文本不能变成两行），不是"哪些 pip 指令算
    合法"。单行 `-r http://...` 之类属于"计划声称要装什么"，与 dependency_changes
    一样是待人工复核的意图，不在本文件的威胁模型内。
    """
    item = FileModification(
        path="requirements.txt",
        reason="新增依赖",
        changes=["httpx>=0.27", "evil\ninjected==1", "also\r\ninjected==2", "  "],
    )

    content, point = _manifest_snippet(item)

    assert content.splitlines() == ["httpx>=0.27"], content
    assert "injected" not in content
    assert point  # 插入点说明仍然给出


def test_the_marker_payload_really_reaches_the_artifacts(tmp_path: Path) -> None:
    """E2E 攻击用例的**非空洞性**守卫。

    `test_pipeline_with_malicious_spec_does_not_execute_payload` 断言 marker 不存在。
    如果 payload 根本没进生成物，那条断言同样会通过——所以这里先证明：payload 里
    可执行的语句确实出现在生成源码中（只是作为**字符串常量**）。
    """
    marker = tmp_path / "pwned"
    _, artifacts = _build(tmp_path, _malicious_spec(marker))
    constants = [
        node.value
        for _, source in _generated_python(artifacts)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]

    # 比较**字符串常量的值**而不是源码文本：payload 里的反斜杠在源码里是转义过的，
    # 文本层面比不出来，而"值里有没有"才是"文本没有被丢弃/没有被执行"的问题。
    #
    # needle 必须按 payload 自己的写法构造（`{str(marker)!r}`）而不是 `str(marker)`：
    # Windows 路径的 repr 会把反斜杠写成 `\\`，所以 payload 的语句文本里本来就是**双**
    # 反斜杠——那才是"可执行"的形态。用单反斜杠去比会得到假阴性，而不是发现真问题。
    needle = f"pathlib.Path({str(marker)!r}).write_text('pwned', encoding='utf-8')"
    assert any("import pathlib" in value and needle in value for value in constants), (
        "payload 的可执行语句没有进入生成源码，E2E 那条断言就是空的"
    )
    assert not marker.exists()
