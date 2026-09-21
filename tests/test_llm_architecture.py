"""P1-1 / P1-2 的架构守卫：把"依赖方向"变成会失败的断言。

这份文件不测行为，只测**结构**。理由：行为测试全绿也可能是一次假重构——把
`from integration_agent.repair.llm_client import LLMClient` 改成
`from integration_agent.llm import LLMClient`，功能一模一样，但耦合一点没少。
所以这里逐个模块解析 AST，直接对 import 与标识符下断言。

它守的是这几条边界：

    llm  →  tools.models      （ToolSpec 是纯数据契约，允许）
    llm  ↛  agent / repair / api_server / pipeline / generation / validation
    llm  ↛  tools.registry    （契约层没有执行工具的入口）

    agent / repair / api_server / pipeline  →  llm      （单向，允许且必须有）
    agent_loop  ↛  provider wire format / 适配器

反向验证（§十九）也在这里：改造前存在的四条旧边
（repair→agent.llm、agent.deepseek_planner⇢repair.llm_client、
api_server→repair 取 transport、agent_loop→DeepSeek wire format）
逐条断言为"已经消失"。
"""

import ast
from importlib import import_module
from pathlib import Path

import pytest

import integration_agent
from integration_agent.agent import agent_loop as loop_module
from integration_agent.agent import deepseek_planner as planner_module
from integration_agent.llm import client as client_module
from integration_agent.llm import deepseek as deepseek_module
from integration_agent.llm import models as models_module

SRC = Path(integration_agent.__file__).resolve().parent

# llm 包的全部模块：每一个都受同样的依赖方向约束
LLM_MODULES = ("__init__.py", "models.py", "client.py", "deepseek.py")

# llm 允许触达的项目内模块，白名单而非黑名单
ALLOWED_PROJECT_IMPORTS = {
    "integration_agent.llm",
    "integration_agent.llm.client",
    "integration_agent.llm.deepseek",
    "integration_agent.llm.models",
    "integration_agent.tools.models",
}

# llm 绝对不允许依赖的 domain 包
FORBIDDEN_PROJECT_PREFIXES = (
    "integration_agent.agent",
    "integration_agent.repair",
    "integration_agent.api_server",
    "integration_agent.pipeline",
    "integration_agent.generation",
    "integration_agent.validation",
    "integration_agent.tools.registry",
    "integration_agent.tools.api_tools",
    "integration_agent.tools.repository_tools",
)

# provider 名字：provider-neutral 模块里一个都不许出现
PROVIDER_NAMES = ("deepseek", "openai", "anthropic")

# transport 标记：适配器之外不该出现这些词
TRANSPORT_MARKERS = (
    "response_format",
    "chat/completions",
    "authorization",
    "bearer",
    "base_url",
    "api_key",
    "urllib",
    "json_mode",
)


# ------------------------------------------------------------------ helpers


def _source(source_file: Path) -> str:
    return source_file.read_text(encoding="utf-8")


def _imported_modules(source_file: Path) -> set[str]:
    tree = ast.parse(_source(source_file))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """模块 / 类 / 函数的第一条字符串表达式——文档，不是代码。"""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    found.add(id(body[0].value))
    return found


def _code_strings(source_file: Path) -> list[str]:
    """代码里的字符串常量（排除文档字符串与 f-string 里的模板片段）。

    排除文档字符串是有意的：文档里写一句"适配器在某个文件里"是在指路，
    而一个真的被构造出来的 {"type": "function", ...} 才是 wire format。
    """
    tree = ast.parse(_source(source_file))
    skip = _docstring_nodes(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                found.append(node.value)
    return found


def _identifiers(source_file: Path) -> list[str]:
    """代码里出现的标识符：变量名、属性名、import 的符号名。"""
    tree = ast.parse(_source(source_file))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(node, ast.alias):
            found.append(node.name)
            if node.asname:
                found.append(node.asname)
    return found


def _llm_files() -> list[Path]:
    return [SRC / "llm" / name for name in LLM_MODULES]


# --------------------------------------------------- §十四.1~4：llm 的边界


@pytest.mark.parametrize("module_name", LLM_MODULES)
def test_llm_module_imports_no_domain_package(module_name: str) -> None:
    """llm 在依赖图的最底层：它不能认识任何一个 domain 包。"""
    imported = _imported_modules(SRC / "llm" / module_name)

    for forbidden in FORBIDDEN_PROJECT_PREFIXES:
        assert not any(item.startswith(forbidden) for item in imported), (
            f"llm/{module_name} 依赖了 {forbidden}"
        )


@pytest.mark.parametrize("module_name", LLM_MODULES)
def test_llm_module_only_reaches_tools_models(module_name: str) -> None:
    imported = _imported_modules(SRC / "llm" / module_name)

    project_imports = {item for item in imported if item.startswith("integration_agent")}
    extra = project_imports - ALLOWED_PROJECT_IMPORTS
    assert not extra, f"llm/{module_name} 出现了白名单之外的项目内 import：{sorted(extra)}"


def test_llm_does_not_import_agent() -> None:
    """§十四.1 —— 旧边 repair.deepseek_client → agent.llm 的另一半。"""
    for path in _llm_files():
        imported = _imported_modules(path)
        assert not any(item.startswith("integration_agent.agent") for item in imported), path.name


def test_llm_does_not_import_repair() -> None:
    """§十四.2 —— LLM contract 不再住在 repair 里，也不再回指 repair。"""
    for path in _llm_files():
        imported = _imported_modules(path)
        assert not any(item.startswith("integration_agent.repair") for item in imported), path.name


def test_llm_does_not_import_api_server() -> None:
    """§十四.3 —— HTTP 适配层是最上层，llm 不可能依赖它。"""
    for path in _llm_files():
        imported = _imported_modules(path)
        assert not any(item.startswith("integration_agent.api_server") for item in imported), (
            path.name
        )


def test_llm_does_not_import_the_tool_registry() -> None:
    """契约层认识 ToolSpec（数据），不认识 ToolRegistry（查找与执行入口）。"""
    for path in _llm_files():
        imported = _imported_modules(path)
        assert "integration_agent.tools" not in imported, path.name
        assert "integration_agent.tools.registry" not in imported, path.name


def test_models_module_needs_no_project_import_at_all() -> None:
    """llm/models.py 是纯数据模型：连 ToolSpec 都不需要，更不该有项目内依赖。"""
    imported = _imported_modules(Path(models_module.__file__).resolve())

    assert not {item for item in imported if item.startswith("integration_agent")}


# ------------------------------- §十四.4：provider-neutral 模块不含 provider 内容


@pytest.mark.parametrize("module", [models_module, client_module])
def test_provider_neutral_module_names_no_provider(module) -> None:
    """models.py / client.py 里连厂商的名字都不该出现——文档里也不行。

    刻意用整份源码而不是只查标识符：这两个文件是"所有 provider 都能用的
    那一层"，一旦文里文外开始提某个厂商，下一步就是往里加它的专有字段。
    """
    text = _source(Path(module.__file__).resolve()).lower()

    for name in PROVIDER_NAMES:
        assert name not in text, f"{Path(module.__file__).name} 里出现了 provider 名字 {name}"


@pytest.mark.parametrize("module", [models_module, client_module])
def test_provider_neutral_module_has_no_transport_logic(module) -> None:
    """§十四.4 —— 契约层与数据层不得出现 transport 逻辑。

    查的是代码标识符 + 非文档字符串常量：`response_format` / `Authorization`
    这类词出现在这些位置，就说明确实有一行代码在拼 provider 的请求。
    """
    path = Path(module.__file__).resolve()
    needles = _identifiers(path) + _code_strings(path)

    for needle in needles:
        lowered = needle.lower()
        for marker in TRANSPORT_MARKERS:
            assert marker not in lowered, f"{path.name} 里出现了 transport 标记 {marker}：{needle}"


# ------------------------------- §十四.5~6：AgentLoop 不碰 provider wire format


def test_agent_loop_does_not_import_the_deepseek_adapter() -> None:
    """§十四.5 —— 反证旧边 agent_loop → tool_spec_to_deepseek_function。"""
    source = _source(Path(loop_module.__file__).resolve())

    assert "tool_spec_to_deepseek_function" not in source
    assert "deepseek" not in source.lower()
    assert "llm.deepseek" not in source
    imported = _imported_modules(Path(loop_module.__file__).resolve())
    assert "integration_agent.llm.deepseek" not in imported
    assert "integration_agent.llm" in imported  # 契约仍然来自 llm


def test_agent_loop_does_not_build_provider_wire_format() -> None:
    """§十四.6 —— 回路里不能出现 {"type": "function", "function": {...}} 的构造。"""
    source = _source(Path(loop_module.__file__).resolve())

    assert '"type": "function"' not in source
    assert "'type': 'function'" not in source
    assert '"function"' not in source and "'function'" not in source


def test_agent_loop_tools_are_specs_not_dicts() -> None:
    """§十四.6 —— 交给客户端的 tools 是 ToolSpec 实例，不是 dict。"""
    source = _source(Path(loop_module.__file__).resolve())

    assert "tools = registry.specs()" in source
    assert "for spec in registry.specs()" not in source  # 不再有"逐个转换"的循环


def test_llm_client_protocol_accepts_tool_spec() -> None:
    """契约的参数类型本身必须说"我要 ToolSpec"。"""
    import inspect

    from integration_agent.llm import ToolCallingClient

    annotation = inspect.signature(ToolCallingClient.chat).parameters["tools"].annotation

    assert "ToolSpec" in str(annotation)


# ------------------------------ §十四.7：适配器负责转换，payload 与原版一致


def test_only_the_adapter_builds_the_wire_format() -> None:
    """整棵源码树里，**构造** tools[] 的 {"type": "function"} 只允许出现在适配器。

    查的是代码里的字符串常量（AST），不是整份源码：文档里写一句「你不需要知道
    "type": "function" 这类形状」是在划边界，而一行真的拼出这个 dict 的代码
    才是把 provider schema 漏到了适配器之外。
    """
    adapter = Path(deepseek_module.__file__).resolve()
    offenders: list[str] = []

    for path in sorted(SRC.rglob("*.py")):
        if path.resolve() == adapter:
            continue
        for literal in _code_strings(path):
            # 两个 wire-format 键：tools[] 元素的 type 值与 function 包装层。
            # 全树只有适配器会真的拼出它们（对照 grep 可自证）。
            if literal in ('"function"', "'function'", '"type": "function"', "'type': 'function'"):
                offenders.append(f"{path.relative_to(SRC)}: {literal!r}")

    assert offenders == [], f"provider wire format 泄漏到了这些模块：{offenders}"


def test_adapter_converts_tool_specs_into_deepseek_tools() -> None:
    """§十四.7 —— ToolSpec 进、DeepSeek tools[] 出，且形状与旧版逐键一致。"""
    from integration_agent.llm import tool_spec_to_deepseek_function
    from integration_agent.tools.models import ToolSpec

    spec = ToolSpec(
        name="search_code",
        description="搜索",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    assert tool_spec_to_deepseek_function(spec) == {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "搜索",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
    # 转换函数住在适配器里，不在契约层
    assert tool_spec_to_deepseek_function.__module__ == "integration_agent.llm.deepseek"
    assert deepseek_module.DeepSeekLLMClient.__module__ == "integration_agent.llm.deepseek"


# ------------------------------ §十九：四条旧边逐条做反向验证


def test_old_llm_transport_modules_are_gone() -> None:
    """旧位置必须真的不存在（不留 re-export shim）。"""
    assert not (SRC / "agent" / "llm.py").exists()
    assert not (SRC / "repair" / "llm_client.py").exists()
    assert not (SRC / "repair" / "deepseek_client.py").exists()


def test_old_edge_repair_to_agent_llm_is_gone() -> None:
    """旧边 1：repair.deepseek_client → agent.llm（模块已迁移，边随之消失）。"""
    for path in (SRC / "repair").rglob("*.py"):
        imported = _imported_modules(path)
        assert not any(item.startswith("integration_agent.agent") for item in imported), path.name


def test_old_edge_planner_to_repair_is_gone() -> None:
    """旧边 2：agent.deepseek_planner ⇢ repair.llm_client（TYPE_CHECKING 变通已删）。

    现在 LLMClient 是**运行时** import，且来自 llm 而不是 repair —— 既不再有
    那条被抑制的边，也不再需要抑制它。
    """
    path = Path(planner_module.__file__).resolve()
    imported = _imported_modules(path)
    source = _source(path)

    assert "integration_agent.llm" in imported  # 运行时 import，不是 TYPE_CHECKING
    assert not any(item.startswith("integration_agent.repair") for item in imported)
    # 那个用来绕开 import cycle 的 if TYPE_CHECKING: 块已经整块删除
    assert "if TYPE_CHECKING:" not in source
    assert "from __future__ import annotations" in source  # 其余前向引用不受影响


def test_old_edge_api_server_to_repair_transport_is_gone() -> None:
    """旧边 3：api_server 从 repair 包里拿 DeepSeek transport。

    现在 api_server 从 llm 取适配器、从 repair 取领域组件（Applier），两条
    来源分开。monkeypatch.setattr(app, "DeepSeekLLMClient", ...) 这类测试因此
    仍然有效——模块级名字没有变，变的只是它从哪来。
    """
    path = SRC / "api_server" / "app.py"
    imported = _imported_modules(path)
    source = _source(path)

    assert "integration_agent.llm" in imported
    assert "integration_agent.repair" in imported
    assert "from integration_agent.llm import DeepSeekConfigError, DeepSeekLLMClient" in source
    assert "from integration_agent.repair import StructuredLLMRepairApplier" in source

    # from integration_agent.api_server import app 拿到的是 FastAPI 实例，
    # 这里要的是模块对象本身（monkeypatch.setattr 打的就是它）。
    app_module = import_module("integration_agent.api_server.app")

    assert app_module.DeepSeekLLMClient.__module__ == "integration_agent.llm.deepseek"
    assert app_module.DeepSeekConfigError.__module__ == "integration_agent.llm.deepseek"


def test_repair_package_no_longer_re_exports_transport() -> None:
    """§十二 —— repair 只导出 repair domain，不留 deprecated 的 transport 再导出。"""
    from integration_agent import repair

    for name in (
        "DEFAULT_MODEL",
        "DeepSeekAPIError",
        "DeepSeekConfigError",
        "DeepSeekLLMClient",
        "DeepSeekLLMError",
        "DeepSeekResponseError",
        "DeepSeekTimeoutError",
        "FakeLLMClient",
        "LLMClient",
    ):
        assert not hasattr(repair, name), f"repair 仍然导出了 {name}"
        assert name not in repair.__all__, f"repair.__all__ 里仍然有 {name}"

    assert hasattr(repair, "StructuredLLMRepairApplier")


def test_dependency_direction_is_one_way() -> None:
    """整张图的最终形状：domain → llm，llm 谁都不指。

    agent / repair / api_server / pipeline 都可以 import llm；
    llm 反过来一个都不 import。单向，因此没有环。
    """
    consumers = {
        "agent": SRC / "agent" / "agent_loop.py",
        "repair": SRC / "repair" / "llm_repair_applier.py",
        "api_server": SRC / "api_server" / "app.py",
    }

    for name, path in consumers.items():
        assert "integration_agent.llm" in _imported_modules(path), f"{name} 没有依赖 llm"

    for path in _llm_files():
        imported = _imported_modules(path)
        assert not any(
            item.startswith(
                (
                    "integration_agent.agent",
                    "integration_agent.repair",
                    "integration_agent.api_server",
                    "integration_agent.pipeline",
                )
            )
            for item in imported
        ), path.name
