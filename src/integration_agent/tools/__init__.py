"""Read-only tools for the APIForge agent runtime.

Agent Tool Loop 的工具层：把"取信息"的能力从 Planner 里独立出来，
让 LLM 按需调用，而不是由程序预先算好检索词。

结构化对象：
    ToolSpec     —— 工具自描述（name / description / parameters / read_only）
    ToolCall     —— 模型发起的一次调用请求（内容不可信）
    ToolResult   —— 执行结果（成功与失败同一结构）
    ToolContext  —— 工具执行时可见的只读上下文

行为契约：
    AgentTool    —— Protocol，工具实现只需提供 spec 与 invoke(args, ctx)
    ToolRegistry —— name → AgentTool 的注册表，构造时校验、不执行、不吞异常

只读工具：
    inspect_project —— 扫描项目结构（复用 repository.scanner）
    search_code     —— 在 .py 文件中按关键词搜索（复用 repository.code_search）
    read_file       —— 读取项目内某个文件，带路径围栏与长度上限
    inspect_api     —— 查看已经解析好的 APIInfo（概要 / 端点索引 / 单端点详情）
    build_default_registry() 一次性拿到注册好这四个工具的注册表。

依赖方向：
    tools → api / repository
工具层不依赖 agent / repair / pipeline / api_server，可被任何上层复用。

安全边界：
    - 全部工具只读：不写文件、不删除文件、不执行 shell / git / 子进程、
      不发网络请求、不装依赖、不修改环境变量。
    - 不读取 API Key，不读取环境变量，拒绝读取 .env 等凭据文件。
    - 一切路径都必须落在 project_root 内，且经 resolve() 确认（可挡 symlink 逃逸）。
    - ToolContext 不含任何凭据或执行能力（见 registry.ToolContext）。
"""

from integration_agent.tools.api_tools import InspectAPITool
from integration_agent.tools.models import ToolCall, ToolResult, ToolSpec
from integration_agent.tools.registry import (
    AgentTool,
    ToolContext,
    ToolRegistry,
    ToolRegistryError,
)
from integration_agent.tools.repository_tools import (
    InspectProjectTool,
    ReadFileTool,
    SearchCodeTool,
    build_default_registry,
)

__all__ = [
    "AgentTool",
    "InspectAPITool",
    "InspectProjectTool",
    "ReadFileTool",
    "SearchCodeTool",
    "ToolCall",
    "ToolContext",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolSpec",
    "build_default_registry",
]
