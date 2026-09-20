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

依赖方向：
    tools → api / repository
工具层不依赖 agent / repair / pipeline / api_server，可被任何上层复用。

安全边界：
    - v1 全部工具只读：不写文件、不执行 shell / git、不发网络请求、不装依赖。
    - 不读取 API Key，不读取环境变量。
    - ToolContext 不含任何凭据或执行能力（见 registry.ToolContext）。
"""

from integration_agent.tools.models import ToolCall, ToolResult, ToolSpec
from integration_agent.tools.registry import (
    AgentTool,
    ToolContext,
    ToolRegistry,
    ToolRegistryError,
)

__all__ = [
    "AgentTool",
    "ToolCall",
    "ToolContext",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolSpec",
]
