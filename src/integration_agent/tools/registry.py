"""Agent Tool 的上下文与注册表：ToolContext / AgentTool / ToolRegistry。

依赖方向（硬约束）：
    agent / tools  →  api / repository
本模块**不得**反向依赖 repair / pipeline / api_server / agent：
    - 不 import DeepSeekPlanner——工具不该知道谁在调用它；
    - 不 import repair——repair 依赖 generation 依赖 agent，反向引用会立刻成环。
工具层是最底层的只读能力，任何上层都可以复用。

安全边界：
    - ToolContext 只承载只读事实：仓库根目录、项目结构、API 信息、已知文件集合。
      **不含** API Key、**不含**环境变量、**不含** shell/子进程执行器、
      **不含**写/删/改能力——这三样是"不许有"，不是"暂时没用到"。
    - ToolRegistry 只维护 name → AgentTool 的映射：**不执行**任何工具，
      也**不捕获/吞掉**工具内部的异常。异常语义由工具实现与调用方各自决定，
      注册表夹在中间做 try/except 只会掩盖真实故障。
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from integration_agent.api import APIInfo
from integration_agent.repository import ProjectStructure
from integration_agent.tools.models import ToolResult, ToolSpec


class ToolRegistryError(ValueError):
    """工具注册失败（名字为空、重复、或不是合法的工具对象）时抛出。

    沿用项目既有错误风格（RepositoryError / PlanningError 同样继承 ValueError），
    不额外引入异常层次。
    """


class ToolContext(BaseModel):
    """工具执行时可见的**只读**环境。

    这里刻意不提供任何"能力型"字段（executor / http client / writer）：
    工具的能力上限应该由它自己的代码决定，而不是由运行时塞进来的对象决定。
    """

    project_root: Path  # 目标仓库根目录；工具的一切路径都必须落在这里面
    project: ProjectStructure  # 仓库结构事实（来自 repository.scanner）
    api: APIInfo  # 本次集成的 API 事实（来自 api.parser）
    known_files: set[str]  # 确认存在的仓库文件（相对 POSIX 路径），供工具做路径白名单


@runtime_checkable
class AgentTool(Protocol):
    """一个可被 Agent 调用的只读工具。

    只依赖 Protocol：注册表不关心实现是查代码、读文件还是查 API，
    也不关心它是确定性实现还是未来的 LLM 驱动实现。
    """

    spec: ToolSpec

    def invoke(
        self,
        args: dict,
        ctx: ToolContext,
    ) -> ToolResult:
        """执行工具并返回结构化结果。

        约定：
            - args 是不可信输入，实现必须自行校验；
            - 任何失败都返回 ``ok=False`` 的 ToolResult，不向上抛异常；
            - 绝不写文件 / 执行 shell / 发网络请求。
        """
        ...


class ToolRegistry:
    """工具注册表：name → AgentTool。

    构造时即完成全部校验（名字非空、不重复、具备 spec 与 invoke），
    因此运行期的 ``get()`` 只需要一次字典查询，不必再做防御。
    """

    def __init__(self, tools: Iterable[AgentTool]) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            spec = getattr(tool, "spec", None)
            if not isinstance(spec, ToolSpec):
                raise ToolRegistryError(
                    f"工具必须提供 ToolSpec（spec 字段），实际是 {type(spec).__name__}"
                )
            if not callable(getattr(tool, "invoke", None)):
                raise ToolRegistryError(f"工具 {spec.name!r} 必须实现 invoke(args, ctx)")
            name = spec.name.strip()
            if not name:
                raise ToolRegistryError("工具名不能为空")
            if name in self._tools:
                raise ToolRegistryError(f"工具名重复：{name!r}")
            self._tools[name] = tool

    def get(self, name: str) -> AgentTool | None:
        """按名字取工具；未知名字返回 None（不抛异常，由调用方决定如何呈现）。"""
        return self._tools.get(name.strip())

    def specs(self) -> list[ToolSpec]:
        """按名称排序返回全部 ToolSpec。

        排序而非插入序：调用方可能传进来一个 set，插入序不稳定；而这份清单会被
        渲染进 prompt，顺序抖动会让"同样输入产生同样 prompt"不再成立。
        """
        return [self._tools[name].spec for name in sorted(self._tools)]

    def describe(self) -> str:
        """渲染确定性的工具说明（供 prompt 使用）。

        用 ``sort_keys=True`` 与紧凑分隔符，保证同一个 spec 无论 dict 内部
        插入顺序如何，渲染结果都逐字节一致。未注册任何工具时返回空串，
        由调用方决定如何在 prompt 中呈现。
        """
        blocks: list[str] = []
        for spec in self.specs():
            schema = json.dumps(
                spec.parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            blocks.append(f"- {spec.name}: {spec.description}\n  parameters: {schema}")
        return "\n".join(blocks)
