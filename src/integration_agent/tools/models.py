"""Agent Tool 的数据契约：ToolSpec / ToolCall / ToolResult。

设计约束：
    - 三个模型都是**纯数据**：不执行工具、不访问文件系统、不读环境变量、
      不执行 shell、不读取 API Key——执行行为属于具体 AgentTool 实现。
    - ToolSpec.parameters 既是给模型看的说明，也是参数校验的单一事实源，
      因此必须是可 JSON 序列化的 JSON-schema 风格对象。
    - ToolResult 是工具执行的**唯一出口**：成功、失败、异常一律返回结构化结果，
      绝不把 traceback 交给上层（traceback 属于服务端日志，不属于 Agent 上下文）。
    - 保证 Pydantic JSON round-trip：工具结果会进入 LLM 上下文与审计轨迹。
"""

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _require_nonblank(value: str, field: str) -> str:
    """去首尾空白后必须非空。

    只查 ``min_length`` 挡不住纯空白字符串（``"   "`` 长度合法却毫无意义），
    而工具名会成为 ToolRegistry 的键，空白键会让"查不到工具"变得难以排查。
    """
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} 不能为空")
    return cleaned


class ToolSpec(BaseModel):
    """一个工具的自描述：name / description / parameters。

    这里是**声明**，不是实现——不持有任何可调用对象，也不引用具体工具类。
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON-schema 风格
    read_only: bool = True  # v1 全部只读；留给未来显式标记非只读工具

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _require_nonblank(value, "工具名")

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        return _require_nonblank(value, "工具描述")

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        """parameters 必须能 JSON 序列化，且顶层是 object。

        序列化检查不是洁癖：ToolRegistry.describe() 会用 ``json.dumps`` 把它渲染进
        prompt。若放任 Path / set 之类的值进来，崩溃点会出现在渲染 prompt 时，
        离真正的肇事代码很远。
        """
        try:
            json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameters 必须可 JSON 序列化：{exc}") from None
        schema_type = value.get("type")
        if schema_type is not None and schema_type != "object":
            # 工具入参永远是一个对象；顶层声明成别的类型说明作者理解有偏差
            raise ValueError(
                f"parameters.type 必须是 'object'（工具入参是对象），实际是 {schema_type!r}"
            )
        return value


class ToolCall(BaseModel):
    """模型发起的一次工具调用请求（内容**不可信**，执行前必须校验）。

    args 来自 LLM 输出，任何字段都可能缺失、类型错误或越界；
    校验与拒绝发生在工具实现内部，本模型只保证"是一个对象"。
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""  # 多轮关联标识；单轮调用时可以留空

    @field_validator("tool")
    @classmethod
    def _validate_tool(cls, value: str) -> str:
        return _require_nonblank(value, "tool 名")


class ToolResult(BaseModel):
    """一次工具执行的结果：成功与失败走同一个结构。

    上层（Agent Runtime）只依赖 ok / content / error，不需要 try/except 包裹工具调用。
    """

    call_id: str
    tool: str
    ok: bool
    content: str  # 回灌给模型的文本，永远是字符串（失败时为空串，不可为 None）
    truncated: bool = False  # content 是否因长度上限被截断
    error: str | None = None  # ok=False 时的**脱敏后**原因
    chars: int = Field(default=0, ge=0)  # 预算计量；0 表示"未显式给出"，按 content 补齐

    @field_validator("error")
    @classmethod
    def _validate_error(cls, value: str | None) -> str | None:
        """错误文本不得夹带 traceback。

        这不是风格要求：ToolResult 会进入 LLM 上下文，也可能被序列化进 API 响应，
        而 traceback 里带着本地绝对路径、模块名与源码行。工具的异常必须由工具自己
        翻译成一句话，完整堆栈只写服务端日志——和 Pipeline 的既有约定一致。
        """
        if value is not None and "Traceback (most recent call last)" in value:
            raise ValueError("error 不允许包含 traceback（完整堆栈只写服务端日志）")
        return value

    @model_validator(mode="after")
    def _fill_chars(self) -> "ToolResult":
        """chars 未显式给出时按 content 实际长度补齐。

        chars 用于累计观察预算（超过上限就停止 Agent 循环）。若调用方构造时忘记传
        chars，静默的 0 会让预算永远涨不上去，循环也就失去了这道刹车——
        所以这里取实际长度而不是留着 0。显式传入的正数不会被覆盖。
        """
        if self.chars == 0 and self.content:
            self.chars = len(self.content)
        return self
