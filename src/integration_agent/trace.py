"""Execution Trace：把一次运行的过程记录成结构化事件，供观察、调试与展示。

这一层是**观测**，不是业务：

    - 只收集数据，不参与任何阶段的输入、输出与判定；
    - ``emit()`` 永不抛异常——观测层不能把一次运行搞崩；
    - 默认不写磁盘、不用数据库、不发网络请求、不引入日志框架（纯内存 Pydantic 对象）。

传播方式：作用域，而不是层层传参
--------------------------------
需要埋点的地方分布在**嵌套很深**的位置（Agent Loop 里的每一次工具调用、Repair Loop
里的每一次测试），而这条路径上的组件（AgentLoop / TestRunner / RepairLoop）都是
依赖注入进来的稳定实现，各有自己的 Protocol。为了埋点去改它们的 `run/plan/generate`
签名，等于把"观测"焊进"业务契约"里。所以这里用 contextvars 做**作用域传播**：

    trace = TraceCollector()
    run_pipeline(spec, repo, trace=trace)   # 各层用 emit(...) 埋点，签名一字不改
    trace.events()

没有打开作用域时 ``emit()`` 是空操作：默认零开销、零行为差异（Backward Compatible）。

用 contextvars 而不是模块级全局变量：api_server 会并发处理请求，全局变量会把两个
请求的事件串到一起；contextvars 按线程 / async task 隔离，各跑各的。

字段与安全
----------
``metadata`` 会被规范化（见 :func:`sanitize_metadata`）：只保留 JSON 可序列化的标量、
按键名脱敏（api_key / token / password / secret ...）、超长字符串截断。它保证的是
"不会因为一句顺手写下的调试代码把凭据或整个源码写进 trace"，**不是**"调用方可以随便
往里塞东西"——埋点处仍然只放摘要：工具名、计数、状态、耗时。

同理，事件里不出现完整 prompt、完整 LLM 响应、完整源码、完整 diff 与完整 repository：
trace 是"运行时发生了什么"的骨架，不是数据副本。
"""

import logging
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# 阶段名：与 pipeline 的六个阶段一一对应，外加 agent（Agent Loop 自身）与 repair。
# 封闭集合：写错一个阶段名应当在类型检查 / 测试里暴露，而不是变成一个静默的新阶段。
TraceStage = Literal[
    "api_understanding",
    "repository_understanding",
    "agent",
    "planner",
    "generation",
    "test_runner",
    "repair",
    "patch",
]

# 事件类型：同样封闭。只覆盖"阶段生命周期 / 规划 / 工具 / LLM / 测试 / 修复 / 产物"，
# 不为想象中的未来一次性设计几十种事件。
#
# planning_* / generation_completed 与 stage_* 分开，是因为它们回答的是两个问题：
# stage_* 是"这次阶段调用花了多久"（由 pipeline 记），planning_* / generation_completed
# 是"这一层自己干了什么"（由规划器 / 生成器记）。这两层被单独使用时没有 pipeline
# 那层，后者就是唯一的信号。
#
# 加一种事件类型时**必须同时**改这里：词表是本模块唯一的准入检查，没登记的
# event_type 会被静默丢掉（只留一条 DEBUG 日志）。tests/test_trace.py 会静态扫描
# 全仓库的 emit() 调用点，把"埋了没登记"挡在测试里。
TraceEventType = Literal[
    "stage_started",
    "stage_completed",
    "stage_failed",
    "planning_started",
    "planning_completed",
    "generation_completed",
    "tool_called",
    "tool_completed",
    "llm_called",
    "llm_completed",
    "test_started",
    "test_completed",
    "repair_started",
    "repair_completed",
    "patch_generated",
]

# 按键名脱敏：键名命中即整个值被替换。这是**兜底**，不是主要防线——主要防线是埋点处
# 不把凭据放进来。做成按键名匹配而不是值模式匹配，是因为值模式永远追不上新的凭据格式
# （新前缀、新长度、新编码），而键名是可枚举的有限集合。
_SECRET_KEY = re.compile(
    r"api[-_]?key|apikey|token|password|passwd|secret|authorization|credential"
    r"|private[-_]?key|access[-_]?key|bearer",
    re.IGNORECASE,
)

# 脱敏后的占位符：保留"这里原本有个值"的信息，但值本身不保留。
REDACTED = "***"

# 单个字符串值的上限。它的作用是挡住"把一段源码 / 一个响应体塞进 metadata"这类
# 顺手操作——trace 里该有的是摘要，不是数据副本。
MAX_STRING_CHARS = 200

# metadata 的嵌套深度上限（超过即替换为占位符），防止病态结构把事件撑大。
MAX_METADATA_DEPTH = 4


class TraceEvent(BaseModel):
    """一次运行中的一个事件：谁（stage）、发生了什么（event_type）、什么时候。

    字段刻意保持扁平且可 JSON round-trip：trace 的消费方是调试界面与测试，
    两者都不该为了读一个事件去理解内部对象图。

    status 的自由取值来自被观测组件自己的状态词表（TestResult.status /
    RepairLoopResult.status / AgentStatus）——这里刻意不做成封闭 Literal：
    把各阶段的领域状态硬塞进一个联合类型，只会让新增状态变成 trace 的破坏性变更。
    """

    timestamp: float  # 事件产生的墙钟时间（epoch 秒）
    stage: TraceStage
    event_type: TraceEventType
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration: float | None = None  # 秒；只对"有明确起止"的事件（阶段 / 调用）有值
    status: str | None = None  # 该事件的结论（passed / failed / error / max_turns ...）


def _is_secret_key(key: Any) -> bool:
    return bool(_SECRET_KEY.search(str(key)))


def _sanitize_value(value: Any, depth: int) -> Any:
    """把任意值收敛成 JSON 可序列化的标量结构（超深 / 超长 / 未知类型都被降级）。"""
    if depth > MAX_METADATA_DEPTH:
        return "<max depth>"
    if isinstance(value, str):
        if len(value) <= MAX_STRING_CHARS:
            return value
        return f"{value[:MAX_STRING_CHARS]}...(+{len(value) - MAX_STRING_CHARS} chars)"
    # bool 必须在 int 之前判断：bool 是 int 的子类，顺序反了会把 True 变成 1
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, Mapping):
        return sanitize_metadata(value, depth=depth + 1)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        # 集合本身无序：排序后再落成列表，否则同一份数据会产生不同顺序的事件内容
        return [_sanitize_value(item, depth + 1) for item in sorted(value, key=repr)]
    return f"<{type(value).__name__}>"


def sanitize_metadata(metadata: Mapping[str, Any] | None, *, depth: int = 1) -> dict[str, Any]:
    """规范化 metadata：按键名脱敏、值收敛成 JSON 标量结构、键统一为字符串。

    返回的一定是**新的**字典：调用方后续改动自己的 dict 不会污染已记录的事件。
    """
    if not metadata:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        name = str(key)
        cleaned[name] = REDACTED if _is_secret_key(name) else _sanitize_value(value, depth)
    return cleaned


class TraceCollector:
    """内存事件收集器：一次运行一个实例，``events()`` 按发生顺序返回全部事件。

    默认不写磁盘：持久化是调用方的事（谁需要，谁在运行结束后自己 dump），
    这样 trace 层就不会悄悄产生文件、也就不会影响"运行不修改真实仓库"这条约束。

    线程安全：``list.append`` 在 CPython 下是原子的，并发 emit 不会丢事件（顺序可能
    交错）。一次 pipeline 运行的埋点都在同一个线程 / async task 内，因此顺序稳定。
    """

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def emit(
        self,
        stage: TraceStage,
        event_type: TraceEventType,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        duration: float | None = None,
        status: str | None = None,
    ) -> None:
        """记录一个事件。

        **不抛异常**：stage / event_type 不合法（拼错、或新加了尚未登记的取值）
        时丢弃该事件并返回。观测层让业务流程因为"埋点写错了"而失败，是本末倒置；
        写错的那一处会在测试里暴露，而不是在这里炸掉一次真实的集成运行。

        丢弃会留下一条 DEBUG 日志：静默丢弃过一次事件类型（埋点写了、词表里没有）
        就足以让一次排障在错误的方向上找很久，而 DEBUG 又不会在生产里吵人。
        """
        try:
            event = TraceEvent(
                timestamp=time.time(),
                stage=stage,
                event_type=event_type,
                message=message,
                metadata=sanitize_metadata(metadata),
                duration=duration,
                status=status,
            )
        except ValidationError as exc:
            logger.debug("Trace 事件被丢弃（stage=%r, event_type=%r）：%s", stage, event_type, exc)
            return
        self._events.append(event)

    def events(self) -> list[TraceEvent]:
        """事件列表的**副本**：调用方排序 / 过滤不会改到收集器内部状态。"""
        return list(self._events)

    def clear(self) -> None:
        """清空已收集的事件（复用同一个 collector 跑下一次运行时用）。"""
        self._events.clear()


# --------------------------------------------------------------- 作用域传播

_active: ContextVar[TraceCollector | None] = ContextVar("apiforge_trace", default=None)


def current_trace() -> TraceCollector | None:
    """当前作用域的收集器；没有打开作用域时返回 None。"""
    return _active.get()


@contextmanager
def use_trace(collector: TraceCollector | None) -> Iterator[TraceCollector | None]:
    """把 collector 设为当前作用域，退出时恢复上一层（嵌套安全）。"""
    token = _active.set(collector)
    try:
        yield collector
    finally:
        _active.reset(token)


@contextmanager
def trace_scope(collector: TraceCollector | None) -> Iterator[None]:
    """``None`` → 保持外层作用域不变；非 None → 本次运行内改用它。

    区分这两种情况是必要的：pipeline 的 ``trace=None`` 表示"调用方没有要求收集"，
    不该顺手把外层（比如测试或上层应用打开的）作用域抹掉。
    """
    if collector is None:
        yield
        return
    with use_trace(collector):
        yield


def emit(
    stage: TraceStage,
    event_type: TraceEventType,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    duration: float | None = None,
    status: str | None = None,
) -> None:
    """向当前作用域发一个事件；没有作用域时什么都不做（默认零开销）。"""
    collector = _active.get()
    if collector is None:
        return
    collector.emit(stage, event_type, message, metadata=metadata, duration=duration, status=status)


__all__ = [
    "MAX_STRING_CHARS",
    "REDACTED",
    "TraceCollector",
    "TraceEvent",
    "TraceEventType",
    "TraceStage",
    "current_trace",
    "emit",
    "sanitize_metadata",
    "trace_scope",
    "use_trace",
]
