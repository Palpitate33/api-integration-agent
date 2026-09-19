"""Pipeline 阶段：把 APIForge 各阶段串成 End-to-End 工作流。

主要 API：
    run_pipeline(spec, repo, ...)  —— 便捷入口（默认全套确定性组件）
    IntegrationPipeline()          —— 可配置实现（依赖注入）
    Pipeline                       —— Protocol，未来扩展共用同一契约

输出契约：
    PipelineResult

流程：
    OpenAPI Spec → API Understanding → Repository Understanding
    → Integration Planning → Code Generation → Repair Loop → PipelineResult

约束：只编排、不复制已有模块逻辑；不修改真实 repository、不执行 git、
不安装依赖、不读取/输出 API Key；默认离线确定性执行。
"""

from integration_agent.pipeline.models import PipelineResult
from integration_agent.pipeline.orchestrator import (
    IntegrationPipeline,
    Pipeline,
    run_pipeline,
)

__all__ = [
    "IntegrationPipeline",
    "Pipeline",
    "PipelineResult",
    "run_pipeline",
]
