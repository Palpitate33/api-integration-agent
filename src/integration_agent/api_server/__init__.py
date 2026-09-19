"""Backend API 阶段：为 APIForge Pipeline 提供最小、清晰的 HTTP Adapter 层。

主要入口：
    app —— FastAPI 应用（uvicorn integration_agent.api_server.app:app --reload）
    GET  /health                  健康检查
    POST /api/integrations/run    调用现有 run_pipeline 并返回 PipelineResult
    GET  /docs                    Swagger UI（自动生成）

约束：只做校验 + 调用 + 序列化；不复制 Pipeline 逻辑、不输出 API Key /
traceback、路径访问限制在 examples/ 内。
"""

from integration_agent.api_server.app import app
from integration_agent.api_server.models import (
    ErrorDetail,
    ErrorResponse,
    IntegrationRunRequest,
)

__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "IntegrationRunRequest",
    "app",
]
