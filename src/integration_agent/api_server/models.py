"""Backend API 的请求/响应契约（HTTP Adapter 层）。"""

from pydantic import BaseModel, Field


class IntegrationRunRequest(BaseModel):
    """POST /api/integrations/run 的请求体。

    api_spec / project_path 只接受相对于 examples/ 的相对路径，
    由 paths.resolve_allowed 做安全校验（拒绝绝对路径 / 越界 / 符号链接逃逸）。
    """

    api_spec: str = Field(min_length=1, max_length=500)  # OpenAPI spec 路径（相对 examples/）
    project_path: str = Field(min_length=1, max_length=500)  # 目标仓库路径（相对 examples/）
    request: str = Field(default="", max_length=2000)  # 用户需求描述，透传 Pipeline
    max_iterations: int = Field(default=3, ge=1, le=10)
    use_llm: bool = False  # 是否启用 DeepSeek LLM Repair（Key 走服务端环境变量）


class ErrorDetail(BaseModel):
    """结构化错误信息。"""

    code: str  # 如 INVALID_PROJECT_PATH / PIPELINE_ERROR
    message: str


class ErrorResponse(BaseModel):
    """统一错误响应结构。"""

    error: ErrorDetail
