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
    # 三个 LLM 开关互相独立，全部默认关闭（Key 一律走服务端环境变量）：
    #   use_llm         —— 既有字段，语义保持不变：启用 DeepSeek LLM Repair
    #   use_llm_repair  —— 新增，与 use_llm 等价，便于显式表达"只开修复"
    #   use_llm_planner —— 新增：用 DeepSeekPlanner 取代 DeterministicPlanner
    use_llm: bool = False
    use_llm_repair: bool = False
    use_llm_planner: bool = False
    demo_mode: bool = False  # Demo-only：注入确定性测试失败（仅限固定 demo 组合）


class ErrorDetail(BaseModel):
    """结构化错误信息。"""

    code: str  # 如 INVALID_PROJECT_PATH / PIPELINE_ERROR
    message: str


class ErrorResponse(BaseModel):
    """统一错误响应结构。"""

    error: ErrorDetail
