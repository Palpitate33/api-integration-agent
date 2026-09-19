"""APIForge Backend API：HTTP Adapter 层。

职责边界（只做三件事）：
    1. 校验输入（路径安全校验 + Pydantic 参数校验）
    2. 调用现有 run_pipeline（不复制任何 Pipeline 内部逻辑）
    3. 序列化 PipelineResult 返回前端

安全边界：
    - project_path / api_spec 只允许 examples/ 内的相对路径（见 paths.py）。
    - 不提供任意文件读取 / shell 执行 / git 命令 API。
    - 错误响应结构化（code + message），绝不返回 Python traceback。
    - API Key 只从服务端环境变量 DEEPSEEK_API_KEY 读取，不进入请求/响应模型。
    - CORS 只允许本地开发前端 origin（不允许 "*"）。

启动：
    uv run uvicorn integration_agent.api_server.app:app --reload
    Swagger UI：http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from integration_agent.api_server.models import IntegrationRunRequest
from integration_agent.api_server.paths import resolve_allowed
from integration_agent.pipeline import PipelineResult, run_pipeline
from integration_agent.repair import (
    DeepSeekConfigError,
    DeepSeekLLMClient,
    StructuredLLMRepairApplier,
)

# 开发环境 CORS：只允许 D1 前端本地 dev server
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

app = FastAPI(
    title="APIForge",
    description="Autonomous API Integration Agent — Backend API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ApiError(Exception):
    """带结构化错误码的 HTTP 错误。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"请求参数校验失败：{exc.errors()[:3]}",
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # 统一 500：不向客户端返回 traceback 或任何敏感内容
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "PIPELINE_ERROR",
                "message": "Pipeline 执行失败（详情见服务端日志）",
            }
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/integrations/run", response_model=PipelineResult)
def run_integration(payload: IntegrationRunRequest) -> PipelineResult:
    spec_path, spec_error = resolve_allowed(payload.api_spec)
    if spec_error:
        raise ApiError(400, "INVALID_SPEC_PATH", spec_error)
    project_path, project_error = resolve_allowed(payload.project_path)
    if project_error:
        raise ApiError(400, "INVALID_PROJECT_PATH", project_error)

    if not spec_path.is_file():
        raise ApiError(404, "SPEC_NOT_FOUND", f"OpenAPI spec 不存在：{payload.api_spec}")
    if not project_path.is_dir():
        raise ApiError(404, "PROJECT_NOT_FOUND", f"项目目录不存在：{payload.project_path}")

    kwargs: dict = {}
    if payload.use_llm:
        try:
            kwargs["repair_applier"] = StructuredLLMRepairApplier(DeepSeekLLMClient(json_mode=True))
        except DeepSeekConfigError as exc:
            # 错误消息只包含环境变量名，不包含 Key（DeepSeek 客户端已保证）
            raise ApiError(400, "LLM_NOT_CONFIGURED", str(exc)) from exc

    try:
        return run_pipeline(
            str(spec_path),
            str(project_path),
            request=payload.request,
            max_iterations=payload.max_iterations,
            **kwargs,
        )
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 - Pipeline 异常统一转为结构化 500
        # 不向客户端返回 traceback / 异常原文（可能含敏感信息）
        raise ApiError(500, "PIPELINE_ERROR", "Pipeline 执行失败（详情见服务端日志）") from exc
