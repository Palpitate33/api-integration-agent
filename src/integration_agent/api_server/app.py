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
    - LLM 开关（use_llm / use_llm_repair / use_llm_planner / use_agent_planner）
      默认全关；开启后仍不提供任何"用户自带 Key"的入口。
    - Agent Planner（use_agent_planner）与 DeepSeek Planner（use_llm_planner）
      互斥，同时开启时只运行 ToolUsingPlanner：一次请求只会有一个 Planner。
    - CORS 只允许本地开发前端 origin（不允许 "*"）。
    - demo_mode（默认 false）只对固定 Demo 组合生效，注入内容在 demo.py 中硬编码，
      用户无法指定目标文件或替换内容；只改内存产物，不触碰真实仓库。

启动：
    uv run uvicorn integration_agent.api_server.app:app --reload
    Swagger UI：http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from integration_agent.agent import DeepSeekPlanner, ToolUsingPlanner
from integration_agent.api_server.demo import DEMO_ONLY_MESSAGE, SabotagedGenerator, is_demo_target
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
    if payload.demo_mode:
        # Demo-only：只允许固定组合，注入确定性失败；普通请求完全不经过这里。
        # 注入目标是硬编码的，不接受任何用户输入（文件路径 / 替换内容都由服务端写死）。
        if not is_demo_target(spec_path, project_path):
            raise ApiError(400, "DEMO_MODE_NOT_ALLOWED", DEMO_ONLY_MESSAGE)
        kwargs["code_generator"] = SabotagedGenerator()

    # LLM 开关互相独立：开启哪个就替换哪个组件，未开启的仍是确定性实现。
    # use_llm 保持既有语义（只影响修复），避免破坏已有调用方。
    use_repair = payload.use_llm or payload.use_llm_repair
    if use_repair or payload.use_llm_planner or payload.use_agent_planner:

        def build_client(*, json_mode: bool) -> DeepSeekLLMClient:
            """构造 DeepSeek 客户端；未配置 Key 时转成结构化 400。

            错误消息只包含环境变量名，不包含 Key（DeepSeek 客户端已保证）。
            """
            try:
                return DeepSeekLLMClient(json_mode=json_mode)
            except DeepSeekConfigError as exc:
                raise ApiError(400, "LLM_NOT_CONFIGURED", str(exc)) from exc

        # 修复侧与 DeepSeek Planner 要求响应本身就是一段可解析的 JSON，因此
        # json_mode=True；客户端无状态（只是配置 + generate/chat），两者共用一个实例。
        json_client = build_client(json_mode=True)

        if use_repair:
            kwargs["repair_applier"] = StructuredLLMRepairApplier(json_client)
        # Planner 选择顺序：Agent Planner > DeepSeek Planner > DeterministicPlanner。
        # 三者**互斥**，只往 kwargs 里放一个 planner：两个开关同时打开时也只构造
        # ToolUsingPlanner，绝不会让两个 Planner 先后各跑一遍。
        if payload.use_agent_planner:
            # Agent Planner 复用同一份已解析的 api / 已扫描的 project（Pipeline 只
            # 构造一次 PlannerState），只读工具仅做补充观察，不重新解析、不重扫仓库。
            #
            # 这里必须换一个 json_mode=False 的客户端：Agent Loop 走的是 tool calling，
            # 而真实 API 会拒绝 response_format=json_object 与 tools 同时出现
            # （HTTP 400 "Prompt must contain the word 'json' to use 'response_format'
            # of type 'json_object'"）；何况 json_object 会把整轮回答限制成一个 JSON
            # 对象，与 tool_calls 这条输出通道互相打架。
            kwargs["planner"] = ToolUsingPlanner(build_client(json_mode=False))
        elif payload.use_llm_planner:
            # 规划失败不做静默 fallback：Planner 抛异常 → Pipeline 记录
            # failed_stage="plan" 并以 status="error" 如实返回，由调用方决定是否重试。
            # 规划阶段失败**不进入** Repair Loop：Repair 只处理"代码生成之后的测试失败"。
            kwargs["planner"] = DeepSeekPlanner(json_client)

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
