/**
 * Backend API client：只负责 HTTP 调用与错误转换，不含任何 Agent 逻辑。
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export interface RunIntegrationRequest {
  api_spec: string;
  project_path: string;
  request: string;
  max_iterations: number;
  use_llm: boolean;
}

// ---------------------------------------------------------------- 后端 PipelineResult 的
// 前端实际使用到的字段（与 Backend Pydantic 模型对应，不做机械全量复制）。

export interface TestResult {
  status: string;
  exit_code: number | null;
  passed: number;
  failed: number;
  errors: number;
  skipped: number;
}

export interface IntegrationPlan {
  integration_strategy: {
    approach: string;
    http_client: string;
    summary: string;
  };
  authentication: {
    scheme: string;
    header_name: string | null;
    required_env_vars: string[];
  } | null;
  endpoints: { method: string; path: string }[];
  error_handling: { rules: { status_code: string; retryable: boolean }[] };
  testing_strategy: {
    unit_tests: unknown[];
    integration_tests: unknown[];
    contract_tests: unknown[];
  };
  risks: { description: string; severity: string }[];
}

export interface PatchFile {
  path: string;
  action: "create" | "modify";
  diff: string | null;
  diff_available: boolean;
  summary: string;
}

export interface PatchResult {
  files: PatchFile[];
  dependency_changes: { name: string; version: string | null }[];
  unified_diff: string;
  files_created: number;
  files_modified: number;
  files_changed: number;
  warnings: string[];
  summary: {
    total_files: number;
    created: number;
    modified: number;
    dependencies: number;
  };
}

export interface RepairLoopResult {
  status: string;
  iterations: number;
  test_result: TestResult | null;
  warnings: string[];
  repair_plans: {
    failure_category: string;
    analysis: string;
    actions: { file: string; action: string; reason: string }[];
  }[];
}

export interface PipelineResult {
  status: "passed" | "tests_failed" | "error";
  api: { name: string; version: string; base_url: string | null } | null;
  project: { name: string } | null;
  plan: IntegrationPlan | null;
  artifacts: { files: { path: string; action: string }[] } | null;
  repair_loop_result: RepairLoopResult | null;
  patch: PatchResult | null;
  failed_stage: string | null;
  warnings: string[];
  error: string | null;
}

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly httpStatus: number
  ) {
    super(message);
  }
}

export async function runIntegration(
  payload: RunIntegrationRequest
): Promise<PipelineResult> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/integrations/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(
      "BACKEND_UNAVAILABLE",
      "Backend unavailable — 请确认 Backend 已启动（uv run uvicorn integration_agent.api_server.app:app）",
      0
    );
  }
  if (!response.ok) {
    let code = "REQUEST_ERROR";
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const body = (await response.json()) as { error?: { code?: string; message?: string } };
      if (body?.error?.message) {
        message = body.error.message;
        code = body.error.code ?? code;
      }
    } catch {
      // 保留默认信息，不把复杂响应对象直接抛给 UI
    }
    throw new ApiError(code, message, response.status);
  }
  return (await response.json()) as PipelineResult;
}
