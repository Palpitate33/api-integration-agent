import { useState } from "react";
import {
  ApiError,
  runIntegration,
  type IntegrationPlan,
  type PipelineResult,
  type TestResult,
} from "./api";
import { DEFAULT_INPUT, STEPS, type StepStatus } from "./data";

const STATUS_GLYPH: Record<StepStatus, string> = {
  pending: "·",
  running: "▶",
  success: "✓",
  failed: "✗",
};

interface StepState {
  status: StepStatus;
  detail: string;
}

function initialSteps(): StepState[] {
  return STEPS.map((step) => ({ status: "pending", detail: step.detail }));
}

const FAILED_STAGE_INDEX: Record<string, number> = {
  parse: 0,
  scan: 1,
  plan: 2,
  generate: 3,
};

function testDetail(test: TestResult): string {
  return `${test.passed} passed / ${test.failed} failed / ${test.errors} errors`;
}

/** 根据真实 PipelineResult 一次性推导 8 个阶段的状态（不做假实时）。 */
function deriveSteps(
  result: PipelineResult,
  useLlm: boolean,
  demoMode: boolean
): StepState[] {
  const steps = initialSteps();
  const loop = result.repair_loop_result;
  const failedIndex =
    result.failed_stage !== null
      ? FAILED_STAGE_INDEX[result.failed_stage]
      : undefined;

  if (result.status === "error" && failedIndex !== undefined) {
    for (let i = 0; i < 4; i += 1) {
      if (i < failedIndex) {
        steps[i].status = "success";
      } else if (i === failedIndex) {
        steps[i].status = "failed";
        steps[i].detail = result.error ?? "阶段失败";
      }
    }
    return steps;
  }
  for (let i = 0; i < 4; i += 1) steps[i].status = "success";
  if (!loop) return steps;

  // 5. Test Runner：repair_plans 为空 = 初始测试一次通过
  const final = loop.test_result;
  if (loop.repair_plans.length === 0) {
    steps[4].status = final?.status === "passed" ? "success" : "failed";
    steps[4].detail = final ? `初始测试：${testDetail(final)}` : "初始测试";
  } else {
    steps[4].status = "failed";
    const origin = demoMode ? "Demo Mode 注入的确定性失败" : "初始测试失败";
    steps[4].detail = `${origin}（${loop.repair_plans[0].failure_category}）→ 进入修复`;
  }

  // 6. DeepSeek / Repair
  if (loop.iterations > 0) {
    steps[5].status = loop.status === "passed" ? "success" : "failed";
    const engine = useLlm ? "DeepSeek" : "deterministic";
    steps[5].detail = `${engine} 修复 ${loop.iterations} 次（loop=${loop.status}）`;
  } else {
    steps[5].status = "success";
    steps[5].detail = "无需修复（初始测试已通过）";
  }

  // 7. Final Validation
  if (final) {
    steps[6].status = final.status === "passed" ? "success" : "failed";
    steps[6].detail = `复测：${testDetail(final)}`;
  }

  // 8. Patch Generation
  if (result.patch) {
    steps[7].status = "success";
    const patch = result.patch;
    steps[7].detail = `Patch available（${patch.summary.total_files} files，${
      patch.unified_diff ? "diff 已生成" : "无可用 diff"
    }）`;
  } else {
    steps[7].status = "failed";
    steps[7].detail = "Patch unavailable";
  }
  return steps;
}

function diffLineClass(line: string): string | undefined {
  if (line.startsWith("+") && !line.startsWith("+++")) return "diff-line-add";
  if (line.startsWith("-") && !line.startsWith("---")) return "diff-line-del";
  if (line.startsWith("@@") || line.startsWith("---") || line.startsWith("+++")) {
    return "diff-line-meta";
  }
  return undefined;
}

export default function App() {
  const [apiSpec, setApiSpec] = useState(DEFAULT_INPUT.apiSpec);
  const [projectPath, setProjectPath] = useState(DEFAULT_INPUT.projectPath);
  const [useLlm, setUseLlm] = useState(false);
  const [demoMode, setDemoMode] = useState(DEFAULT_INPUT.demoMode);
  const [phase, setPhase] = useState<"idle" | "running" | "done">("idle");
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [patchOpen, setPatchOpen] = useState(false);

  const steps = result ? deriveSteps(result, useLlm, demoMode) : initialSteps();

  async function handleRun() {
    setPhase("running");
    setRunError(null);
    setResult(null);
    setPatchOpen(false);
    try {
      const data = await runIntegration({
        api_spec: apiSpec,
        project_path: projectPath,
        request: DEFAULT_INPUT.request,
        max_iterations: DEFAULT_INPUT.maxIterations,
        use_llm: useLlm,
        demo_mode: demoMode,
      });
      setResult(data);
      setPhase("done");
    } catch (error) {
      setRunError(
        error instanceof ApiError ? error.message : "请求失败，请重试"
      );
      setPhase("idle");
    }
  }

  const loop = result?.repair_loop_result;
  const patch = result?.patch;
  const statusLabel =
    result?.status === "passed"
      ? "PASSED"
      : result?.status === "tests_failed"
        ? "TESTS FAILED"
        : "ERROR";

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1 className="title">APIForge</h1>
          <p className="subtitle">Autonomous API Integration Agent</p>
        </div>
        <span className="demo-badge">Live Pipeline</span>
      </header>

      <div className="demo-notice">
        真实集成模式：点击 Run Integration 会调用本地 Backend（
        http://127.0.0.1:8000）真实执行 APIForge Pipeline 并展示真实结果。
        需先启动 Backend：uv run uvicorn integration_agent.api_server.app:app --reload。
      </div>

      <main className="layout">
        <section className="column">
          <div className="panel">
            <h2 className="panel-title">Project</h2>
            <div className="field">
              <span className="field-label">API Specification（相对 examples/）</span>
              <input
                className="text-input"
                value={apiSpec}
                onChange={(event) => setApiSpec(event.target.value)}
                disabled={phase === "running"}
              />
            </div>
            <div className="field">
              <span className="field-label">Existing Repository（相对 examples/）</span>
              <input
                className="text-input"
                value={projectPath}
                onChange={(event) => setProjectPath(event.target.value)}
                disabled={phase === "running"}
              />
            </div>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={useLlm}
                onChange={(event) => setUseLlm(event.target.checked)}
                disabled={phase === "running"}
              />
              <span>
                DeepSeek Repair（{useLlm ? "ON" : "OFF"}，Key 由 Backend 环境变量提供）
              </span>
            </label>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={demoMode}
                onChange={(event) => setDemoMode(event.target.checked)}
                disabled={phase === "running"}
              />
              <span>
                Demo Mode（{demoMode ? "ON" : "OFF"}，Backend 注入确定性失败 —— 仅对默认 Demo
                组合生效）
              </span>
            </label>
            {demoMode && (
              <div className="demo-notice inline">
                Demo Mode 只允许 api_spec=openapi/petstore.yaml + project_path=demo_project，
                其他组合会被 Backend 拒绝（DEMO_MODE_NOT_ALLOWED）。注入内容由 Backend 硬编码，
                前端不发送任何文件路径或代码片段。
              </div>
            )}
            <button
              className="run-button"
              onClick={handleRun}
              disabled={phase === "running"}
            >
              {phase === "running" ? "Processing…" : "Run Integration"}
            </button>
            {runError && <div className="error-box">{runError}</div>}
          </div>

          <div className="panel">
            <h2 className="panel-title">Agent Progress</h2>
            {phase === "running" && (
              <div className="processing-note">
                Processing… 真实 Pipeline 执行中（无前端模拟），完成后一次性更新阶段状态。
              </div>
            )}
            <ol className="steps">
              {steps.map((step, index) => (
                <li key={STEPS[index].id} className={`step step-${step.status}`}>
                  <span className="step-glyph">{STATUS_GLYPH[step.status]}</span>
                  <div className="step-body">
                    <span className="step-label">{STEPS[index].label}</span>
                    <span className="step-detail">{step.detail}</span>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section className="column">
          <div className="panel">
            <h2 className="panel-title">Integration Plan</h2>
            <PlanPanel plan={result?.plan ?? null} />
          </div>

          <div className="panel result-panel">
            <h2 className="panel-title">Final Result</h2>
            {result ? (
              <>
                <div className="result-grid">
                  <div className="result-item">
                    <span className="result-label">Status</span>
                    <span
                      className={`result-value ${
                        result.status === "passed" ? "result-passed" : "result-failed"
                      }`}
                    >
                      {statusLabel}
                    </span>
                  </div>
                  <div className="result-item">
                    <span className="result-label">Repair iterations</span>
                    <span className="result-value">{loop?.iterations ?? "—"}</span>
                  </div>
                  <div className="result-item">
                    <span className="result-label">Tests</span>
                    <span className="result-value">
                      {loop?.test_result ? testDetail(loop.test_result) : "—"}
                    </span>
                  </div>
                  <div className="result-item">
                    <span className="result-label">Files changed</span>
                    <span className="result-value">
                      {patch ? patch.summary.total_files : "—"}
                    </span>
                  </div>
                  <div className="result-item">
                    <span className="result-label">Patch</span>
                    <span
                      className={`result-value ${
                        patch ? "result-passed" : "result-failed"
                      }`}
                    >
                      {patch ? "Available" : "Unavailable"}
                    </span>
                  </div>
                </div>
                {result.error && (
                  <div className="error-box">
                    {result.failed_stage
                      ? `${result.failed_stage}: ${result.error}`
                      : result.error}
                  </div>
                )}
                {result.warnings.length > 0 && (
                  <div className="warnings-box">
                    {result.warnings.slice(0, 5).map((warning, index) => (
                      <div key={index}>⚠ {warning}</div>
                    ))}
                  </div>
                )}
                <button
                  className="run-button secondary"
                  onClick={() => setPatchOpen(true)}
                >
                  View Patch
                </button>
              </>
            ) : (
              <p className="result-placeholder">
                点击 Run Integration 调用真实 Backend，完成后在此展示 PipelineResult。
              </p>
            )}
          </div>
        </section>
      </main>

      {patchOpen && (
        <div className="modal-overlay" onClick={() => setPatchOpen(false)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h2 className="panel-title">Patch / Diff</h2>
              <button className="modal-close" onClick={() => setPatchOpen(false)}>
                ✕
              </button>
            </div>
            {patch ? (
              <>
                <p className="patch-note">真实 PatchResult（由 Backend 生成）。</p>
                <div className="patch-meta">
                  <div className="patch-group">
                    <h3>Created files</h3>
                    <ul>
                      {patch.files
                        .filter((item) => item.action === "create")
                        .map((item) => (
                          <li key={item.path}>
                            <code>{item.path}</code>
                          </li>
                        ))}
                    </ul>
                  </div>
                  <div className="patch-group">
                    <h3>Modified files</h3>
                    <ul>
                      {patch.files
                        .filter((item) => item.action === "modify")
                        .map((item) => (
                          <li key={item.path}>
                            <code>{item.path}</code>
                          </li>
                        ))}
                    </ul>
                  </div>
                  <div className="patch-group">
                    <h3>Dependencies</h3>
                    <ul>
                      {patch.dependency_changes.map((item) => (
                        <li key={item.name}>
                          <code>
                            {item.name}
                            {item.version ?? ""}
                          </code>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
                {patch.files.some((item) => item.diff_available) ? (
                  patch.files
                    .filter((item) => item.diff_available && item.diff)
                    .map((item) => (
                      <div key={item.path} className="diff-block">
                        <div className="diff-label">
                          {item.path} ({item.action})
                        </div>
                        <pre className="diff">
                          {(item.diff ?? "").split("\n").map((line, index) => (
                            <span key={index} className={diffLineClass(line)}>
                              {line}
                              {"\n"}
                            </span>
                          ))}
                        </pre>
                      </div>
                    ))
                ) : (
                  <p className="patch-note">
                    unified diff 不可用（modify 片段无法安全推导完整内容），以下为结构化信息。
                  </p>
                )}
                {patch.warnings.length > 0 && (
                  <div className="warnings-box">
                    {patch.warnings.slice(0, 5).map((warning, index) => (
                      <div key={index}>⚠ {warning}</div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="patch-note">
                Patch unavailable
                {result && result.warnings.length > 0
                  ? `（${result.warnings.slice(0, 3).join("；")}）`
                  : ""}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function PlanPanel({ plan }: { plan: IntegrationPlan | null }) {
  if (!plan) {
    return <p className="result-placeholder">Integration plan unavailable</p>;
  }
  const retryRules = plan.error_handling.rules
    .filter((rule) => rule.retryable)
    .map((rule) => rule.status_code);
  const testCount =
    plan.testing_strategy.unit_tests.length +
    plan.testing_strategy.integration_tests.length +
    plan.testing_strategy.contract_tests.length;
  return (
    <dl className="plan">
      <div className="plan-row">
        <dt>Client</dt>
        <dd>
          {plan.integration_strategy.http_client}（{plan.integration_strategy.approach}）
        </dd>
      </div>
      <div className="plan-row">
        <dt>Authentication</dt>
        <dd>{plan.authentication ? plan.authentication.scheme : "无认证"}</dd>
      </div>
      <div className="plan-row">
        <dt>Endpoints</dt>
        <dd>
          {plan.endpoints.slice(0, 8).map((endpoint) => (
            <span key={`${endpoint.method} ${endpoint.path}`} className="chip">
              {endpoint.method} {endpoint.path}
            </span>
          ))}
          {plan.endpoints.length > 8 && (
            <span className="chip">+{plan.endpoints.length - 8}</span>
          )}
        </dd>
      </div>
      <div className="plan-row">
        <dt>Error Handling</dt>
        <dd>
          {plan.error_handling.rules.map((rule) => (
            <span key={rule.status_code} className="chip">
              {rule.status_code}
            </span>
          ))}
        </dd>
      </div>
      <div className="plan-row">
        <dt>Retry</dt>
        <dd>{retryRules.length > 0 ? retryRules.join(" / ") : "无"}</dd>
      </div>
      <div className="plan-row">
        <dt>Testing</dt>
        <dd>{testCount} 个测试文件（unit/integration/contract）</dd>
      </div>
      <div className="plan-row">
        <dt>Risks</dt>
        <dd>{plan.risks.length} 项</dd>
      </div>
    </dl>
  );
}
