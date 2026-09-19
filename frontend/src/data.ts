/**
 * 前端固定常量：阶段定义与 Demo 默认输入。
 * 运行数据全部来自真实 Backend（PipelineResult），不再使用模拟数据。
 */

export type StepStatus = "pending" | "running" | "success" | "failed";

export interface Step {
  id: string;
  label: string;
  detail: string;
}

export const STEPS: Step[] = [
  { id: "api", label: "API Understanding", detail: "解析 OpenAPI spec" },
  { id: "repo", label: "Repository Understanding", detail: "扫描目标仓库" },
  { id: "plan", label: "Integration Planning", detail: "生成 IntegrationPlan" },
  { id: "codegen", label: "Code Generation", detail: "生成集成代码" },
  { id: "test", label: "Test Runner", detail: "运行生成测试" },
  { id: "repair", label: "DeepSeek Repair", detail: "失败分析 / 修复" },
  { id: "validate", label: "Final Validation", detail: "复测最终产物" },
  { id: "patch", label: "Patch Generation", detail: "生成 unified diff" },
];

/** 与 Backend 路径白名单（examples/ 内相对路径）对应的 Demo 默认值。 */
export const DEFAULT_INPUT = {
  apiSpec: "openapi/petstore.yaml",
  projectPath: "demo_project",
  request: "Integrate the API into the existing project",
  maxIterations: 3,
};
