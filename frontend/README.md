# APIForge Demo UI（前端）

React 18 + Vite 5 + TypeScript。

**真实集成模式**：点击 Run Integration 会调用本地 Backend，真实执行 APIForge
Pipeline 并展示真实 PipelineResult（无前端定时器模拟）。

## Start Backend（必须先启动）

```bash
uv run uvicorn integration_agent.api_server.app:app --reload
```

- Backend：http://127.0.0.1:8000
- Swagger：http://127.0.0.1:8000/docs

## Start Frontend

```bash
cd frontend
npm install
npm run dev
```

- Frontend：http://localhost:5173
- Backend 地址可用 `VITE_API_BASE_URL` 覆盖（见 `.env.example`，默认 http://127.0.0.1:8000）

## 页面内容

- Project 输入区（默认值为 Backend 白名单内的合法路径：`openapi/petstore.yaml` / `demo_project`）+ Run Integration
- DeepSeek Repair 开关（默认 OFF；ON 时 Backend 使用服务端 DEEPSEEK_API_KEY，前端不接收/保存任何 Key）
- Demo Mode 开关（默认 OFF）：请求里只多带一个 `demo_mode: true`，由 Backend 在**固定 Demo 组合**
  （`openapi/petstore.yaml` + `demo_project`）下注入确定性失败，用于展示
  「初始测试失败 → DeepSeek 修复 → 复测通过 → Patch」完整闭环。
  其他组合会被 Backend 拒绝（`DEMO_MODE_NOT_ALLOWED`）。前端**不发送**任何文件路径或代码片段，
  注入内容全部由 Backend 硬编码（[`api_server/demo.py`](../src/integration_agent/api_server/demo.py)）
- Agent Progress：8 个阶段的状态由真实 PipelineResult 一次性推导（无假实时）
- Integration Plan：真实 `plan` 数据（plan 缺失时显示 unavailable）
- Final Result：PASSED / TESTS FAILED / ERROR 三态 + 真实测试计数与修复轮数
- Patch / Diff：真实 `patch`（unified diff 行着色；modify 片段无 diff 时如实展示结构化信息）

## 构建 / 类型检查

```bash
npm run build      # tsc 类型检查 + vite 生产构建
npm run preview    # 预览生产构建
```
