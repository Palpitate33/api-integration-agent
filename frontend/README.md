# APIForge Demo UI

APIForge 的浏览器演示前端：把 APIForge Pipeline 的真实执行结果可视化出来。

**这里没有模拟数据。** 点击 Run Integration 会真的发起 HTTP 请求调用本地 Backend，真的执行
APIForge Pipeline（含真实的 pytest 执行与真实的 DeepSeek 调用），页面上的每一个数字都来自
后端返回的 `PipelineResult`。

---

## 技术栈

| 项 | 版本 / 说明 |
|---|---|
| React | 18.3 |
| Vite | 5.4（dev server 固定 5173 端口） |
| TypeScript | 5.6，`tsc` 严格类型检查 |
| 依赖 | **只有 react / react-dom** —— HTTP 用原生 `fetch`，不引入 axios / 状态管理库 |

源码共 6 个文件：

```text
src/
├── App.tsx           # 主界面 + 由 PipelineResult 推导 8 个阶段状态
├── api.ts            # Backend HTTP client（类型定义 + 错误转换）
├── data.ts           # 阶段定义（STEPS）与默认输入（DEFAULT_INPUT）
├── main.tsx          # 入口
├── styles.css        # 样式
└── vite-env.d.ts     # Vite 环境变量类型声明
```

---

## 与 FastAPI Backend 的关系

前端**只做展示**，不含任何 Agent 逻辑：

```text
React（本目录）
   │  fetch POST /api/integrations/run   ← 只有这一个业务请求
   ▼
FastAPI（../src/integration_agent/api_server/）
   │  run_pipeline(...)
   ▼
APIForge Pipeline → PipelineResult
   │
   └─► 原样返回给前端渲染
```

- 前端不解析 OpenAPI、不生成代码、不执行测试、不接触 LLM。
- 前端**不接触 API Key**：只发送布尔开关 `use_llm`，Key 由 Backend 从服务端环境变量读取。
  Key 不存在时 Backend 返回结构化错误 `LLM_NOT_CONFIGURED`。
- 输入路径必须落在 Backend 白名单内（`examples/` 下的相对路径），否则 Backend 返回 400。

**Backend 必须先启动**，否则页面会显示 `BACKEND_UNAVAILABLE`。

---

## 启动方式

### 1. 先启动 Backend（根目录）

```powershell
uv run uvicorn integration_agent.api_server.app:app --reload
```

- Backend：<http://127.0.0.1:8000>
- Swagger UI：<http://127.0.0.1:8000/docs>

### 2. 再启动前端（本目录）

```powershell
npm install
npm run dev
```

打开 <http://localhost:5173>。

### 3. 生产构建 / 类型检查

```powershell
npm run build      # tsc 类型检查 + vite 生产构建
npm run preview    # 预览生产构建产物
```

---

## VITE_API_BASE_URL

Backend 地址通过 Vite 环境变量覆盖，默认 `http://127.0.0.1:8000`：

```powershell
Copy-Item .env.example .env
# 然后按需修改 .env 中的 VITE_API_BASE_URL
```

```ini
VITE_API_BASE_URL=http://127.0.0.1:8000
```

- 该变量在构建时被内联进产物，**不要在里放任何密钥**。
- `.env` 由仓库根目录的 `.gitignore` 忽略（规则 `.env` 同时覆盖 `frontend/.env`），不会进入 Git。
- 修改后需要重启 `npm run dev` 才会生效。

---

## 页面功能

### 输入区

| 控件 | 说明 |
|---|---|
| API Specification | OpenAPI spec 路径，相对 `examples/`，默认 `openapi/petstore.yaml` |
| Existing Repository | 目标项目路径，相对 `examples/`，默认 `demo_project` |
| **DeepSeek Repair** 开关 | 默认 **OFF**。ON 时请求体带 `use_llm: true`，Backend 从 `DEEPSEEK_API_KEY` 读取 Key 并启用 LLM 修复。前端不接收、不保存、不显示任何 Key。 |
| **Demo Mode** 开关 | 默认 **OFF**。详见下一节。 |

### Agent Progress

8 个阶段的图标由真实 `PipelineResult` 一次性推导，**不做假实时**：

```text
API Understanding → Repository Understanding → Integration Planning → Code Generation
→ Test Runner → DeepSeek Repair → Final Validation → Patch Generation
```

运行期间只显示 `Processing…`；请求返回后一次性把每个阶段标成 ✓ / ✗ 并附上真实细节
（例如 `初始测试失败（assertion）→ 进入修复`、`DeepSeek 修复 1 次（loop=passed）`）。

### Integration Plan

渲染真实的 `IntegrationPlan`：HTTP client 选型、认证方式、端点列表、错误处理规则、
可重试状态码、测试文件数量、风险数量。plan 缺失时显示 `Integration plan unavailable`。

### Final Result

`PASSED` / `TESTS FAILED` / `ERROR` 三态 + 真实测试计数（passed / failed / errors）、
真实修复轮数、变更文件数、Patch 是否可用。Pipeline 返回的 warnings 与 error 如实展示。

### Patch / Diff

真实 `PatchResult`：created / modified 文件清单、依赖变更、unified diff 行级着色。
若 modify 片段因缺少原始文件内容而无法生成 diff，会**如实说明**并展示结构化信息，
不伪造 diff。

---

## Demo Mode

Demo Mode 的唯一目的：让「初始测试失败 → DeepSeek 修复 → 复测通过 → Patch 可用」这条闭环
**每次演示都能稳定复现**。

前端在这件事上只做一件事：请求体里多带一个 `demo_mode: true`。

```ts
// src/api.ts
{ api_spec, project_path, request, max_iterations, use_llm, demo_mode }
```

关键点：

- 前端**不发送**任何文件路径、代码片段或替换内容 —— 注入什么完全由 Backend 决定。
- 注入的目标与内容硬编码在
  [`api_server/demo.py`](../src/integration_agent/api_server/demo.py)，
  且只对**唯一固定组合**（`openapi/petstore.yaml` + `demo_project`）生效。
- 其他组合会被 Backend 拒绝（HTTP 400，code `DEMO_MODE_NOT_ALLOWED`），
  页面错误框会如实展示 Backend 返回的拒绝原因。
- 注入只改内存中的产物，**不修改真实仓库**。
- 默认关闭；不打开它时，请求与普通模式完全一致。

**完整演示步骤**：打开 DeepSeek Repair + Demo Mode → Run Integration → 预期看到
初始测试失败、DeepSeek 修复 1 次、复测 `10 passed`、Patch `9 files`。

> 需要先在 Backend 侧配置 `DEEPSEEK_API_KEY` 并重启 Backend。只打开 Demo Mode 而不打开
> DeepSeek Repair 时，确定性的 RepairApplier 无法修复注入的断言，结果是
> `TESTS FAILED`（`loop=no_progress`）—— 这是预期行为，不是 Bug。

---

## 已知边界

- **无实时流式反馈**：刻意不做 WebSocket / SSE / 轮询 / 任务队列，Pipeline 是同步执行、
  一次性返回。长任务期间页面只显示 `Processing…`。
- **单次运行**：页面只保留最近一次运行结果，刷新即丢失。
- **无用户输入 API Key 的入口**：这是设计约束，不是缺失。
- **输入路径受白名单限制**：浏览器无法请求 `examples/` 之外的任何本机路径。
