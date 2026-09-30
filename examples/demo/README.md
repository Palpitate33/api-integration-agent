# APIForge 面试 Demo

一个**可复现、确定性、离线**的端到端演示：把一份 OpenAPI 文档集成进一个
模拟的既有 Python 项目，五分钟内看到完整 Pipeline 跑通。

## 一、输入是什么

| 输入 | 文件 | 内容 |
|---|---|---|
| OpenAPI Spec | [openapi.yaml](openapi.yaml) | User Management API：`GET /users/{id}`、`POST /users`、`GET /users/{id}/profile`，Bearer Token（JWT）认证 |
| Existing Repository | [demo_project/](demo_project/) | 模拟的既有项目：`UserService`（service.py）、`User` / `Profile` 领域模型（models.py）、声明了 `httpx` + `pydantic` 的 pyproject.toml |

`demo_project` 当前只操作本地内存数据，**还没有接任何第三方 API**——本次
集成要补上的正是这部分。

## 二、怎么运行

在**仓库根目录**（需要先 `uv sync` 装好依赖）：

```bash
uv run python examples/demo/run_demo.py
```

全程不需要 API Key、不需要网络、不调用 LLM——pytest 用 `httpx.MockTransport`
打桩，Token 从环境变量读取的路径只是被验证，不会被真实使用。

## 三、Pipeline 怎么跑（5 个阶段）

`run_demo.py` 只调用 `integration_agent.pipeline.run_pipeline`，所有阶段
逻辑都是项目已有能力，Demo 本身不复刻任何一步：

```
OpenAPI + Existing Repository
        ↓
[1] API Understanding        parse_openapi      → APIInfo（3 个端点、http-bearer 认证）
[2] Repository Understanding scan_repository    → ProjectStructure（2 个 Python 文件、2 个依赖）
[3] Integration Planning     DeterministicPlanner → IntegrationPlan（发现既有业务模块 models.py / service.py）
[4] Code Generation          DeterministicCodeGenerator → GeneratedArtifacts（5 源文件 + 3 测试文件 + 2 修改片段）
[5] Test Running             DeterministicTestRunner → TestResult（MockTransport 打桩，不发起真实网络请求）
```

值得注意的两点：

1. **Repository Understanding 发现了既有业务模块**。Planner 从 API 名称与
   tags 提取领域关键词（`user` / `management` / `users`），经 `search_code`
   检索到 `models.py` 与 `service.py`，把它们列入 `files_to_modify`——生成
   的修改片段是「导入新客户端并接入既有用户处理流程」的结构化建议。
2. **Test Runner 不修改真实仓库**。生成文件落盘与 pytest 执行都发生在
   Pipeline 的临时工作区内，`demo_project/` 运行前后逐字节不变。

## 四、最终生成什么

控制台按阶段打印结论行与细节；结构化结果写入 [output/](output/)（已在
`.gitignore` 中，每次运行重新生成）：

| 文件 | 内容 |
|---|---|
| `integration_plan.json` | 完整 `IntegrationPlan`：端点选择、认证方案、错误处理、`files_to_create` / `files_to_modify`、风险与假设 |
| `generated_artifacts.json` | 全部生成产物：client / models / exceptions / config 完整源码 + 测试源码 + modify 片段 |
| `test_result.json` | 最终 `TestResult`：passed / failed / errors / skipped 计数与耗时 |
| `trace.json` | 全程 Execution Trace：每个阶段、每次工具调用的结构化事件 |

此外 Pipeline 还产出了建议性 Patch（10 files：8 create + 2 modify），由
`PatchResult` 表达、可渲染成 unified diff，但 Demo 不写盘。

## 五、测试结果是什么

Demo 运行一次，`Test Running` 阶段预期输出：

```
10 passed / 0 failed / 0 errors / 0 skipped
repair: 0 次修复（一次通过，无需修复）
```

三个测试文件分别验证：

- `test_user_management_client.py` —— 请求构造、路径参数替换、响应解析（单元测试）
- `test_user_management_integration.py` —— `MockTransport` 走完整链路：认证头注入、错误码映射（集成测试）
- `test_user_management_contract.py` —— 响应结构与 OpenAPI schema 一致（契约测试）

## 六、确定性

`integration_plan.json` 与 `generated_artifacts.json` **跨运行逐字节一致**
（无随机数、无时间戳、无 LLM）；`test_result.json` 的耗时与 `trace.json` 的
时间戳是真实执行的墙钟时间，属于有意保留的唯一波动。若测试未一次通过，
Pipeline 会自动进入 Repair Loop 重试（本 Demo 的测试设计为一次通过，因此
repair 次数为 0）。

## 七、与内置 Demo 的关系

项目还内置了 `python -m integration_agent demo`（fixture：petstore.yaml ×
demo_project），它是 CLI 展示层与测试的固定组合。本目录是**独立面试 Demo**：
自带输入文件、自产结构化输出，不依赖内置 fixture，也不被任何测试引用。
