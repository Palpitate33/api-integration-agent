# APIForge — Autonomous API Integration Agent

将第三方 OpenAPI API 自动集成到已有 Python 项目中的 AI Agent。

> 产品名：APIForge；Python 包名：`integration_agent`。

## 核心工作流

```
OpenAPI Spec
    ↓
API Parser            ← 已实现（API Understanding）
    ↓
Repository Scanner    ← 已实现（Repository Understanding）
    ↓
Integration Planner
    ↓
Code Generator
    ↓
Test Runner
    ↓
Repair Loop
    ↓
最终代码
```

## 模块划分

| 模块 | 对应阶段 | 职责 | 状态 |
|------|---------|------|------|
| `integration_agent.api` | API Parser | 解析 OpenAPI 3.x（YAML/JSON）为结构化对象 | ✅ 已实现 |
| `integration_agent.repository` | Repository Scanner | 扫描目标 Python 项目的结构、依赖，并按关键词搜索代码 | ✅ 已实现 |
| `integration_agent.agent` | Integration Planner | 生成集成方案（IntegrationPlan） | ✅ 已实现 |
| `integration_agent.generation` | Code Generator | 把 IntegrationPlan 转换为 GeneratedArtifacts | ✅ 已实现 |
| `integration_agent.validation` | Test Runner | 在隔离临时工作区运行测试，返回 TestResult | ✅ 已实现 |
| `integration_agent.repair` | Repair Loop | RepairPlanner / RepairApplier / RepairLoop / LLMRepairApplier / DeepSeekLLMClient | ✅ 已实现 |
| `integration_agent.tools` | 其他工具 | 预留 | 骨架 |

## 项目结构

```
.
├── pyproject.toml              # 项目配置（pytest / ruff 均已配置）
├── src/
│   └── integration_agent/
│       ├── api/                # API Parser（schema.py + parser.py）
│       ├── agent/              # Integration Planner
│       ├── repository/         # Repository Scanner（scanner.py + code_search.py）
│       ├── tools/              # Code Generator / Test Runner
│       ├── validation/         # Validation
│       └── repair/             # Repair Loop
├── tests/                      # 单元测试
├── examples/
│   ├── openapi/petstore.yaml   # OpenAPI 示例 fixture
│   └── demo_project/           # 集成测试目标项目
├── .env.example
└── .gitignore
```

## DeepSeek LLM 配置（LLM Repair Applier）

DeepSeek 接入通过环境变量配置（参见 `.env.example`，切勿提交真实 Key）：

```bash
# 1. 设置 API Key（必需）
export DEEPSEEK_API_KEY=sk-xxx

# 2. 可选：模型与接口地址（默认 deepseek-flash / https://api.deepseek.com）
export DEEPSEEK_MODEL=deepseek-flash
export DEEPSEEK_BASE_URL=https://api.deepseek.com

# 3. 手动运行真实 API smoke test（pytest 不会自动执行，不消耗 Key）
uv run python scripts/smoke_deepseek.py
```

用法：

```python
from integration_agent.repair import DeepSeekLLMClient, StructuredLLMRepairApplier

client = DeepSeekLLMClient(json_mode=True)  # Key 从环境变量读取
applier = StructuredLLMRepairApplier(client)  # 可注入 RepairLoopRunner
```

注意：`deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 停用，当前推荐使用 `deepseek-flash`（DeepSeek-V4.1-Flash）等模型 ID。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
uv sync

# 2. 运行测试
uv run pytest

# 3. 运行代码检查
uv run ruff check .
uv run ruff format --check .
```

## API Understanding 用法

```python
from integration_agent.api import parse_openapi, parse_openapi_text

# 从文件加载（自动识别 JSON / YAML）
api = parse_openapi("examples/openapi/petstore.yaml")
print(api.name, api.version, api.base_url)  # API 名称、版本、base URL
for endpoint in api.endpoints:
    print(endpoint.method, endpoint.path, endpoint.operation_id)
    print(endpoint.parameters, endpoint.request_body, endpoint.responses)

# 从文本解析（适合从 URL 拉取 spec 的场景）
api = parse_openapi_text(raw_spec_text)
```

解析结果只保留 Agent 推理所需的结构化信息（端点、参数、请求体、响应 schema、认证），
不会把整份 OpenAPI 文档直接塞进 LLM 上下文。无效文档会抛出 `OpenAPISpecError`。

## Repository Understanding 用法

```python
from integration_agent.repository import scan_repository, search_code

# 扫描项目结构：项目名、清单文件、源码/测试目录、依赖、Python 文件列表
project = scan_repository("examples/demo_project")
print(project.name, project.is_python_project)
print(project.manifest_files, project.source_dirs, project.test_dirs)
print(project.dependencies, project.python_files)

# 按关键词搜索代码，返回带行号的受限片段（非向量检索）
result = search_code(project.root, "get_user_profile", context_lines=2, max_results=5)
for match in result.matches:
    print(match.path, match.line_number)
    print(match.snippet)
print(result.truncated, result.files_scanned)
```

扫描时固定忽略 `.git`、`.venv`、`__pycache__` 等目录；搜索结果受单片段字符数、
结果条数与总字符数三重限制，`truncated=True` 表示应缩小关键词或放宽限制。
所有路径均为相对项目根目录的 POSIX 风格字符串。

## 当前范围

- 已实现：API Understanding（OpenAPI 3.x 解析）、Repository Understanding（项目扫描 + 关键词搜索）。
- 其余阶段为骨架，暂未实现业务逻辑。
- 暂不包含：MCP、向量数据库、Knowledge Graph、Kubernetes 等功能，后续按需引入。
