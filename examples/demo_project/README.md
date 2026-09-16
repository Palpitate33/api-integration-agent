# Demo Project

APIForge 的集成测试目标项目：一个刻意保持简单的小型 Python 项目，用于验证
Repository Scanner 与后续集成流程（生成代码、运行测试、修复）的行为。

## 结构

- `demo_project/service.py` — 占位服务模块，后续由 Agent 集成真实第三方 API 调用
- `pyproject.toml` — 仅作为被扫描目标项目的元数据示例（不安装、不构建）
