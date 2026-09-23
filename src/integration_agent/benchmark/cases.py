"""默认 Benchmark 套件：4 个固定、离线、确定性的集成任务。

数据全部复用 examples/ 已有的 fixture（openapi/petstore.yaml + demo_project），
不新建测试数据；失败注入复用 api_server.demo.SabotagedGenerator，不复制注入逻辑；
"另一规模"复用 DeterministicPlanner 已有的 max_endpoints 参数，不新写生成逻辑。

四个 case 各回答一个问题：

    1. petstore_success       正常闭环：API → 仓库 → 规划 → 生成 → 测试 → patch，全绿
    2. petstore_sabotage      失败闭环：测试真失败 → Repair 真尝试 → 如实 no_progress
    3. multi_file_generation  多文件生成：端点数更小的一轮，多份源码 + 测试仍然全绿
    4. deterministic_repeat   确定性：同一 case 连跑两次，核心结构化结果逐项一致
"""

from integration_agent.benchmark.models import BenchmarkCase

DEFAULT_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="petstore_success",
        description="正常成功：Petstore 全流程集成，测试全绿",
        spec_path="openapi/petstore.yaml",
        project_path="demo_project",
        request="集成 Petstore 的查询与创建接口",
        expected_outcome="passed",
    ),
    BenchmarkCase(
        name="petstore_sabotage",
        description="正常失败：注入确定性错误 → 测试失败 → Repair 尝试 → no_progress",
        spec_path="openapi/petstore.yaml",
        project_path="demo_project",
        request="集成 Petstore 的查询与创建接口",
        expected_outcome="failed",
        sabotage=True,
    ),
    BenchmarkCase(
        name="multi_file_generation",
        description="多文件生成：端点数更小的一轮，验证多份源码与测试产物仍能全绿",
        spec_path="openapi/petstore.yaml",
        project_path="demo_project",
        request="集成 Petstore 的查询与创建接口",
        expected_outcome="passed",
        max_endpoints=2,
    ),
    BenchmarkCase(
        name="deterministic_repeat",
        description="确定性：同一 case 连跑两次，核心结构化结果必须逐项一致",
        spec_path="openapi/petstore.yaml",
        project_path="demo_project",
        request="集成 Petstore 的查询与创建接口",
        expected_outcome="passed",
        repeat=2,
    ),
]

__all__ = ["DEFAULT_CASES"]
