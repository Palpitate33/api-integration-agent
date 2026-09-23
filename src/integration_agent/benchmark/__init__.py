"""APIForge Benchmark：小型、离线、确定性的能力测量。

    用真实 Pipeline 运行结果量化 Agent 的集成能力——不模拟、不伪造、不联网。

结构：
    models.py   BenchmarkCase / BenchmarkResult / BenchmarkSummary（纯数据模型）
    cases.py    默认套件：4 个固定集成任务（复用 examples/ 已有 fixture）
    runner.py   BenchmarkRunner：真实调用 run_pipeline，从 Trace 归纳结果

用法：
    python -m integration_agent benchmark      # 跑默认套件并打印指标

    from integration_agent.benchmark import BenchmarkRunner
    summary = BenchmarkRunner().run_default()
    summary.pass_rate                          # 0.75 —— 从真实结果动态计算
"""

from integration_agent.benchmark.cases import DEFAULT_CASES
from integration_agent.benchmark.models import BenchmarkCase, BenchmarkResult, BenchmarkSummary
from integration_agent.benchmark.runner import BenchmarkRunner

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSummary",
    "DEFAULT_CASES",
]
