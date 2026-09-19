"""Demo-only 能力：为浏览器 Demo 制造确定性测试失败（sabotage）。

安全约束：
    - 仅作为 demo_mode 的组成部分，由 app.py 限制在固定 demo
      （openapi/petstore.yaml + demo_project）下使用。
    - sabotage 目标是硬编码的固定字符串替换，不接收任何用户输入。
    - 只修改内存中的 GeneratedArtifacts，绝不触碰真实仓库。
"""

from pathlib import Path

from integration_agent.api_server.paths import ALLOWED_ROOT
from integration_agent.generation import DeterministicCodeGenerator, GeneratedArtifacts

# 允许 demo_mode 的固定组合（相对 examples/ 的路径）
DEMO_SPEC = "openapi/petstore.yaml"
DEMO_PROJECT = "demo_project"

# 固定组合的绝对路径（经 resolve，用于与 resolve_allowed 的结果做等值比较）
DEMO_SPEC_PATH = (ALLOWED_ROOT / DEMO_SPEC).resolve()
DEMO_PROJECT_PATH = (ALLOWED_ROOT / DEMO_PROJECT).resolve()

# demo_mode 只接受这一个固定组合；不接收任何用户指定的目标文件或替换内容
DEMO_ONLY_MESSAGE = (
    f"demo_mode 仅允许固定 Demo 组合：api_spec={DEMO_SPEC!r} 且 project_path={DEMO_PROJECT!r}"
)

# 注入的确定性错误：把生成的单元测试断言改错，使初始测试必然失败
SABOTAGE_FILE = "tests/test_demo_petstore_client.py"
SABOTAGE_FROM = "assert result[0].id == 1"
SABOTAGE_TO = "assert result[0].id == 2"


def is_demo_target(spec_path: Path, project_path: Path) -> bool:
    """判断（已通过路径校验的）目标是否正是唯一允许的固定 Demo 组合。"""
    return spec_path == DEMO_SPEC_PATH and project_path == DEMO_PROJECT_PATH


class SabotagedGenerator:
    """演示用生成器包装：在生成产物中注入一个确定性错误。

    仅在 demo_mode 下使用；自动化测试与普通请求均不经过它。
    """

    def __init__(self) -> None:
        self._inner = DeterministicCodeGenerator()

    def generate(self, plan) -> GeneratedArtifacts:
        artifacts = self._inner.generate(plan)
        injected = False
        for item in artifacts.files:
            if item.path == SABOTAGE_FILE:
                if SABOTAGE_FROM not in item.content:
                    raise RuntimeError(f"注入目标不存在于生成文件中：{SABOTAGE_FROM!r}")
                item.content = item.content.replace(SABOTAGE_FROM, SABOTAGE_TO)
                injected = True
        if not injected:
            # 宁可显式失败，也不静默退化成"恰好通过"的正常 Demo
            raise RuntimeError(f"注入目标文件不在生成产物中：{SABOTAGE_FILE!r}")
        return artifacts
