"""Code Generator 阶段：把 IntegrationPlan 转换为 GeneratedArtifacts。

主要 API：
    generate_code(plan)                          —— 便捷入口
    DeterministicCodeGenerator().generate(plan)  —— 确定性实现
    CodeGenerator                                —— Protocol，未来 LLMCodeGenerator 实现同一契约

输出契约：
    GeneratedArtifacts / GeneratedFile / DependencyChange

约束：纯生成组件——不写磁盘、不修改仓库、不执行命令、不发起网络请求、
不读取 secret；同样输入必定同样输出。
"""

from integration_agent.generation.code_generator import (
    CodeGenerator,
    DeterministicCodeGenerator,
    generate_code,
)
from integration_agent.generation.models import (
    DependencyChange,
    GeneratedArtifacts,
    GeneratedFile,
)

__all__ = [
    "CodeGenerator",
    "DependencyChange",
    "DeterministicCodeGenerator",
    "GeneratedArtifacts",
    "GeneratedFile",
    "generate_code",
]
