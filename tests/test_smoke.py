"""APIForge 骨架冒烟测试：验证主包与各阶段子模块可正常导入。"""

import importlib

import pytest

import integration_agent

STAGE_SUBMODULES = ["agent", "api", "repository", "tools", "validation", "repair"]


def test_package_imports() -> None:
    assert integration_agent.__version__ == "0.1.0"


@pytest.mark.parametrize("submodule", STAGE_SUBMODULES)
def test_stage_submodule_importable(submodule: str) -> None:
    module = importlib.import_module(f"integration_agent.{submodule}")
    assert module.__doc__
