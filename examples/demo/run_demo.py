"""APIForge 面试 Demo：一次真实、确定性的端到端集成运行。

用法（在仓库根目录）：
    uv run python examples/demo/run_demo.py

本脚本是**展示层**：只调用已有的 Pipeline 能力，不复刻任何阶段逻辑。

    OpenAPI Spec + Existing Repository
            ↓
        run_pipeline(...)          # 真实 Pipeline：解析 → 扫描 → 规划 → 生成 → 测试
            ↓
        5 阶段进度（控制台）       # 只读 PipelineResult 的事实字段，不做二次推断
            ↓
        output/*.json              # 四份结构化结果，供面试时翻阅

不修改真实 repository：生成文件的落盘与 pytest 的执行都发生在 Pipeline 的
临时工作区内，examples/demo/demo_project 在运行前后逐字节不变。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from integration_agent.pipeline import PipelineResult, run_pipeline
from integration_agent.trace import TraceCollector

DEMO_DIR = Path(__file__).resolve().parent
SPEC_PATH = DEMO_DIR / "openapi.yaml"
REPO_PATH = DEMO_DIR / "demo_project"
OUTPUT_DIR = DEMO_DIR / "output"

REQUEST = "集成用户查询、创建与资料接口"

# 展示的 5 个阶段：key 与 trace 的 TraceStage 一一对应。
STAGES: tuple[tuple[str, str], ...] = (
    ("api_understanding", "API Understanding"),
    ("repository_understanding", "Repository Understanding"),
    ("planner", "Integration Planning"),
    ("generation", "Code Generation"),
    ("test_runner", "Test Running"),
)

# 与内置 CLI demo 一致的退出码约定：passed=0 是唯一"成功"。
EXIT_CODES = {"passed": 0, "tests_failed": 1, "error": 2}

# PipelineResult.failed_stage 的取值（parse/scan/plan/generate/repair）→ 展示阶段 key。
FAILED_STAGE_TO_KEY = {
    "parse": "api_understanding",
    "scan": "repository_understanding",
    "plan": "planner",
    "generate": "generation",
    "repair": "test_runner",
}

STATUS_LABELS = {"passed": "SUCCESS", "tests_failed": "TESTS FAILED", "error": "ERROR"}


def _stage_index(key: str) -> int:
    return next(index for index, (stage_key, _) in enumerate(STAGES) if stage_key == key)


def _stage_view(result: PipelineResult, key: str) -> tuple[str, list[str]]:
    """返回 (结论行, 细节行)。字段全部来自 PipelineResult，不做二次推断。"""
    failed_key = FAILED_STAGE_TO_KEY.get(result.failed_stage) if result.status == "error" else None
    if failed_key == key:
        return f"✗ failed: {result.error or result.failed_stage}", []
    if failed_key is not None and _stage_index(failed_key) < _stage_index(key):
        return "⊘ not reached（上游阶段失败）", []

    if key == "api_understanding":
        api = result.api
        if api is None:
            return "✗ 无 API 解析结果", []
        auth_label = "none"
        if api.auth is not None:
            auth_label = f"http-{api.auth.scheme or api.auth.type}"
            if api.auth.bearer_format:
                auth_label += f" ({api.auth.bearer_format})"
        details = [f"{endpoint.method.upper()} {endpoint.path}" for endpoint in api.endpoints]
        return f"{len(api.endpoints)} endpoints · 认证 {auth_label}", details

    if key == "repository_understanding":
        project = result.project
        if project is None:
            return "✗ 无仓库扫描结果", []
        headline = (
            f"{project.name} · {len(project.python_files)} Python files · "
            f"{len(project.dependencies)} dependencies"
        )
        details = list(project.python_files)
        if project.skipped_paths:
            details.append(f"skipped: {', '.join(project.skipped_paths)}")
        return headline, details

    if key == "planner":
        plan = result.plan
        if plan is None:
            return "✗ 无规划结果", []
        headline = (
            f"{len(plan.endpoints)} endpoints planned · "
            f"{len(plan.files_to_create)} files to create · "
            f"{len(plan.files_to_modify)} files to modify"
        )
        details: list[str] = []
        if plan.repository.existing_modules:
            details.append("复用既有业务模块：" + ", ".join(plan.repository.existing_modules))
        details.append(
            f"HTTP 客户端：{plan.integration_strategy.http_client}"
            f"（{plan.integration_strategy.approach}）"
        )
        if plan.authentication is not None:
            details.append(
                f"认证：{plan.authentication.scheme} → "
                f"环境变量 {', '.join(plan.authentication.required_env_vars)}"
            )
        return headline, details

    if key == "generation":
        artifacts = result.artifacts
        if artifacts is None:
            return "✗ 无生成结果", []
        sources = [item for item in artifacts.created_files if not item.path.startswith("tests/")]
        tests = [item for item in artifacts.created_files if item.path.startswith("tests/")]
        modified = artifacts.modified_files
        headline = (
            f"{len(artifacts.files)} artifacts · {len(sources)} source · "
            f"{len(tests)} test · {len(modified)} modify snippets"
        )
        details = [f"+ {item.path}" for item in sources]
        details += [f"+ {item.path}" for item in tests]
        details += [
            f"~ {item.path}（插入点：{item.insertion_point or '未指定'}）" for item in modified
        ]
        return headline, details

    # test_runner：结论直接来自 RepairLoop 的最终 TestResult。
    loop = result.repair_loop_result
    if loop is None or loop.test_result is None:
        return "✗ 无测试结果", []
    test = loop.test_result
    headline = (
        f"{test.passed} passed / {test.failed} failed / {test.errors} errors"
        f" / {test.skipped} skipped · {test.duration:.2f}s"
    )
    details = [
        "repair: 0 次修复（一次通过，无需修复）"
        if loop.iterations == 0
        else f"repair: {loop.iterations} 次修复尝试"
    ]
    if test.exit_code is not None:
        details.append(f"pytest exit code: {test.exit_code}")
    return headline, details


def _render(result: PipelineResult) -> None:
    """把 PipelineResult 渲染成 5 阶段进度（只读事实字段）。"""
    print("=" * 72)
    print("APIForge Demo — User Management API × demo-user-service")
    print("=" * 72)
    print(f"spec   : {SPEC_PATH.relative_to(DEMO_DIR.parent.parent)}")
    print(f"repo   : {REPO_PATH.relative_to(DEMO_DIR.parent.parent)}")
    print(f"request: {REQUEST}")
    print()
    for index, (key, title) in enumerate(STAGES, start=1):
        headline, details = _stage_view(result, key)
        print(f"[{index}] {title}")
        print(f"    {headline}")
        for line in details:
            print(f"      {line}")
        print()
    if result.patch is not None:
        print(
            f"Patch（建议性，不写盘）：{result.patch.files_changed} files"
            f"（{result.patch.files_created} create · {result.patch.files_modified} modify）"
        )
        print()


def _dump(model: Any) -> Any:
    """Pydantic 模型 → JSON 可序列化的 dict；非模型值原样返回。"""
    return model.model_dump(mode="json") if model is not None else None


def _write_outputs(result: PipelineResult, trace: TraceCollector) -> None:
    """把四份结构化结果写入 output/（谁需要持久化，谁自己 dump）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loop = result.repair_loop_result
    files = (
        ("integration_plan.json", _dump(result.plan)),
        ("generated_artifacts.json", _dump(result.artifacts)),
        ("test_result.json", _dump(loop.test_result if loop is not None else None)),
        ("trace.json", [event.model_dump(mode="json") for event in trace.events()]),
    )
    for name, payload in files:
        path = OUTPUT_DIR / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  写 {path.relative_to(DEMO_DIR.parent.parent)}")


def main() -> int:
    trace = TraceCollector()
    result = run_pipeline(SPEC_PATH, REPO_PATH, request=REQUEST, trace=trace)

    _render(result)

    print("结构化结果（examples/demo/output/）：")
    _write_outputs(result, trace)

    print()
    print("=" * 72)
    print(f"最终状态：{STATUS_LABELS.get(result.status, result.status)}")
    if result.error:
        print(f"error: {result.error}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    print("=" * 72)
    return EXIT_CODES.get(result.status, 2)


if __name__ == "__main__":
    sys.exit(main())
