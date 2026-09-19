"""End-to-End Pipeline 手动演示脚本（pytest 不执行）。

用法：
    uv run python scripts/e2e_demo.py            # 默认：离线确定性组件（生成测试全绿）
    uv run python scripts/e2e_demo.py --llm      # 真实 DeepSeek E2E：
                                                 # 注入确定性失败 → RepairPlanner
                                                 # → DeepSeek 修复 → 重新测试 → PASSED
    uv run python scripts/e2e_demo.py --max-iterations 5

--llm 模式说明：
    - 需要一个确定性"可控失败"：SabotagedGenerator 在生成完成后把单元测试里的
      一个正确断言改成错误断言（assert result[0].id == 1 → == 2），
      保证初始测试确定性失败，且失败信息足够让 DeepSeek 理解并修复。
    - 只调用真实 DeepSeek API 一次（每次修复迭代最多一次）。
    - 所有修改只发生在内存中的 GeneratedArtifacts，绝不触碰真实仓库。
    - 不输出 API Key。
"""

import argparse
import sys
from pathlib import Path

from integration_agent.api_server.demo import SabotagedGenerator
from integration_agent.pipeline import run_pipeline
from integration_agent.repair import DeepSeekLLMClient, StructuredLLMRepairApplier

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# SabotagedGenerator 只有一份实现（api_server/demo.py，同时被 Backend 的 demo_mode 使用），
# 此脚本直接复用，避免两处注入逻辑漂移。


def main() -> int:
    parser = argparse.ArgumentParser(description="APIForge End-to-End Pipeline 演示")
    parser.add_argument(
        "--llm", action="store_true", help="真实 DeepSeek E2E：注入确定性失败并启用 LLM Repair"
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    spec = EXAMPLES / "openapi" / "petstore.yaml"
    repo = EXAMPLES / "demo_project"
    kwargs = {}
    mode = "offline deterministic"
    if args.llm:
        mode = "DeepSeek LLM repair (sabotaged)"
        kwargs["code_generator"] = SabotagedGenerator()
        kwargs["repair_applier"] = StructuredLLMRepairApplier(DeepSeekLLMClient(json_mode=True))
    print(f"[mode] {mode}")
    result = run_pipeline(
        spec,
        repo,
        request="集成 Petstore 的查询与创建接口",
        max_iterations=args.max_iterations,
        **kwargs,
    )

    print(f"status            : {result.status}")
    if result.api is not None:
        print(f"api               : {result.api.name} v{result.api.version}")
    if result.project is not None:
        print(f"project           : {result.project.name}")
    print(f"failed_stage      : {result.failed_stage}")
    loop = result.repair_loop_result
    if loop is not None:
        print(f"repair status     : {loop.status}")
        print(f"repair iterations : {loop.iterations}")
        if loop.test_result is not None:
            test = loop.test_result
            print(
                f"final test        : {test.status} "
                f"({test.passed} passed, {test.failed} failed, {test.errors} errors)"
            )
        for index, plan in enumerate(loop.repair_plans):
            print(f"repair plan #{index + 1}   : category={plan.failure_category}")
            for action in plan.actions[:3]:
                print(
                    f"  action          : [{action.action}] {action.file or '<未知>'} "
                    f"(confidence={action.confidence})"
                )
        for index, application in enumerate(loop.application_results):
            print(
                f"application #{index + 1}   : changed={application.changed}, "
                f"applied={len(application.applied_actions)}, "
                f"skipped={len(application.skipped_actions)}"
            )
        if result.artifacts is not None:
            print(f"files ({len(result.artifacts.files)}):")
            for item in result.artifacts.files:
                print(f"  - [{item.action}] {item.path}")
    if result.patch is not None:
        summary = result.patch.summary
        print(
            f"patch summary     : {summary.total_files} files "
            f"({summary.created} created, {summary.modified} modified, "
            f"{summary.dependencies} deps)"
        )
        print("patch preview     :")
        for line in result.patch.unified_diff.splitlines()[:12]:
            print(f"  {line}")
        if result.patch.warnings:
            for warning in result.patch.warnings[:3]:
                print(f"patch warning     : {warning}")
    for warning in result.warnings[:10]:
        print(f"warning           : {warning}")
    if result.error:
        print(f"error             : {result.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
