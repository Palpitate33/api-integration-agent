"""DeepSeek Planner 真实 API 端到端演示脚本（pytest 不执行）。

这是**唯一**会真实调用 DeepSeek 的 Planner 入口：单元测试全部使用 FakeLLMClient，
真实 API 只允许出现在 scripts/ 下的显式脚本里。

用法：
    uv run python scripts/e2e_deepseek_planner.py              # 只跑 Planning 阶段
    uv run python scripts/e2e_deepseek_planner.py --pipeline   # 再跑完整 Pipeline
    uv run python scripts/e2e_deepseek_planner.py --json       # 额外打印计划全文 JSON

安全说明：
    - API Key 只从环境变量 DEEPSEEK_API_KEY 读取，由 DeepSeekLLMClient 内部完成，
      本脚本不读取、不打印、不写入任何文件。
    - 只读：不修改 examples/ 下的真实 Demo 项目，不执行 git / shell / 安装依赖。
    - 输出中出现的任何 credential 都由 Planner 层脱敏后才会打印。
"""

import argparse
import json
import sys
from pathlib import Path

from integration_agent.agent import DeepSeekPlanner, DeterministicPlanner, PlannerState
from integration_agent.agent.prompt import build_planner_prompt
from integration_agent.api import parse_openapi
from integration_agent.pipeline import run_pipeline
from integration_agent.repair import DeepSeekConfigError, DeepSeekLLMClient
from integration_agent.repository import scan_repository

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
DEFAULT_SPEC = EXAMPLES / "openapi" / "petstore.yaml"
DEFAULT_PROJECT = EXAMPLES / "demo_project"

REQUEST = "集成 Petstore 的查询与创建接口，复用项目已有的 httpx 客户端与测试约定"


class _ReplayPlanner:
    """把已经算好的计划原样交回 Pipeline，避免再次调用真实 API。"""

    def __init__(self, plan) -> None:
        self._plan = plan

    def plan(self, state):
        return self._plan


def _section(title: str) -> None:
    print()
    print(f"===== {title} " + "=" * max(0, 60 - len(title)))


def main() -> int:
    parser = argparse.ArgumentParser(description="APIForge DeepSeek Planner E2E")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--request", default=REQUEST)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--pipeline", action="store_true", help="规划之后继续跑完整 Pipeline")
    parser.add_argument("--json", action="store_true", help="额外打印计划全文 JSON")
    args = parser.parse_args()

    spec_path, project_path = Path(args.spec), Path(args.project)

    _section("1. 输入")
    print(f"spec        : {spec_path}")
    print(f"project     : {project_path}")
    print(f"request     : {args.request}")

    api = parse_openapi(spec_path)
    project = scan_repository(project_path)
    state = PlannerState(api=api, project=project, request=args.request)
    print(f"api         : {api.name} v{api.version}（{len(api.endpoints)} 个端点）")
    print(f"repo        : {project.name}（{len(project.python_files)} 个 Python 文件）")

    _section("2. Prompt 预算（不把整个仓库交给模型）")
    prompt = build_planner_prompt(state)
    print(f"prompt 字符数 : {len(prompt)}")
    print("证据来源      : 调用方未提供 → Planner 自行发起受限定向检索")
    print("裁剪优先级    : HTTP 客户端 > 认证 > 业务 > 模型 > 测试 > 配置")

    _section("3. 调用 DeepSeek Planner")
    try:
        client = DeepSeekLLMClient(json_mode=True)
    except DeepSeekConfigError as exc:
        # 客户端的错误信息只包含环境变量名，不含 Key
        print(f"[配置错误] {exc}")
        return 2

    print(f"model         : {client.model}")
    print(f"base_url      : {client.base_url}")
    print(f"json_mode     : {client.json_mode}")
    planner = DeepSeekPlanner(client)
    try:
        plan = planner.plan(state)
    except Exception as exc:  # noqa: BLE001 - 演示脚本需打印失败原因
        print(f"[Planning 失败] {type(exc).__name__}: {exc}")
        return 3

    _section("4. IntegrationPlan（DeepSeek 生成）")
    print(f"integration_goal : {plan.integration_goal}")
    print(f"target_api       : {plan.target_api.name} v{plan.target_api.version}")
    print(f"base_url         : {plan.target_api.base_url}")
    print(f"approach         : {plan.integration_strategy.approach}")
    print(f"http_client      : {plan.integration_strategy.http_client}")
    print(f"client_module    : {plan.integration_strategy.client_module}")
    print(f"summary          : {plan.integration_strategy.summary}")

    print(f"\nendpoints ({len(plan.endpoints)}):")
    for item in plan.endpoints:
        print(f"  - {item.method:6} {item.path:24} {item.purpose}")

    print(f"\nfiles_to_create ({len(plan.files_to_create)}):")
    for item in plan.files_to_create:
        print(f"  - [{item.kind}] {item.path}")
        print(f"      {item.purpose}")

    print(f"\nfiles_to_modify ({len(plan.files_to_modify)}):")
    for item in plan.files_to_modify:
        print(f"  - {item.path}：{item.reason}")
        for change in item.changes:
            print(f"      · {change}")

    print(f"\ndependencies ({len(plan.dependencies)}):")
    for item in plan.dependencies:
        flag = "已安装" if item.already_installed else "需新增"
        print(f"  - {item.name} {item.version or ''} [{flag}] {item.reason}")

    if plan.authentication is not None:
        auth = plan.authentication
        print(f"\nauthentication   : {auth.scheme} / header={auth.header_name}")
        print(f"  strategy       : {auth.strategy}")
        print(f"  config_location: {auth.config_location}")
        print(f"  env_vars       : {auth.required_env_vars}")
    else:
        print("\nauthentication   : (无)")

    print(f"\nerror_handling   : {plan.error_handling.exception_type}")
    for rule in plan.error_handling.rules:
        retry = "可重试" if rule.retryable else "不重试"
        print(f"  - {rule.status_code:5} {rule.category:15} [{retry}] {rule.strategy}")

    testing = plan.testing_strategy
    print(
        f"\ntesting_strategy : unit={len(testing.unit_tests)} "
        f"integration={len(testing.integration_tests)} contract={len(testing.contract_tests)}"
    )
    for spec in (*testing.unit_tests, *testing.integration_tests, *testing.contract_tests):
        print(f"  - {spec.path}：{spec.purpose}")

    if plan.assumptions:
        print(f"\nassumptions ({len(plan.assumptions)}):")
        for item in plan.assumptions:
            print(f"  - {item}")

    if plan.risks:
        print(f"\nrisks ({len(plan.risks)}):")
        for item in plan.risks:
            print(f"  - [{item.severity}] {item.description} → {item.mitigation}")

    _section("5. Warnings（证据不足 / 约束层发现的问题）")
    if plan.warnings:
        for warning in plan.warnings:
            print(f"  ! {warning}")
    else:
        print("  (无)")

    _section("6. 与 DeterministicPlanner 对比（同一输入）")
    baseline_state = PlannerState(api=api, project=project, request=args.request)
    baseline = DeterministicPlanner().plan(baseline_state)
    print(f"{'':16}{'DeepSeek':<40}{'Deterministic'}")
    print(
        f"{'approach':16}{plan.integration_strategy.approach:<40}{baseline.integration_strategy.approach}"
    )
    print(
        f"{'http_client':16}{str(plan.integration_strategy.http_client):<40}{baseline.integration_strategy.http_client}"
    )
    print(f"{'endpoints':16}{len(plan.endpoints):<40}{len(baseline.endpoints)}")
    print(f"{'create':16}{len(plan.files_to_create):<40}{len(baseline.files_to_create)}")
    print(f"{'modify':16}{len(plan.files_to_modify):<40}{len(baseline.files_to_modify)}")
    print(f"{'dependencies':16}{len(plan.dependencies):<40}{len(baseline.dependencies)}")
    print(f"{'warnings':16}{len(plan.warnings):<40}{len(baseline.warnings)}")

    if args.json:
        _section("7. 计划全文 JSON")
        print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))

    if not args.pipeline:
        print()
        print("（未运行完整 Pipeline；加 --pipeline 继续跑 Code Generation → Test → Patch）")
        return 0

    _section("8. 完整 Pipeline（用上面这份 DeepSeek 计划驱动）")
    # 复用已经拿到的计划，而不是让 Pipeline 再规划一次：否则会多发一次真实 API 请求，
    # 而且上面展示的计划与真正驱动 Pipeline 的计划是两份不同的输出（LLM 有随机性）。
    result = run_pipeline(
        spec_path,
        project_path,
        request=args.request,
        max_iterations=args.max_iterations,
        planner=_ReplayPlanner(plan),
    )
    print(f"status            : {result.status}")
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
            # 失败/收集错误时把 pytest 原文打出来——否则 "1 errors" 无法定位原因
            if test.status in {"failed", "error"}:
                output = (test.stdout or "") + (test.stderr or "")
                if output.strip():
                    print("\n--- pytest 输出（尾部）---")
                    print(output[-3000:].rstrip())
    if result.artifacts is not None:
        print(f"generated files   : {len(result.artifacts.files)}")
    if result.patch is not None:
        summary = result.patch.summary
        print(
            f"patch summary     : {summary.total_files} files "
            f"({summary.created} created, {summary.modified} modified, "
            f"{summary.dependencies} deps)"
        )
    for warning in result.warnings[:10]:
        print(f"warning           : {warning}")
    if result.error:
        print(f"error             : {result.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
