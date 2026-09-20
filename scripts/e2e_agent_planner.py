"""Agent Planner（ToolUsingPlanner + Agent Loop）真实 API 端到端演示脚本。

pytest **不执行**本脚本：真实 DeepSeek 只允许出现在 scripts/ 下的显式脚本里。
单元测试与集成测试一律使用 FakeToolCallingClient（见
tests/test_agent_planner_integration.py）。

用法：
    uv run python scripts/e2e_agent_planner.py              # 只跑 Agent 规划阶段
    uv run python scripts/e2e_agent_planner.py --pipeline   # 再跑完整 Pipeline
    uv run python scripts/e2e_agent_planner.py --json       # 额外打印计划全文 JSON

安全说明：
    - API Key 只从环境变量 DEEPSEEK_API_KEY 读取，由 DeepSeekLLMClient 内部完成；
      本脚本不读取、不打印、不写入任何文件。未配置时打印 SKIPPED 并以 0 退出，
      **不伪造成功**。
    - 只读：不修改 examples/ 下的真实 Demo 项目，不执行 git / shell / 安装依赖。
    - Agent 只有四个只读工具，工具调用全部发生在内存里。
"""

import argparse
import json
import sys
from pathlib import Path

from integration_agent.agent import DeterministicPlanner, PlannerState, ToolUsingPlanner
from integration_agent.agent.tool_prompt import build_tool_planner_prompts
from integration_agent.api import parse_openapi
from integration_agent.pipeline import run_pipeline
from integration_agent.repair import DeepSeekConfigError, DeepSeekLLMClient
from integration_agent.repository import scan_repository
from integration_agent.tools import build_default_registry

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
DEFAULT_SPEC = EXAMPLES / "openapi" / "petstore.yaml"
DEFAULT_PROJECT = EXAMPLES / "demo_project"

REQUEST = "集成 Petstore 的查询与创建接口，复用项目已有的 httpx 客户端与测试约定"

SKIPPED = "SKIPPED: DEEPSEEK_API_KEY is not configured"


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
    parser = argparse.ArgumentParser(description="APIForge Agent Planner E2E")
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

    _section("2. 客户端与工具面")
    # 真实 API 实测：response_format=json_object 与 tools 同时出现会被拒绝
    # （HTTP 400 "Prompt must contain the word 'json' to use 'response_format' of
    # type 'json_object'"），因此 tool calling 路径必须用 json_mode=False。
    try:
        client = DeepSeekLLMClient(json_mode=False)
    except DeepSeekConfigError:
        # 客户端的错误信息只包含环境变量名，不含 Key
        print(SKIPPED)
        return 0

    registry = build_default_registry()
    print(f"model         : {client.model}")
    print(f"base_url      : {client.base_url}")
    print(f"json_mode     : {client.json_mode}")
    print(f"tools         : {[spec.name for spec in registry.specs()]}")

    _section("3. Agent Planner 的 prompt（不把整个仓库交给模型）")
    system_prompt, user_prompt = build_tool_planner_prompts(
        state, tools_description=registry.describe()
    )
    print(f"system 字符数 : {len(system_prompt)}")
    print(f"user   字符数 : {len(user_prompt)}")
    print("证据来源      : Pipeline 已解析的 APIInfo + 已扫描的 ProjectStructure")
    print("补充观察      : Agent 自行决定调哪个只读工具（有轮次与字符预算上限）")

    _section("4. 运行 Agent Loop + 解析 IntegrationPlan（真实 API）")
    # 用真实的 AgentLoopRunner（ToolUsingPlanner 的默认实现），不是 _ReplayPlanner：
    # 本脚本要证明的正是"多轮 tool calling 真的能跑通并产出计划"。
    planner = ToolUsingPlanner(client, registry=registry)
    try:
        plan = planner.plan(state)
    except Exception as exc:  # noqa: BLE001 - 演示脚本需打印失败原因
        print(f"[Planning 失败] {type(exc).__name__}: {exc}")
        return 3

    _section("5. IntegrationPlan（Agent 产出）")
    print(f"integration_goal : {plan.integration_goal}")
    print(f"target_api       : {plan.target_api.name} v{plan.target_api.version}")
    print(f"base_url         : {plan.target_api.base_url}")
    print(f"approach         : {plan.integration_strategy.approach}")
    print(f"http_client      : {plan.integration_strategy.http_client}")
    print(f"client_module    : {plan.integration_strategy.client_module}")

    print(f"\nendpoints ({len(plan.endpoints)}):")
    for item in plan.endpoints:
        print(f"  - {item.endpoint.method:6} {item.endpoint.path:24} {item.purpose}")

    print(f"\nfiles_to_create ({len(plan.files_to_create)}):")
    for item in plan.files_to_create:
        print(f"  - [{item.kind}] {item.path}")

    print(f"\nfiles_to_modify ({len(plan.files_to_modify)}):")
    for item in plan.files_to_modify:
        print(f"  - {item.path}：{item.reason}")

    if plan.authentication is not None:
        auth = plan.authentication
        print(f"\nauthentication   : {auth.scheme} / header={auth.header_name}")
        print(f"  env_vars       : {auth.required_env_vars}")

    _section("6. Warnings（证据不足 / 约束层发现的问题）")
    if plan.warnings:
        for warning in plan.warnings:
            print(f"  ! {warning}")
    else:
        print("  (无)")

    _section("7. 与 DeterministicPlanner 对比（同一输入）")
    baseline = DeterministicPlanner().plan(
        PlannerState(api=api, project=project, request=args.request)
    )
    print(f"{'':16}{'Agent':<40}{'Deterministic'}")
    print(
        f"{'approach':16}{plan.integration_strategy.approach:<40}"
        f"{baseline.integration_strategy.approach}"
    )
    print(f"{'endpoints':16}{len(plan.endpoints):<40}{len(baseline.endpoints)}")
    print(f"{'create':16}{len(plan.files_to_create):<40}{len(baseline.files_to_create)}")
    print(f"{'modify':16}{len(plan.files_to_modify):<40}{len(baseline.files_to_modify)}")
    print(f"{'warnings':16}{len(plan.warnings):<40}{len(baseline.warnings)}")

    if args.json:
        _section("8. 计划全文 JSON")
        print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))

    if not args.pipeline:
        print()
        print("（未运行完整 Pipeline；加 --pipeline 继续跑 Code Generation → Test → Patch）")
        return 0

    _section("9. 完整 Pipeline（用上面这份 Agent 计划驱动）")
    # 复用已经拿到的计划，而不是让 Pipeline 再规划一次：否则会多发一整轮真实
    # API 请求，而且上面展示的计划与真正驱动 Pipeline 的计划会变成两份不同的输出。
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
