"""Deterministic Repair Planner：把失败的 TestResult 分析为结构化 RepairPlan。

契约：GeneratedArtifacts + TestResult → RepairPlan。

本阶段是纯分析组件：
    - 不修改仓库、不创建/删除文件、不执行 shell / git / pytest、
      不发网络请求、不读取 secret、不安装依赖。
    - 不接 LLM：分类与动作生成全部由确定性规则完成，同样输入必定同样输出。
    - 只基于输入证据（failure_details / status / exit_code / stdout / stderr /
      dependency_warnings）分析，不编造证据之外的信息；无法可靠分类时归为
      unknown 并停止生成修复动作。

失败分类（按优先级）：
    dependency > import > collection > assertion > unknown
    status 级信号优先：passed → none；timeout → timeout；error → environment
    （若 stdout 出现收集失败信号则归为 collection）。
"""

import re
from typing import Protocol, runtime_checkable

from integration_agent.generation import GeneratedArtifacts
from integration_agent.repair.models import RepairAction, RepairPlan
from integration_agent.validation import FailureDetail, TestResult

DEFAULT_MAX_ITERATIONS = 3
MAX_ACTIONS_PER_PLAN = 5  # 修复动作数量上限，保证计划有限可审

# 类别优先级：数字越大越优先（用于多失败详情时确定整体类别）
_PRIORITY = {"dependency": 40, "import": 30, "collection": 20, "assertion": 10, "unknown": 0}

# 类别 → 默认置信度：只反映"证据能指向该类别"的把握，不代表修复必成功
_CONFIDENCE = {
    "assertion": 0.6,
    "import": 0.5,
    "dependency": 0.7,
    "collection": 0.4,
    "timeout": 0.3,
}

# 收集期失败的信号（出现在消息 / traceback / stdout / stderr 中）
_COLLECTION_SIGNALS = ("fixture", "SyntaxError", "during collection", "Interrupted")


@runtime_checkable
class RepairPlanner(Protocol):
    """RepairPlanner 接口：确定性实现与未来的 LLM 实现共用同一契约。"""

    def plan(
        self,
        artifacts: GeneratedArtifacts,
        test_result: TestResult,
    ) -> RepairPlan:
        """基于测试结果产出修复计划。"""
        ...


def plan_repair(
    artifacts: GeneratedArtifacts,
    test_result: TestResult,
) -> RepairPlan:
    """便捷入口：默认使用 DeterministicRepairPlanner。"""
    return DeterministicRepairPlanner().plan(artifacts, test_result)


class DeterministicRepairPlanner:
    """规则驱动的确定性修复计划器：只分析、只规划，不执行任何副作用。"""

    def __init__(self, *, max_iterations: int = DEFAULT_MAX_ITERATIONS, iteration: int = 0) -> None:
        if max_iterations <= 0:
            raise ValueError(f"max_iterations 必须大于 0，实际为 {max_iterations}")
        if iteration > max_iterations:
            raise ValueError(f"iteration ({iteration}) 不能大于 max_iterations ({max_iterations})")
        self.max_iterations = max_iterations
        self.iteration = iteration

    def plan(
        self,
        artifacts: GeneratedArtifacts,
        test_result: TestResult,
    ) -> RepairPlan:
        category = _classify(test_result, artifacts)
        analysis, actions, warnings = self._build(category, artifacts, test_result)
        if self.iteration >= self.max_iterations:
            should_repair = False
            actions = []
            warnings.append(f"已达到最大迭代次数（{self.max_iterations}），停止生成修复动作")
        else:
            should_repair = category in {
                "assertion",
                "import",
                "dependency",
                "collection",
                "timeout",
            }
        return RepairPlan(
            should_repair=should_repair,
            iteration=self.iteration,
            max_iterations=self.max_iterations,
            failure_category=category,
            analysis=analysis,
            actions=actions,
            warnings=warnings,
        )

    def _build(
        self,
        category: str,
        artifacts: GeneratedArtifacts,
        result: TestResult,
    ) -> tuple[str, list[RepairAction], list[str]]:
        warnings: list[str] = []
        if category == "none":
            return _analysis(category, result), [], []
        if category == "timeout":
            action, extra = _timeout_action(result)
            return _analysis(category, result), [action], warnings + extra
        if category == "environment":
            return _analysis(category, result), [], warnings
        if category == "unknown":
            warnings.append("无法可靠分类失败原因，不生成修复动作")
            return _analysis(category, result), [], warnings
        actions, extra = _actions_for_details(category, artifacts, result)
        return _analysis(category, result), actions, warnings + extra


# ---------------------------------------------------------------- 失败分类


def _classify(result: TestResult, artifacts: GeneratedArtifacts) -> str:
    if result.status == "passed":
        return "none"
    if result.status == "timeout":
        return "timeout"
    if result.status == "error":
        # 收集期崩溃（如测试模块导入失败）在 stdout 中有明确信号
        if _has_collection_signal(result.stdout):
            return "collection"
        return "environment"
    return _classify_failed(result, artifacts)


def _classify_failed(result: TestResult, artifacts: GeneratedArtifacts) -> str:
    details = result.failure_details
    if not details:
        return "unknown"
    categories = [_detail_category(detail, result, artifacts) for detail in details]
    if result.exit_code == 2 and result.errors > 0:
        categories.append("collection")
    return max(categories, key=lambda item: _PRIORITY[item])


def _detail_category(
    detail: FailureDetail,
    result: TestResult,
    artifacts: GeneratedArtifacts,
) -> str:
    text = " ".join(
        part for part in (detail.message, detail.traceback or "", result.stderr) if part
    ).lower()
    module = _missing_module(text)
    dep_names = {change.name for change in artifacts.dependency_changes}
    if result.dependency_warnings:
        return "dependency"
    if module and module in dep_names:
        return "dependency"
    if module:
        return "import"
    if _has_collection_signal(text):
        return "collection"
    if "assert" in text or "assertionerror" in text:
        return "assertion"
    return "unknown"


def _missing_module(text: str) -> str | None:
    """从 "No module named 'xxx.yyy'" 中提取顶层模块名。"""
    match = re.search(r"no module named ['\"]([\w.]+)['\"]", text)
    if not match:
        return None
    return match.group(1).split(".")[0]


def _has_collection_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal.lower() in lowered for signal in _COLLECTION_SIGNALS)


# ---------------------------------------------------------------- 动作生成


def _actions_for_details(
    category: str,
    artifacts: GeneratedArtifacts,
    result: TestResult,
) -> tuple[list[RepairAction], list[str]]:
    """按失败详情生成有限修复动作：按 (file, 类别) 去重合并，数量有上限。"""
    warnings: list[str] = []
    by_key: dict[tuple[str, str], RepairAction] = {}
    for detail in result.failure_details:
        detail_category = _detail_category(detail, result, artifacts)
        key = (detail.file or "", detail_category)
        evidence = _evidence_for(detail)
        if key in by_key:
            existing = by_key[key]
            merged = existing.evidence + [
                item for item in evidence if item not in existing.evidence
            ]
            existing.evidence = merged[:6]
            continue
        by_key[key] = RepairAction(
            file=detail.file or "",
            action="modify",
            reason=_reason_for(detail_category),
            target=_target_for(detail_category, detail),
            changes=_changes_for(
                detail_category,
                _missing_module(
                    " ".join(
                        part for part in (detail.message, detail.traceback or "") if part
                    ).lower()
                ),
            ),
            confidence=_CONFIDENCE[detail_category],
            evidence=evidence,
        )
    actions = list(by_key.values())
    if category == "dependency" and result.dependency_warnings:
        for action in actions:
            if "依赖" in action.reason:
                action.evidence = (
                    action.evidence
                    + [
                        f"dependency warning: {item}"
                        for item in result.dependency_warnings[:3]
                        if f"dependency warning: {item}" not in action.evidence
                    ]
                )[:6]
    if len(actions) > MAX_ACTIONS_PER_PLAN:
        warnings.append(
            f"失败详情超过 {MAX_ACTIONS_PER_PLAN} 类，仅保留前 {MAX_ACTIONS_PER_PLAN} 条修复动作"
        )
        actions = actions[:MAX_ACTIONS_PER_PLAN]
    for action in actions:
        if not action.file:
            warnings.append(
                "Exact source file cannot be determined from the available failure evidence."
            )
    return actions, warnings


def _evidence_for(detail: FailureDetail) -> list[str]:
    evidence = [f"failure message: {detail.message}"]
    if detail.file:
        evidence.append(f"test file: {detail.file}")
    if detail.line is not None:
        evidence.append(f"line: {detail.line}")
    return evidence


def _reason_for(category: str) -> str:
    return {
        "assertion": "测试失败证据显示断言期望与实际行为不一致",
        "import": "测试失败证据显示 import 错误",
        "dependency": "测试依赖的包缺失或 import 与依赖清单不一致",
        "collection": "测试收集阶段失败",
        "timeout": "测试执行超过时间限制",
    }[category]


def _target_for(category: str, detail: FailureDetail) -> str | None:
    if category == "assertion" and detail.line is not None:
        return f"line {detail.line}"
    if category == "import":
        return "import section"
    if category == "dependency":
        return "依赖声明 / import 语句"
    if category == "collection":
        return "fixture / 测试定义 / 导入"
    return None


def _changes_for(category: str, module: str | None) -> str:
    if category == "import":
        name = module or "被导入模块"
        return f"修正 import '{name}' 的路径或名称，确保被导入模块存在于生成产物中"
    if category == "dependency":
        name = module or "该依赖"
        return f"确认依赖 '{name}' 已安装（安装动作留给应用阶段或人工处理），或修正 import"
    if category == "collection":
        return "修复测试收集失败：检查 fixture 定义、测试文件语法与导入"
    if category == "assertion":
        return "检查断言期望与生成代码的实际行为，修正断言逻辑或对应生成代码"
    return "基于现有证据进行有限修正"


def _timeout_action(result: TestResult) -> tuple[RepairAction, list[str]]:
    """超时无法定位具体文件：给出低置信度的有限建议，并在 warnings 中说明。"""
    limit_match = re.search(r"超过 ([\d.]+)s", result.stderr)
    evidence = [f"status: {result.status}", f"duration: {result.duration}s"]
    if limit_match:
        evidence.append(f"timeout limit: {limit_match.group(1)}s")
    action = RepairAction(
        file="",
        action="modify",
        reason="测试执行超过时间限制",
        target="TestRunner 超时配置",
        changes="缩短测试中的长耗时操作，或提高 TestRunner 的 timeout 阈值",
        confidence=_CONFIDENCE["timeout"],
        evidence=evidence,
    )
    warnings = ["Exact source file cannot be determined from the available failure evidence."]
    return action, warnings


# ------------------------------------------------------------------ 分析文本


def _analysis(category: str, result: TestResult) -> str:
    """按类别输出确定性分析文本（全部来自输入证据，不做猜测）。"""
    if category == "none":
        return (
            f"测试全部通过（{result.passed} passed, {result.skipped} skipped），"
            "当前生成产物无需修复。"
        )
    if category == "assertion":
        return (
            f"{result.failed} 个测试断言失败：断言期望与生成代码的实际行为不一致，"
            "需要核对断言逻辑或修正生成代码。"
        )
    if category == "import":
        return "测试因导入失败：import 指向的模块不存在或路径不正确，需修正导入。"
    if category == "dependency":
        return "测试依赖的包缺失或 import 与依赖清单不一致；安装依赖的动作不在本阶段执行。"
    if category == "collection":
        return "pytest 收集阶段失败：测试文件语法、导入或 fixture 定义存在问题。"
    if category == "timeout":
        return (
            f"测试执行超过时间限制被终止（已运行 {result.duration}s），"
            "现有证据无法定位具体卡点，仅提供有限分析。"
        )
    if category == "environment":
        return f"pytest 执行环境异常（exit_code={result.exit_code}），无法通过修改代码修复。"
    return "失败原因无法可靠分类，不生成修复动作。"
