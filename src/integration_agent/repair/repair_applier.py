"""Deterministic Repair Applier：把 RepairPlan 安全地应用到 GeneratedArtifacts。

契约：RepairPlan → RepairApplicationResult（内含修复后的 GeneratedArtifacts）。

本阶段是安全中间层：
    - 不修改真实 repository、不创建/删除文件、不执行 shell / git / pytest、
      不发网络请求、不安装依赖；只操作内存中的 GeneratedArtifacts。
    - 不根据自然语言 changes 编造代码：无法安全推导完整代码的动作一律
      skip + warning，保持原内容不变，留待 LLMRepairApplier 阶段处理。
    - 原始 GeneratedArtifacts 保持不变：输出永远是深拷贝出的新对象。
    - 全部行为确定性：同样输入必定同样输出。

跳过规则（按序判定，全部记录到 skipped_actions + warnings）：
    1. file == ""            → 无法定位目标文件，不猜
    2. action == "create"    → 确定性阶段不创建新文件
    3. file 不在产物中       → 不自动创建
    4. 目标产物是 modify 片段 → 无法叠加修改
    5. changes 是自然语言描述 → 需要代码合成，确定性阶段无法安全应用
"""

from typing import Protocol, runtime_checkable

from integration_agent.generation import GeneratedArtifacts, GeneratedFile
from integration_agent.repair.models import (
    RepairAction,
    RepairApplicationResult,
    RepairPlan,
)

CODE_SYNTHESIS_WARNING = (
    "Repair action requires code synthesis and cannot be safely applied deterministically."
)


@runtime_checkable
class RepairApplier(Protocol):
    """RepairApplier 接口：确定性实现与未来的 LLM 实现共用同一契约。"""

    def apply(
        self,
        artifacts: GeneratedArtifacts,
        plan: RepairPlan,
    ) -> RepairApplicationResult:
        """把修复计划应用到生成产物，返回修复结果。"""
        ...


def apply_repair(
    artifacts: GeneratedArtifacts,
    plan: RepairPlan,
) -> RepairApplicationResult:
    """便捷入口：默认使用 DeterministicRepairApplier。"""
    return DeterministicRepairApplier().apply(artifacts, plan)


class DeterministicRepairApplier:
    """规则驱动的确定性应用器：只应用可安全推导的修改，其余全部跳过并审计。"""

    def apply(
        self,
        artifacts: GeneratedArtifacts,
        plan: RepairPlan,
    ) -> RepairApplicationResult:
        repaired = artifacts.model_copy(deep=True)
        warnings = list(plan.warnings)
        applied: list[RepairAction] = []
        skipped: list[RepairAction] = []

        if not plan.should_repair:
            return RepairApplicationResult(
                artifacts=repaired,
                applied_actions=applied,
                skipped_actions=skipped,
                warnings=warnings,
                changed=False,
            )
        if not plan.actions:
            warnings.append("No repair actions were provided.")
            return RepairApplicationResult(
                artifacts=repaired,
                applied_actions=applied,
                skipped_actions=skipped,
                warnings=warnings,
                changed=False,
            )

        files_by_path = {item.path: item for item in repaired.files}
        for action in plan.actions:
            skip_reason = _skip_reason(action, files_by_path)
            if skip_reason:
                warnings.append(skip_reason)
                skipped.append(action)
                continue
            # 目标文件存在且完整，但 changes 是自然语言描述：
            # 确定性阶段无法安全推导代码，保持原内容不变。
            warnings.append(CODE_SYNTHESIS_WARNING)
            skipped.append(action)

        return RepairApplicationResult(
            artifacts=repaired,
            applied_actions=applied,
            skipped_actions=skipped,
            warnings=warnings,
            changed=bool(applied),
        )


def _skip_reason(action: RepairAction, files_by_path: dict[str, GeneratedFile]) -> str:
    """返回该动作必须被跳过的原因；返回空字符串表示可进入应用流程。"""
    if not action.file:
        return "无法定位目标文件（file 为空），已跳过该修复动作"
    if action.action == "create":
        return "create 动作暂不受支持（确定性阶段不创建新文件），已跳过该修复动作"
    target = files_by_path.get(action.file)
    if target is None:
        return f"目标文件不存在于生成产物中：{action.file}，已跳过该修复动作"
    if target.action == "modify":
        return f"目标产物 {action.file} 是修改片段而非完整文件，无法叠加修改，已跳过"
    return ""
