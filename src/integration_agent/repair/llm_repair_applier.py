"""LLM Repair Applier：根据 RepairPlan + TestResult + GeneratedArtifacts 生成具体代码修改。

契约：GeneratedArtifacts + RepairPlan + TestResult → RepairApplicationResult。

职责划分（与 RepairPlanner 严格分离）：
    RepairPlanner      = 决定"要不要修、修哪里"（should_repair / actions / 分类）
    LLMRepairApplier   = 决定"具体代码怎么改"（通过 LLMClient 生成完整文件内容）

安全边界：
    - 不修改真实 Repository：只对 GeneratedArtifacts 的深拷贝做内存变换。
    - 不执行 shell / git、不发网络请求（网络调用由注入的 LLMClient 负责，
      本模块不直接联网）、不读取 secrets。
    - LLM 输出必须是严格 JSON：非法 JSON / 缺字段 / 类型错误一律拒绝，
      不做任何自动猜测或修正。
    - path 必须是安全相对路径；修改数量与内容大小都有硬上限。
    - 原始 GeneratedArtifacts 保持不变。
    - 同样输入 + 确定性 LLMClient → 同样输出。

LLM 返回格式（严格 JSON）：
    {"changes": [{"path": "src/x.py", "action": "modify|create",
                  "content": "<完整文件内容>", "reason": "..."}],
     "summary": "...", "warnings": ["..."]}
"""

import json
import re
from typing import Protocol, runtime_checkable

from integration_agent.generation import GeneratedArtifacts, GeneratedFile
from integration_agent.repair.llm_client import LLMClient
from integration_agent.repair.models import (
    LLMFileChange,
    LLMRepairResponse,
    RepairAction,
    RepairApplicationResult,
    RepairPlan,
)
from integration_agent.validation import TestResult

DEFAULT_MAX_CHANGES = 5  # 单次允许的最大修改文件数量
DEFAULT_MAX_FILE_CONTENT_CHARS = 20_000  # 单个文件内容上限
DEFAULT_MAX_TOTAL_CONTENT_CHARS = 50_000  # 单次修改内容总量上限
MAX_PROMPT_FILE_CHARS = 4_000  # prompt 中单个文件上下文上限
MAX_PROMPT_TOTAL_CHARS = 12_000  # prompt 中文件上下文总量上限
MAX_PROMPT_FAILURE_DETAILS = 5

_WINDOWS_ABSOLUTE = re.compile(r"^[a-zA-Z]:")


@runtime_checkable
class LLMRepairApplier(Protocol):
    """LLM 修复应用器接口：决定"具体代码怎么改"。"""

    def apply(
        self,
        artifacts: GeneratedArtifacts,
        plan: RepairPlan,
        test_result: TestResult,
    ) -> RepairApplicationResult:
        """基于修复计划与测试结果生成并应用代码修改。"""
        ...


class StructuredLLMRepairApplier:
    """通过 LLMClient 生成严格 JSON 修改并安全应用到产物副本的应用器。

    test_result 参数可选（默认 None）：RepairLoop 只传 (artifacts, plan)，
    本实现可直接注入现有循环而不破坏 RepairApplier 调用约定。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_changes: int = DEFAULT_MAX_CHANGES,
        max_file_content_chars: int = DEFAULT_MAX_FILE_CONTENT_CHARS,
        max_total_content_chars: int = DEFAULT_MAX_TOTAL_CONTENT_CHARS,
    ) -> None:
        self.llm_client = llm_client
        self.max_changes = max_changes
        self.max_file_content_chars = max_file_content_chars
        self.max_total_content_chars = max_total_content_chars

    def apply(
        self,
        artifacts: GeneratedArtifacts,
        plan: RepairPlan,
        test_result: TestResult | None = None,
    ) -> RepairApplicationResult:
        repaired = artifacts.model_copy(deep=True)
        warnings = list(plan.warnings)

        if not plan.should_repair:
            warnings.append("RepairPlan.should_repair=False，不调用 LLM，产物保持不变")
            return RepairApplicationResult(artifacts=repaired, warnings=warnings, changed=False)
        if not plan.actions:
            warnings.append("No repair actions were provided.")
            return RepairApplicationResult(artifacts=repaired, warnings=warnings, changed=False)

        prompt = build_repair_prompt(artifacts, plan, test_result)
        try:
            raw = self.llm_client.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - LLM 失败不能导致崩溃
            warnings.append(f"LLMClient 调用失败：{type(exc).__name__}: {exc}")
            return RepairApplicationResult(artifacts=repaired, warnings=warnings, changed=False)

        response, parse_warnings = parse_llm_response(raw)
        warnings.extend(parse_warnings)
        if response is None:
            return RepairApplicationResult(artifacts=repaired, warnings=warnings, changed=False)
        warnings.extend(response.warnings)

        applied, skipped, apply_warnings = self._apply_changes(repaired, response.changes, plan)
        warnings.extend(apply_warnings)
        return RepairApplicationResult(
            artifacts=repaired,
            applied_actions=applied,
            skipped_actions=skipped,
            warnings=warnings,
            changed=bool(applied),
        )

    def _apply_changes(
        self,
        repaired: GeneratedArtifacts,
        changes: list[LLMFileChange],
        plan: RepairPlan,
    ) -> tuple[list[RepairAction], list[RepairAction], list[str]]:
        """逐条校验并应用 LLM 修改；任何问题只跳过该条并记录 warning。"""
        applied: list[RepairAction] = []
        skipped: list[RepairAction] = []
        warnings: list[str] = []
        files_by_path = {item.path: item for item in repaired.files}
        plan_confidence = {action.file: action.confidence for action in plan.actions}
        seen_paths: set[str] = set()
        total_chars = 0

        for change in changes:
            path_error = validate_path(change.path)
            if path_error:
                warnings.append(f"跳过修改：{path_error}")
                skipped.append(_record(change, path_error))
                continue
            if change.path in seen_paths:
                warnings.append(f"跳过修改：不允许重复修改同一个文件 {change.path}")
                skipped.append(_record(change, "重复修改同一个文件"))
                continue
            seen_paths.add(change.path)

            existing = files_by_path.get(change.path)
            if change.action == "modify":
                if existing is None:
                    warnings.append(
                        f"跳过修改：目标文件不存在于生成产物中：{change.path}（modify 不自动创建）"
                    )
                    skipped.append(_record(change, "目标文件不存在"))
                    continue
                if existing.action != "create":
                    warnings.append(
                        f"跳过修改：目标产物 {change.path} 是修改片段而非完整文件，无法替换"
                    )
                    skipped.append(_record(change, "目标产物是修改片段"))
                    continue
            else:  # create
                if existing is not None:
                    warnings.append(f"跳过创建：文件已存在于生成产物中：{change.path}")
                    skipped.append(_record(change, "文件已存在"))
                    continue

            if not change.content.strip():
                warnings.append(f"跳过修改：{change.path} 的内容为空")
                skipped.append(_record(change, "内容为空"))
                continue
            if len(change.content) > self.max_file_content_chars:
                warnings.append(
                    f"跳过修改：{change.path} 内容 {len(change.content)} 字符，"
                    f"超过单文件上限 {self.max_file_content_chars}"
                )
                skipped.append(_record(change, "单文件内容超过上限"))
                continue
            total_chars += len(change.content)
            if total_chars > self.max_total_content_chars:
                warnings.append(f"跳过修改：累计内容超过总上限 {self.max_total_content_chars} 字符")
                skipped.append(_record(change, "总内容超过上限"))
                continue
            if len(applied) >= self.max_changes:
                warnings.append(f"跳过修改：修改文件数量超过上限 {self.max_changes}")
                skipped.append(_record(change, "修改数量超过上限"))
                continue

            if change.action == "modify":
                existing.content = change.content
                if change.reason:
                    existing.changes = list(existing.changes) + [change.reason]
            else:
                purpose = f"LLM 修复生成：{change.reason}" if change.reason else "LLM 修复生成"
                repaired.files.append(
                    GeneratedFile(
                        path=change.path,
                        action="create",
                        content=change.content,
                        purpose=purpose,
                    )
                )
            applied.append(
                RepairAction(
                    file=change.path,
                    action=change.action,
                    reason=change.reason or "LLM 生成修改",
                    target=None,
                    changes="LLM 输出完整文件内容",
                    confidence=plan_confidence.get(change.path, 0.5),
                )
            )
        return applied, skipped, warnings


def _record(change: LLMFileChange, reason: str) -> RepairAction:
    """把被跳过的 LLM 修改转成可审计的 RepairAction 记录。"""
    return RepairAction(
        file=change.path,
        action=change.action,
        reason=reason,
        target=None,
        changes=change.reason or "",
        confidence=0.0,
    )


# ------------------------------------------------------------------ prompt


def build_repair_prompt(
    artifacts: GeneratedArtifacts,
    plan: RepairPlan,
    test_result: TestResult | None,
) -> str:
    """构建受控上下文的修复 prompt：计划 + 失败信息 + 相关文件内容，绝不全量塞仓库。"""
    lines = [
        "你是 APIForge 的代码修复专家。根据修复计划修改生成产物中的文件。",
        "",
        "# 修复计划",
        f"failure_category: {plan.failure_category}",
        f"analysis: {plan.analysis}",
        "actions:",
    ]
    for action in plan.actions:
        lines.append(
            f"  - file={action.file or '<未知>'} target={action.target or '-'} "
            f"changes={action.changes} evidence={action.evidence}"
        )
    lines += ["", "# 测试结果"]
    if test_result is None:
        lines.append("（未提供）")
    else:
        lines.append(
            f"status={test_result.status} exit_code={test_result.exit_code} "
            f"passed={test_result.passed} failed={test_result.failed} "
            f"errors={test_result.errors} skipped={test_result.skipped}"
        )
        if test_result.failure_details:
            lines.append("失败详情：")
            for detail in test_result.failure_details[:MAX_PROMPT_FAILURE_DETAILS]:
                lines.append(
                    f"  - {detail.test_name}: {detail.message}"
                    f" ({detail.file or '-'}:{detail.line or '-'})"
                )

    lines += ["", "# 目标文件内容"]
    budget = MAX_PROMPT_TOTAL_CHARS
    for action in plan.actions:
        if not action.file or budget <= 0:
            continue
        target = next((f for f in artifacts.files if f.path == action.file), None)
        if target is None:
            continue
        snippet = target.content[:MAX_PROMPT_FILE_CHARS]
        budget -= len(snippet)
        lines.append(f"### {action.file}")
        lines.append(snippet)
        lines.append("")

    lines += [
        "# 要求",
        "1. 只输出一个 JSON 对象，不要输出任何 Markdown 代码块或解释文字。",
        "2. JSON 必须严格符合结构：",
        '   {"changes": [{"path": "...", "action": "modify|create", '
        '"content": "...", "reason": "..."}], "summary": "...", "warnings": [...]}',
        "3. modify 的 content 是修改后的完整文件内容；create 的 content 是新文件完整内容。",
        "4. path 必须是相对路径：禁止 ../、禁止绝对路径（如 /etc/x、C:\\\\x）。",
        "5. 只能修改上面列出的目标文件；create 可以新增文件，但不要创建无关文件。",
        "6. 无法安全修改时返回空 changes，并在 warnings 中说明原因。",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------ 解析与校验


def parse_llm_response(raw: str) -> tuple[LLMRepairResponse | None, list[str]]:
    """严格解析 LLM 输出：非法 JSON / 缺字段 / 类型错误一律拒绝，不做任何修正。"""
    if not raw or not raw.strip():
        return None, ["LLM 返回空响应，无法解析"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"LLM 返回非法 JSON：{exc}"]
    try:
        response = LLMRepairResponse.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - ValidationError 等一律转为警告
        return None, [f"LLM 返回的 JSON 无法通过 Pydantic 校验：{exc}"]
    return response, []


def validate_path(path: str) -> str | None:
    """校验 path 为安全相对路径；返回错误描述，合法时返回 None。"""
    if not path or not path.strip():
        return f"path 为空：{path!r}"
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        return f"path 不能是绝对路径：{path!r}"
    if _WINDOWS_ABSOLUTE.match(normalized):
        return f"path 不能是 Windows 绝对路径：{path!r}"
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return f"path 包含越界片段：{path!r}"
    return None
