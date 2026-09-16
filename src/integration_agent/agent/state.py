"""Planner 的输入状态与证据记录。

PlannerState 严格区分两类输入：
    - api：第三方 API 的事实（来自 api/parser.py）
    - project：目标项目的结构事实（来自 repository/scanner.py）

证据（evidence）只保存**受长度限制**的代码检索结果，不保存仓库快照，
从结构上杜绝"把整个仓库塞给 LLM"。
"""

from pydantic import BaseModel, Field

from integration_agent.api import APIInfo
from integration_agent.repository import CodeMatch, ProjectStructure, SearchResult


class CodeEvidence(BaseModel):
    """一次受控代码检索的结果。"""

    query: str
    matches: list[CodeMatch] = Field(default_factory=list)
    files_scanned: int = 0
    truncated: bool = False

    @property
    def paths(self) -> list[str]:
        """命中文件的去重路径，保持出现顺序。"""
        unique: list[str] = []
        for match in self.matches:
            if match.path not in unique:
                unique.append(match.path)
        return unique

    @property
    def total_chars(self) -> int:
        """本次证据占用的字符数，用于约束交给 LLM 的上下文体积。"""
        return sum(len(match.snippet) for match in self.matches)


class PlannerState(BaseModel):
    """贯穿 Planner 的输入状态。

    api 与 project 是两个独立的信息来源，Planning 过程只在 evidence 中追加
    检索记录，不会修改 api / project，也不会触碰目标仓库的磁盘内容。
    """

    api: APIInfo
    project: ProjectStructure
    request: str = ""  # 用户原始需求，可为空
    evidence: list[CodeEvidence] = Field(default_factory=list)

    def record(self, result: SearchResult) -> CodeEvidence:
        """记录一次代码检索结果并返回新增的证据对象。"""
        evidence = CodeEvidence(
            query=result.query,
            matches=result.matches,
            files_scanned=result.files_scanned,
            truncated=result.truncated,
        )
        self.evidence.append(evidence)
        return evidence
