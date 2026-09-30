"""User domain models：既有业务模块（模拟）。

这是"被集成目标项目"的既有模型层。APIForge 生成的客户端不会复制这些
结构——Integration Planner 会把本文件识别为可复用的既有模块，并在集成
方案里给出接入建议（见 output/integration_plan.json 的 files_to_modify）。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """用户公开资料。"""

    bio: str
    avatar_url: str | None = None


@dataclass(frozen=True)
class User:
    """系统内的用户实体。"""

    id: str
    name: str
    email: str
    profile: Profile | None = None
