"""User management service：既有业务模块（模拟）。

系统内已有的用户管理入口，当前只操作本地内存数据，还没有接任何第三方
API——这正是本次集成要补上的部分。Planner 会通过领域关键词检索
（user / management / users）发现本模块，把它列入 files_to_modify，
生成"接入新客户端、复用既有用户处理流程"的修改建议。
"""

from models import Profile, User


class UserService:
    """既有用户管理服务：管理本地用户数据。"""

    def __init__(self) -> None:
        self._users: dict[str, User] = {
            "u-1": User(
                id="u-1",
                name="Ada Lovelace",
                email="ada@example.com",
                profile=Profile(bio="counting", avatar_url=None),
            ),
        }

    def list_users(self) -> list[User]:
        """返回全部用户。"""
        return list(self._users.values())

    def get_user(self, user_id: str) -> User | None:
        """按 ID 查询用户。"""
        return self._users.get(user_id)

    def update_user_profile(self, user_id: str, profile: Profile) -> User | None:
        """更新用户资料；用户不存在时返回 None。"""
        user = self._users.get(user_id)
        if user is None:
            return None
        updated = User(id=user.id, name=user.name, email=user.email, profile=profile)
        self._users[user_id] = updated
        return updated
