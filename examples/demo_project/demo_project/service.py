"""占位服务模块：后续由 APIForge 集成真实第三方 API 调用。"""


def get_user_profile(user_id: str) -> dict:
    """返回占位数据；集成完成后将替换为真实 API 响应。"""
    return {"id": user_id, "name": "placeholder"}
