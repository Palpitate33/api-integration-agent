"""DeepSeek 真实 API smoke test（手动运行，pytest 不会执行）。

用法：
    uv run python scripts/smoke_deepseek.py

需要环境变量：
    DEEPSEEK_API_KEY（必需）
    DEEPSEEK_MODEL / DEEPSEEK_BASE_URL（可选）

只验证一件事：prompt → DeepSeek → 收到文本响应。
不会输出 API Key；不运行完整 Repair Loop。
"""

import sys

from integration_agent.repair import DeepSeekLLMClient


def main() -> int:
    try:
        client = DeepSeekLLMClient()
    except Exception as exc:  # noqa: BLE001 - smoke 脚本需要把任何失败转为可读输出
        print(f"[FAIL] 客户端初始化失败：{exc}")
        return 1
    try:
        text = client.generate("只回复四个字：你好世界")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 调用 DeepSeek 失败：{exc}")
        return 1
    print(f"[OK] 收到响应（{len(text)} 字符）：{text.strip()[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
