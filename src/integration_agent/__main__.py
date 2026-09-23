"""``python -m integration_agent <command>`` 的入口。

真正的实现在 cli/ 包里；这里只做"模块可执行"这一层，保持入口与实现分开，
`python -m integration_agent demo` 和测试里直接调 cli.main() 走的是同一份代码。
"""

import sys

from integration_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
