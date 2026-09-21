"""把不可信文本安全地渲染成 Python 源码片段。

威胁模型
--------
OpenAPI 文档里的 ``info.title`` / ``summary`` / ``description`` / ``operationId`` /
``path`` / 参数名全部来自**不可信输入**，而 Code Generator 必须把它们写进生成代码。
朴素拼接（把文本直接塞进一对三引号之间）会让一段普通文本改变生成代码的
**语法结构**：文本里只要带上三引号闭合与换行，后面接的就变成真实语句了，
而且语法完全合法。

所以黑名单在这里是无效防线：它挡的是**字符串**，挡不住**语法**。
把 ``os.system`` 换成 ``subprocess`` / ``eval`` / ``__import__`` / 十六进制拼串，
黑名单的每一条都能被绕过，而攻击者本来就可以随便换。

对策：白名单式的三条渲染路径
----------------------------
    string_literal(text)  —— 任意文本 → 一个 Python 字符串字面量（表达式位置可用）
    docstring(text)       —— 任意文本 → 一条 docstring 语句（可带缩进）
    identifier(text)      —— 任意文本 → 一个合法标识符（名字位置用）

保底渲染器用标准库 ``json``：JSON 字符串与 Python 字符串字面量在这里是兼容子集
（``\\"`` ``\\\\`` ``\\b`` ``\\f`` ``\\n`` ``\\r`` ``\\t`` ``\\uXXXX`` 两边都认），
而 json 是被充分测试过的实现。手写转义表等于维护一份迟早会漏的黑名单——
这里改用标准库的既有保证，它不随攻击手法变化。

注意 json 的转义不只是"更全"，它同时抹掉了**行结构**：换行变成 ``\\n`` 两个字符，
所以一个字面量永远落在一行里，不可能把后面的文本挤到新的一行去。

可读性只在**能证明安全**时才启用（``_is_triple_quote_safe``）：文本不含引号、
反斜杠、控制字符时，才原样输出三引号形式。这一支纯粹是装饰——判定不通过就
回退到 ``string_literal()``。**回退路径是通用正确的，不是补丁**：特殊字符一个
都没被禁止，只是被转义了，正文与原文仍然逐字相等。

标识符（``identifier``）是另一回事：它没法"转义"，只能改写。所以那里做的是
收敛——非法字符折叠、数字开头补前缀、关键字补后缀——影响的只是**名字**，
进入字面量的语义信息一律不经过它。
"""

import json
import keyword
import re

# Python 标识符只允许这些字符（首字符另有约束，见 identifier）
_NON_IDENTIFIER_CHARS = re.compile(r"[^0-9a-zA-Z_]")
_LEADING_DIGIT = re.compile(r"^[0-9]")


def _strip_unrepresentable(text: str) -> str:
    """去掉无法写进 Python 源码的字符：孤立代理项（U+D800–U+DFFF）。

    代理项不是合法 UTF-8，这一条不是格式偏好：``json.dumps`` 能把它转成
    ``\\udXXX`` 让源码本身是 ASCII，但 CPython 在生成常量时会 UTF-8 编码这个
    字符串，``compile()`` 直接抛 ``UnicodeEncodeError: surrogates not allowed``。
    也就是说**没有任何写法**能让它变成可编译的 Python 常量；照抄只会产出
    一个语法上"看起来对"、一 import 就炸的文件。

    所以这里换成 U+FFFD：它是可表示的替代字符，也是解码非法字节时的标准做法。
    代价是这一个码位不再逐字相等——但它在生成文件里本来就无法存在，
    换成可见的替代字符比留一颗早晚会炸的雷更接近"保留语义"。

    逐字相等这个保证对**所有可表示的文本**仍然成立，包括换行、引号、反斜杠、
    控制字符、emoji：它们一个都没被丢弃，只是被转义。
    """
    return "".join("�" if 0xD800 <= ord(char) <= 0xDFFF else char for char in text)


def string_literal(text: str) -> str:
    """任意文本 → Python 字符串字面量（含引号，可直接放在表达式位置）。

    用双引号风格：生成代码既有风格就是双引号，转义后仍然保持可读。
    ``ensure_ascii=False``：中文是可表示的，没必要为了 ASCII 把它变成 ``\\uXXXX``
    （生成文件本来就按 UTF-8 写盘）；引号、反斜杠、换行这些**有语法含义**的字符
    仍然一律转义。
    """
    return json.dumps(_strip_unrepresentable(text), ensure_ascii=False)


def _is_triple_quote_safe(text: str) -> bool:
    """文本能否原样放进三引号字面量。

    白名单判定：只放行**能证明无害**的字符，出现任何一个拿不准的字符就返回 False。
    判据只有三条，都不依赖对攻击手法的枚举：

    1. 不含引号 —— 含引号就有凑出三引号闭合的可能；
    2. 不含反斜杠 —— 否则文本里的 ``\\n`` 会被解释成真换行，正文与原文不再逐字相等；
    3. 每个字符要么可打印，要么是换行 / 制表符 —— 其余字符（回车、垂直制表、
       NUL、Unicode 行分隔符 U+2028 等）在 Python 源码里另有行与语句语义。
    """
    if '"' in text or "\\" in text:
        return False
    return all(char.isprintable() or char in "\n\t" for char in text)


def docstring(text: str, indent: str = "") -> str:
    """任意文本 → 一条 docstring 语句（含缩进，可直接 append 进 lines）。

    多行文本在三引号形式下，续行保持原样、不额外缩进：三引号字符串的内容是
    逐字的，重新缩进会改动正文。缩进只加在起始那一行——Python 允许字符串
    字面量跨行，续行的列位置不影响语法。
    """
    if _is_triple_quote_safe(text):
        return f'{indent}"""{text}"""'
    return f"{indent}{string_literal(text)}"


def identifier(text: str, *, fallback: str = "value") -> str:
    """任意文本 → 合法的 Python 标识符。

    标识符必须是一个**名字**，不能像字符串那样转义，所以这里只能改写：
    非标识符字符折叠成下划线、数字开头补前缀、Python 关键字补后缀。

    fallback 用于文本被折叠成空串的情况（例如参数名全是标点）。有兜底名
    才能保证"每个参数都还有一个名字"，而不是凭空少掉一个形参——少一个形参
    会让方法体和调用方一起错位。
    """
    candidate = _NON_IDENTIFIER_CHARS.sub("_", text).strip("_")
    if not candidate:
        candidate = fallback
    if _LEADING_DIGIT.match(candidate):
        candidate = f"_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"
    return candidate


def module_path(dotted: str) -> str:
    """任意路径 → 合法的点分模块路径（用于 import 语句）。

    ``from X import Y`` 里的 X 是标识符序列，同样不能转义。逐段收敛，
    空段（``a..b`` / 结尾的点）用 fallback 补齐而不是原样保留——
    留一个空段就是一行 SyntaxError。
    """
    return ".".join(identifier(part, fallback="module") for part in dotted.split("."))
