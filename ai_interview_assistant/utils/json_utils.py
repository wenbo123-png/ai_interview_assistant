"""
JSON 工具模块（MVP 版）

用途：
1) 解析 LLM 返回的 JSON（含不规范格式容错）
2) 从 Markdown/普通文本中提取 JSON
3) 做基础结构校验与默认值补齐
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


# 匹配 markdown 代码块中的 json 内容，如：
# ```json
# {...}
# ```
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def loads_json_safe(text: str, default: Any = None) -> Any:
    """
    安全解析 JSON：
    - 解析成功返回对象
    - 解析失败返回 default（默认 None）
    """
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def dumps_json_pretty(data: Any, ensure_ascii: bool = False) -> str:
    """格式化输出 JSON 字符串，便于日志和调试。"""
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=2)


def normalize_json_text(raw_text: str) -> str:
    """
    轻量清洗 JSON 文本，修复常见问题：
    - 中文引号 -> 英文引号
    - 去掉尾逗号（对象/数组）
    """
    text = (raw_text or "").strip()
    if not text:
        return text

    # 替换中文引号
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 去掉对象/数组中的尾逗号
    # 例：{"a":1,} -> {"a":1}
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text


def extract_json_from_markdown(text: str) -> str | None:
    """
    从 markdown 代码块中提取第一个 JSON 片段。
    """
    if not text:
        return None

    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None

    candidate = m.group(1).strip()
    return candidate if candidate else None


def extract_first_json_fragment(text: str) -> str | None:
    """
    从普通文本中提取第一个看起来完整的 JSON 片段（对象或数组）。
    使用括号计数法，支持嵌套。
    """
    if not text:
        return None

    s = text.strip()

    # 优先找对象
    obj = _extract_balanced_fragment(s, "{", "}")
    if obj:
        return obj

    # 找不到对象再找数组
    arr = _extract_balanced_fragment(s, "[", "]")
    return arr


def _extract_balanced_fragment(text: str, open_ch: str, close_ch: str) -> str | None:
    """
    从字符串中提取首个配对完整的括号片段（支持字符串内转义）。
    """
    start = text.find(open_ch)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        # 不在字符串中
        if ch == '"':
            in_string = True
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_llm_json_response(text: str, default: Any = None) -> Any:
    """
    面向 LLM 输出的 JSON 解析入口（推荐主函数）：
    1) 先尝试整体解析
    2) 再尝试从 markdown 代码块提取
    3) 再尝试从普通文本提取首个 JSON 片段
    4) 每一步都做轻量 normalize
    """
    if not text:
        return default

    # 1) 直接解析
    normalized = normalize_json_text(text)
    parsed = loads_json_safe(normalized, default=None)
    if parsed is not None:
        return parsed

    # 2) 从 markdown 代码块提取后解析
    block = extract_json_from_markdown(text)
    if block:
        parsed = loads_json_safe(normalize_json_text(block), default=None)
        if parsed is not None:
            return parsed

    # 3) 从普通文本提取首个 JSON 片段
    frag = extract_first_json_fragment(text)
    if frag:
        parsed = loads_json_safe(normalize_json_text(frag), default=None)
        if parsed is not None:
            return parsed

    return default


def ensure_dict(data: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """确保返回 dict，失败返回 default 或 {}。"""
    if isinstance(data, dict):
        return data
    return default.copy() if default is not None else {}


def ensure_list(data: Any, default: list[Any] | None = None) -> list[Any]:
    """确保返回 list，失败返回 default 或 []。"""
    if isinstance(data, list):
        return data
    return list(default) if default is not None else []

def ensure_text(value, default=""):
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)

def fill_defaults(data: dict[str, Any], required_fields: dict[str, Any]) -> dict[str, Any]:
    """
    按 required_fields 补齐缺失字段，不覆盖已有字段。
    """
    result = dict(data or {})
    for k, v in required_fields.items():
        if k not in result:
            result[k] = v
    return result


def validate_required_keys(data: dict[str, Any], required_keys: Iterable[str]) -> tuple[bool, list[str]]:
    """
    校验必填字段是否存在。
    返回：(是否通过, 缺失字段列表)
    """
    missing = [k for k in required_keys if k not in data]
    return (len(missing) == 0, missing)