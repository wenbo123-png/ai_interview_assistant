"""
文本处理工具（MVP 版）
用于：
1) 简历/JD 原文预处理
2) 构造 prompt 前的长度控制
3) 规则化关键词抽取（非模型版）
"""

from __future__ import annotations

import re
from typing import Iterable


# 零宽字符和 BOM，常见于复制粘贴后的脏文本
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# 多个空格/制表符压缩为单个空格（不影响中文）
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")


def normalize_newlines(text: str) -> str:
    """统一换行符为 \\n。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_zero_width_chars(text: str) -> str:
    """去除零宽字符/BOM，避免影响检索与分词。"""
    return _ZERO_WIDTH_PATTERN.sub("", text)


def collapse_spaces(text: str) -> str:
    """压缩行内多余空白。"""
    return _MULTI_SPACE_PATTERN.sub(" ", text)


def clean_text(text: str) -> str:
    """
    基础清洗：
    - 统一换行
    - 去除零宽字符
    - 去除每行首尾空白
    - 压缩行内空格
    - 合并连续空行（最多保留一行）
    """
    if not text:
        return ""

    text = normalize_newlines(text)
    text = remove_zero_width_chars(text)

    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = collapse_spaces(raw_line).strip()
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # 把 3 个及以上换行压缩成 2 个，保留段落结构
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def truncate_text(text: str, max_chars: int, ellipsis: str = "...[TRUNCATED]") -> str:
    """
    按字符长度截断文本，避免 prompt 超长。
    """
    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    keep_len = max(0, max_chars - len(ellipsis))
    return text[:keep_len].rstrip() + ellipsis


def split_by_paragraph(text: str, min_len: int = 1) -> list[str]:
    """
    按空行分段，过滤过短段落。
    """
    if not text:
        return []

    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if len(p) >= min_len]


def split_by_sentence(text: str, min_len: int = 2) -> list[str]:
    """
    按中英文句号/问号/感叹号做简易切句（MVP 规则版）。
    """
    if not text:
        return []

    # 保留句末标点，便于后续展示
    chunks = re.split(r"(?<=[。！？!?\.])\s*", text)
    return [c.strip() for c in chunks if len(c.strip()) >= min_len]


def extract_keywords_rule_based(
    text: str,
    stopwords: Iterable[str] | None = None,
    top_k: int = 20,
) -> list[str]:
    """
    规则关键词提取（MVP）：
    - 用于无模型场景下的快速关键词初筛
    - 支持英文词、数字、常见中文词块
    """
    if not text:
        return []

    stop = set(stopwords or [])

    # 英文/数字 token
    en_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-\+#\.]{1,}", text)
    # 简单中文词块（连续 2~8 个中文字符）
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,8}", text)

    token_counts: dict[str, int] = {}

    for token in en_tokens + zh_tokens:
        key = token.strip().lower()
        if not key or key in stop:
            continue
        token_counts[key] = token_counts.get(key, 0) + 1

    # 频次降序 + 词长降序（同频时更偏向信息量高的词）
    sorted_tokens = sorted(
        token_counts.items(),
        key=lambda x: (x[1], len(x[0])),
        reverse=True,
    )

    return [token for token, _ in sorted_tokens[: max(top_k, 0)]]


def prepare_resume_text(raw_text: str, max_chars: int = 8000) -> str:
    """
    简历文本预处理：
    先清洗，再截断，保证后续解析稳定。
    """
    cleaned = clean_text(raw_text)
    return truncate_text(cleaned, max_chars=max_chars)


def prepare_jd_text(raw_text: str, max_chars: int = 8000) -> str:
    """
    JD 文本预处理：
    与简历一致，便于统一输入下游模块。
    """
    cleaned = clean_text(raw_text)
    return truncate_text(cleaned, max_chars=max_chars)