"""
文件处理工具：
- 文件类型筛选
- 文档加载（txt/pdf）
- 文件 MD5 计算
"""

from __future__ import annotations
import hashlib
import os
from typing import Sequence
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from ai_interview_assistant.utils.logger_handler import logger


def get_file_md5_hex(file_path: str) -> str | None:
    """计算文件 MD5，失败返回 None。"""
    if not os.path.exists(file_path):
        logger.error(f"[md5] 文件不存在: {file_path}")
        return None

    if not os.path.isfile(file_path):
        logger.error(f"[md5] 不是文件: {file_path}")
        return None

    md5_obj = hashlib.md5()     # 创建md5对象
    chunk_size = 4096       # 4KB分片，避免文件过大爆内存

    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"[md5] 计算失败: path={file_path}, err={str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: Sequence[str]) -> tuple[str, ...]:
    """
    返回目录下符合后缀的文件绝对路径。
    allowed_types 例子: ("txt", "pdf") 或 [".txt", ".pdf"]
    """
    files: list[str] = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] 不是目录: {path}")
        return tuple(files)

    # 统一后缀格式，兼容 "txt" 和 ".txt"
    normalized = tuple(
        t if t.startswith(".") else f".{t}"
        for t in allowed_types
    )

    for name in os.listdir(path):
        abs_path = os.path.join(path, name)
        if not os.path.isfile(abs_path):
            continue
        if name.lower().endswith(tuple(s.lower() for s in normalized)):
            files.append(abs_path)

    return tuple(files)


def pdf_loader(file_path: str, passwd: str | None = None) -> list[Document]:
    """加载 PDF 文档。"""
    return PyPDFLoader(file_path, password=passwd).load()


def txt_loader(file_path: str, encoding: str = "utf-8") -> list[Document]:
    """加载 TXT 文档。"""
    return TextLoader(file_path, encoding=encoding).load()