"""
日志处理模块：提供统一日志记录能力（控制台 + 文件）。

日志格式：
- 控制台：精简格式，方便开发时快速扫读
  示例：14:30:05 [INFO] agent: 意图识别完成
- 文件：详细格式，方便排查问题
  示例：2025-05-04 14:30:05 | INFO | agent | react_agent.py:180 | 意图识别完成
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from ai_interview_assistant.utils.path_tool import get_abs_path


# 日志目录
LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

# 控制台精简格式：时间 [级别] 模块: 消息
CONSOLE_FORMAT = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# 文件详细格式：完整时间 | 级别 | 模块 | 文件:行号 | 消息
FILE_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _has_custom_handlers(logger_obj: logging.Logger) -> bool:
    """判断当前 logger 是否已经挂载过本模块创建的 handler。"""
    for h in logger_obj.handlers:
        if getattr(h, "_interview_logger_handler", False):
            return True
    return False


def get_logger(
    name: str = "interview_assistant",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    获取 logger：
    - 控制台：精简格式，只看关键信息
    - 文件：详细格式，完整上下文
    - 自动避免重复添加 handler
    """
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.DEBUG)
    logger_obj.propagate = False

    if _has_custom_handlers(logger_obj):
        return logger_obj

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(CONSOLE_FORMAT)
    console_handler._interview_logger_handler = True
    logger_obj.addHandler(console_handler)

    # 文件 handler（按日期分文件）
    if not log_file:
        log_file = os.path.join(
            LOG_ROOT,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log",
        )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(FILE_FORMAT)
    file_handler._interview_logger_handler = True
    logger_obj.addHandler(file_handler)

    return logger_obj


# 快捷全局日志器
logger = get_logger()
