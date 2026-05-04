"""
为整个工程提供统一路径工具。
"""
from __future__ import annotations

from pathlib import Path


def get_project_root() -> str:
    """
    获取项目根目录。

    当前文件位于：ai_interview_assistant/utils/path_tool.py
    因此需要向上两级，才能回到真正的项目根目录：AI面试准备助手项目/
    """
    # Path(__file__).resolve() -> .../ai_interview_assistant/utils/path_tool.py
    # parents[0] = utils/
    # parents[1] = ai_interview_assistant/
    # parents[2] = 项目根目录
    return str(Path(__file__).resolve().parents[2])


def get_abs_path(path_str: str) -> str:
    """
    获取绝对路径：
    - 如果传入已是绝对路径，直接返回
    - 如果传入相对路径，按项目根目录拼接
    """
    if Path(path_str).is_absolute():
        return str(Path(path_str))

    project_root = get_project_root()
    return str(Path(project_root) / path_str)
