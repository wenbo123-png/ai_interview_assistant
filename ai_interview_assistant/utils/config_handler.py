"""
配置读取模块
统一加载 config/ 下的 yaml 配置文件。
"""
from __future__ import annotations
import os
import yaml
from typing import Any
from ai_interview_assistant.utils.path_tool import get_abs_path


def _load_yaml(config_path: str, encoding: str = "utf-8") -> dict[str, Any]:
    """
    通用 YAML 读取函数。
    - 文件不存在：抛出 FileNotFoundError
    - 文件为空：返回 {}
    - 内容不是 dict：抛出 ValueError
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding=encoding) as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"配置文件内容必须是键值结构(dict): {config_path}")

    return data


def load_rag_config(
    config_path: str = get_abs_path("config/rag.yml"),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return _load_yaml(config_path, encoding)


def load_chroma_config(
    config_path: str = get_abs_path("config/chroma.yml"),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return _load_yaml(config_path, encoding)


def load_prompts_config(
    config_path: str = get_abs_path("config/prompts.yml"),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return _load_yaml(config_path, encoding)


def load_agent_config(
    config_path: str = get_abs_path("config/agent.yml"),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return _load_yaml(config_path, encoding)


def load_interview_config(
    config_path: str = get_abs_path("config/interview.yml"),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return _load_yaml(config_path, encoding)


def load_app_config(
    config_path: str = get_abs_path("config/app.yml"),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    return _load_yaml(config_path, encoding)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
interview_conf = load_interview_config()
app_conf = load_app_config()


