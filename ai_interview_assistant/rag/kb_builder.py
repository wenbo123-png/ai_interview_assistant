"""
知识库构建入口模块（KB Builder）

作用：
- 调用 VectorStoreService 完成固定知识库文档的加载、切分、向量化、入库
- 输出构建统计信息，便于调试与运维
- 作为后续 app/agent 初始化知识库时的统一入口
"""

from __future__ import annotations

from typing import Any

from ai_interview_assistant.rag.vector_store import VectorStoreService
from ai_interview_assistant.utils.logger_handler import logger


def build_knowledge_base(print_result: bool = True) -> dict[str, Any]:
    """
    构建固定知识库并返回统计结果。

    Args:
        print_result: 是否打印构建结果到控制台（方便手动运行时查看）

    Returns:
        dict: 构建统计信息，例如：
        {
            "total_files": 4,
            "loaded_files": 4,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 283
        }
    """
    logger.info("[KB构建] 开始构建固定知识库...")

    service = VectorStoreService()
    stats = service.load_documents()

    logger.info(f"[KB构建] 构建完成: {stats}")

    if print_result:
        print("=== [KB BUILD RESULT] ===")
        print(stats)

    return stats


def build_and_check() -> dict[str, Any]:
    """
    构建后做一个简单状态判定，便于脚本或CI调用。
    """
    stats = build_knowledge_base(print_result=False)

    # 这里做轻量健康检查：
    # 1) 只要失败文件数为 0 且总文件数>0，通常说明本次构建健康
    # 2) 如果 total_files 为 0，说明数据目录为空或未放入文件
    total_files = stats.get("total_files", 0)
    failed_files = stats.get("failed_files", 0)

    if total_files == 0:
        logger.warning("[KB构建] 未发现可构建文件，请检查 data 目录。")
    elif failed_files > 0:
        logger.warning("[KB构建] 构建中存在失败文件，请查看日志定位问题。")
    else:
        logger.info("[KB构建] 构建状态正常。")

    return stats


if __name__ == "__main__":
    """
    直接运行方式：
    python ai_interview_assistant/rag/kb_builder.py
    """
    result = build_and_check()
    print("=== [KB BUILD CHECK] ===")
    print(result)