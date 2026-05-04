"""
模型工厂

职责：
- 创建并缓存 Chat 模型和 Embedding 模型实例
- 为 Chat 模型添加 tenacity 重试，提升调用稳定性
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ai_interview_assistant.utils.config_handler import rag_conf
from ai_interview_assistant.utils.logger_handler import logger


# =========================
# 重试装饰器
# =========================
LLM_RETRY = retry(
    stop=stop_after_attempt(3),                      # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避：1s → 2s → 4s
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        f"[LLM] 调用失败，第 {retry_state.attempt_number} 次重试: "
        f"{retry_state.outcome.exception() if retry_state.outcome else 'unknown'}"
    ),
    reraise=True,
)


def _create_retry_model(model_name: str) -> ChatTongyi:
    """
    创建带重试能力的 ChatTongyi 模型。

    直接对模型类的 invoke / generate 方法做猴子补丁，
    因为 ChatTongyi 是 Pydantic 模型，不能用包装类替代。
    """
    model = ChatTongyi(model=model_name)

    # 保存原始方法
    _original_invoke = ChatTongyi.invoke
    _original_generate = ChatTongyi.generate

    # 只补丁一次，避免重复包装
    if not getattr(ChatTongyi, "_retry_patched", False):
        @LLM_RETRY
        def retry_invoke(self, *args: Any, **kwargs: Any) -> Any:
            return _original_invoke(self, *args, **kwargs)

        @LLM_RETRY
        def retry_generate(self, *args: Any, **kwargs: Any) -> Any:
            return _original_generate(self, *args, **kwargs)

        ChatTongyi.invoke = retry_invoke  # type: ignore[assignment]
        ChatTongyi.generate = retry_generate  # type: ignore[assignment]
        ChatTongyi._retry_patched = True  # type: ignore[attr-defined]
        logger.info("[Factory] ChatTongyi 重试补丁已应用")

    return model


class BaseModelFactory:
    """模型工厂基类。"""

    def generator(self) -> Optional[Embeddings | Any]:
        pass


class ChatModelFactory(BaseModelFactory):
    """Chat 模型工厂，自动附加重试。"""

    def generator(self) -> Optional[Embeddings | Any]:
        return _create_retry_model(rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    """Embedding 模型工厂。"""

    def generator(self) -> Optional[Embeddings | Any]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 全局单例：所有工具共享同一个模型实例
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
