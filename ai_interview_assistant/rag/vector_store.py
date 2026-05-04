from __future__ import annotations

import os
import re
from typing import Any

import jieba
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from ai_interview_assistant.model.factory import embed_model
from ai_interview_assistant.utils.config_handler import chroma_conf
from ai_interview_assistant.utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.path_tool import get_abs_path


# =========================
# BM25 检索器（基于中文分词的关键词检索）
# =========================
class BM25Retriever(BaseRetriever):
    """
    基于 BM25 的关键词检索器，使用 jieba 中文分词。
    与向量检索互补：向量擅长语义匹配，BM25 擅长精确关键词匹配。
    """

    bm25: Any = None
    documents: list[Document] = []
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 中文分词，过滤空白 token。"""
        return [t for t in jieba.cut(text) if t.strip()]

    @classmethod
    def from_documents(cls, documents: list[Document], k: int = 4) -> "BM25Retriever":
        """从文档列表构建 BM25 索引。"""
        tokenized_docs = [cls._tokenize(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        return cls(bm25=bm25, documents=documents, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """BM25 检索，返回得分最高的 top-k 文档。"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
        return [self.documents[i] for i in top_indices]


class VectorStoreService:
    """
    固定知识库向量服务：
    - 读取 data/ 下的知识文件（txt/pdf）
    - 切分为 chunk
    - 写入 Chroma 向量库
    - 通过 md5 记录避免重复入库
    """

    def __init__(self) -> None:
        self.collection_name: str = chroma_conf["collection_name"]
        self.persist_directory: str = get_abs_path(chroma_conf["persist_directory"])
        self.data_path: str = get_abs_path(chroma_conf["data_path"])
        self.md5_store_path: str = get_abs_path(chroma_conf["md5_hex_store"])
        self.allowed_types: tuple[str, ...] = tuple(chroma_conf["allow_knowledge_file_type"])

        # 初始化向量库
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embed_model,
            persist_directory=self.persist_directory,
        )

        # 初始化文本切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        # 确保向量目录存在（避免部分环境下目录不存在）
        os.makedirs(self.persist_directory, exist_ok=True)

    def get_retriever(self):
        """返回向量检索器。"""
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _get_all_documents_from_store(self) -> list[Document]:
        """从已持久化的 Chroma 向量库中读取全部文档，用于构建 BM25 索引。"""
        try:
            collection = self.vector_store._collection
            result = collection.get(include=["documents", "metadatas"])
            docs = []
            for content, meta in zip(result["documents"], result["metadatas"]):
                docs.append(Document(page_content=content, metadata=meta or {}))
            return docs
        except Exception as e:
            logger.warning(f"[VectorStore] 读取向量库文档失败: {e}")
            return []

    @staticmethod
    def _create_ensemble_retriever(retrievers, weights):
        """
        创建混合检索器，兼容不同 langchain 版本。
        优先使用 langchain.retrievers.ensemble.EnsembleRetriever，
        若不存在则使用 langchain_community 或手动实现。
        """
        try:
            from langchain.retrievers.ensemble import EnsembleRetriever
            return EnsembleRetriever(retrievers=retrievers, weights=weights)
        except ImportError:
            pass

        try:
            from langchain_community.retrievers import EnsembleRetriever
            return EnsembleRetriever(retrievers=retrievers, weights=weights)
        except ImportError:
            pass

        # 手动实现：按权重融合多路检索结果并去重
        logger.warning("[VectorStore] EnsembleRetriever 不可用，使用手动融合实现")

        class ManualEnsembleRetriever(BaseRetriever):
            retrievers_list: Any = None
            weights_list: Any = None

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                seen_contents: set[str] = set()
                merged: list[Document] = []
                # 按权重轮流取结果，实现简单的交错融合
                all_results = []
                for retriever, weight in zip(self.retrievers_list, self.weights_list):
                    docs = retriever.invoke(query)
                    for i, doc in enumerate(docs):
                        # 越靠前的文档得分越高，用权重加成
                        score = weight * (1.0 - i * 0.1)
                        all_results.append((score, doc))
                # 按融合得分排序去重
                all_results.sort(key=lambda x: x[0], reverse=True)
                for _, doc in all_results:
                    content = doc.page_content.strip()
                    if content not in seen_contents:
                        seen_contents.add(content)
                        merged.append(doc)
                return merged

        return ManualEnsembleRetriever(
            retrievers_list=retrievers,
            weights_list=weights,
        )

    def get_hybrid_retriever(self):
        """
        构建混合检索器：BM25 关键词检索 + 向量语义检索。
        两者通过 EnsembleRetriever 按权重融合。
        """
        hybrid_conf = chroma_conf.get("hybrid_search", {})
        bm25_weight = hybrid_conf.get("bm25_weight", 0.3)
        vector_weight = hybrid_conf.get("vector_weight", 0.7)
        k = chroma_conf["k"]

        # 向量检索器
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # BM25 检索器：从向量库中读取全部文档构建索引
        all_docs = self._get_all_documents_from_store()
        if not all_docs:
            logger.warning("[VectorStore] 无文档可构建 BM25，降级为纯向量检索")
            return vector_retriever

        bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
        logger.info(f"[VectorStore] BM25 索引构建完成，文档数={len(all_docs)}")

        # 混合检索器（兼容不同 langchain 版本的导入路径）
        ensemble = self._create_ensemble_retriever(
            [bm25_retriever, vector_retriever],
            [bm25_weight, vector_weight],
        )
        logger.info(
            f"[VectorStore] 混合检索器就绪 | bm25_weight={bm25_weight} vector_weight={vector_weight}"
        )
        return ensemble

    @staticmethod
    def _dashscope_rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
        """
        调用 DashScope Reranking API 对候选文档精排。
        使用 gte-rerank 模型，无需本地加载，通过 API 调用。
        """
        if not docs:
            return []
        try:
            import dashscope
            from dashscope import TextReRank

            # 构造 API 输入：每条文档的文本
            documents_text = [doc.page_content for doc in docs]

            response = TextReRank.call(
                model="gte-rerank",
                query=query,
                documents=documents_text,
                top_n=top_n,
                return_documents=False,
            )

            if response.status_code != 200:
                logger.warning(f"[VectorStore] DashScope Rerank API 错误: {response.code} {response.message}")
                return docs[:top_n]

            # 按 API 返回的索引重排文档
            results = response.output.get("results", [])
            reranked = []
            for item in results:
                idx = item.get("index", -1)
                if 0 <= idx < len(docs):
                    reranked.append(docs[idx])
            return reranked if reranked else docs[:top_n]

        except ImportError:
            logger.warning("[VectorStore] dashscope SDK 未安装，跳过 Reranking")
            return docs[:top_n]
        except Exception as e:
            logger.warning(f"[VectorStore] DashScope Rerank 调用失败: {e}，降级为粗召回结果")
            return docs[:top_n]

    def get_hybrid_retriever_with_reranker(self):
        """
        混合检索 + DashScope API 精排。
        先用 BM25+Vector 粗召回较多候选，再调用 DashScope Reranking API 精排。
        无需本地加载模型，零等待。
        """
        rerank_conf = chroma_conf.get("reranking", {})
        if not rerank_conf.get("enabled", False):
            return self.get_hybrid_retriever()

        top_n = rerank_conf.get("top_n", 4)
        coarse_k = max(top_n * 3, 12)

        hybrid_conf = chroma_conf.get("hybrid_search", {})
        bm25_weight = hybrid_conf.get("bm25_weight", 0.3)
        vector_weight = hybrid_conf.get("vector_weight", 0.7)

        # 粗召回阶段用更大的 k
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": coarse_k})
        all_docs = self._get_all_documents_from_store()

        if not all_docs:
            logger.warning("[VectorStore] 无文档可构建 BM25，降级为纯向量检索")
            return vector_retriever

        bm25_retriever = BM25Retriever.from_documents(all_docs, k=coarse_k)

        ensemble = self._create_ensemble_retriever(
            [bm25_retriever, vector_retriever],
            [bm25_weight, vector_weight],
        )

        # 包装一层 DashScope Reranking 精排
        rerank_fn = self._dashscope_rerank

        class DashScopeRerankedRetriever(BaseRetriever):
            """先粗召回，再通过 DashScope API 精排，零本地模型加载。"""
            base_retriever: Any = None
            top_n: int = 4
            rerank_fn: Any = None

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                candidates = self.base_retriever.invoke(query)
                if not candidates:
                    return []
                return self.rerank_fn(query, candidates, self.top_n)

        logger.info(f"[VectorStore] 混合检索 + DashScope Reranking 就绪 | top_n={top_n}")
        return DashScopeRerankedRetriever(
            base_retriever=ensemble,
            top_n=top_n,
            rerank_fn=rerank_fn,
        )

    def _ensure_md5_store_file(self) -> None:
        """确保 md5 记录文件存在。"""
        md5_dir = os.path.dirname(self.md5_store_path)
        if md5_dir:
            os.makedirs(md5_dir, exist_ok=True)

        if not os.path.exists(self.md5_store_path):
            open(self.md5_store_path, "w", encoding="utf-8").close()

    def _check_md5_exists(self, md5_hex: str) -> bool:
        """检查 md5 是否已记录。"""
        self._ensure_md5_store_file()
        with open(self.md5_store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_hex:
                    return True
        return False

    def _save_md5(self, md5_hex: str) -> None:
        """保存 md5 记录，避免重复入库。"""
        self._ensure_md5_store_file()
        with open(self.md5_store_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")

    @staticmethod
    def _load_documents_by_path(file_path: str) -> list[Document]:
        """按后缀加载文档。"""
        lower_path = file_path.lower()

        if lower_path.endswith(".txt"):
            return txt_loader(file_path)

        if lower_path.endswith(".pdf"):
            return pdf_loader(file_path)

        return []

    def load_documents(self) -> dict[str, int]:
        """
        从 data_path 加载知识库文档并写入向量库。
        返回处理统计信息，便于日志和外部展示。
        """
        # 目录不存在时直接返回统计信息，不抛异常，便于脚本化调用
        if not os.path.isdir(self.data_path):
            logger.warning(f"[向量库构建] 数据目录不存在: {self.data_path}")
            return {
                "total_files": 0,
                "loaded_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
            }

        file_paths = listdir_with_allowed_type(self.data_path, self.allowed_types)

        stats = {
            "total_files": len(file_paths),
            "loaded_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
        }

        if not file_paths:
            logger.info(f"[向量库构建] 数据目录为空或无可用文件: {self.data_path}")
            return stats

        for path in file_paths:
            md5_hex = get_file_md5_hex(path)

            if not md5_hex:
                logger.warning(f"[向量库构建] 无法计算 md5，跳过: {path}")
                stats["skipped_files"] += 1
                continue

            if self._check_md5_exists(md5_hex):
                logger.info(f"[向量库构建] 文件已入库，跳过: {path}")
                stats["skipped_files"] += 1
                continue

            try:
                documents = self._load_documents_by_path(path)
                if not documents:
                    logger.warning(f"[向量库构建] 文件无有效内容，跳过: {path}")
                    stats["skipped_files"] += 1
                    continue

                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[向量库构建] 文本切分后为空，跳过: {path}")
                    stats["skipped_files"] += 1
                    continue

                self.vector_store.add_documents(split_documents)
                self._save_md5(md5_hex)

                stats["loaded_files"] += 1
                stats["total_chunks"] += len(split_documents)
                logger.info(
                    f"[向量库构建] 入库成功: {path} | chunks={len(split_documents)}"
                )

            except Exception as e:
                stats["failed_files"] += 1
                logger.error(f"[向量库构建] 入库失败: {path} | err={str(e)}", exc_info=True)

        logger.info(f"[向量库构建] 完成: {stats}")
        return stats

    # 为兼容旧调用名保留别名，后续可逐步统一为 load_documents
    def load_document(self) -> dict[str, int]:
        return self.load_documents()


if __name__ == "__main__":
    """
    本地最小测试：
    1) 初始化向量服务
    2) 执行知识库构建
    3) 执行检索测试
    """
    print("=== [TEST] VectorStoreService 测试开始 ===")

    try:
        service = VectorStoreService()
        print("[OK] VectorStoreService 初始化成功")
        print(f"[INFO] collection_name: {service.collection_name}")
        print(f"[INFO] data_path: {service.data_path}")
        print(f"[INFO] persist_directory: {service.persist_directory}")
        print(f"[INFO] md5_store_path: {service.md5_store_path}")
        print(f"[INFO] allowed_types: {service.allowed_types}")
    except Exception as e:
        print(f"[FAIL] 初始化失败: {e}")
        raise

    # 1) 执行知识库构建
    try:
        build_stats = service.load_documents()
        print("\n=== [TEST] 构建结果 ===")
        print(build_stats)

        # 简单判定
        if build_stats["loaded_files"] > 0:
            print("[OK] 至少有文件成功入库")
        elif build_stats["total_files"] == 0:
            print("[WARN] data 目录没有可用文件，请确认已放入 txt/pdf")
        else:
            print("[WARN] 有文件但未成功入库，请检查日志和文件内容")
    except Exception as e:
        print(f"[FAIL] 知识库构建失败: {e}")
        raise

    # 2) 检索测试（只要 retriever 能创建并返回列表，就说明链路基本通）
    try:
        retriever = service.get_retriever()
        print("\n[OK] retriever 创建成功")

        test_queries = [
            "agent开发面试常见问题",
            "请给我一些大模型岗位的面试题",
            "如何准备RAG相关岗位面试",
        ]

        print("\n=== [TEST] 检索结果 ===")
        for q in test_queries:
            docs = retriever.invoke(q)
            print(f"\n[QUERY] {q}")
            print(f"[HITS] 命中文档数: {len(docs)}")

            # 只展示前2条，避免输出太长
            for i, doc in enumerate(docs[:2], 1):
                preview = doc.page_content[:120].replace("\n", " ")
                print(f"  - Doc{i}: {preview}...")
    except Exception as e:
        print(f"[FAIL] 检索测试失败: {e}")
        raise

    print("\n=== [TEST] VectorStoreService 测试结束 ===")