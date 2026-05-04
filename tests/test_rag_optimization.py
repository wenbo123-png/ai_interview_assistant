"""
RAG 检索优化测试脚本

测试内容：
1. BM25Retriever 分词和检索
2. 混合检索器（BM25 + Vector）
3. 查询改写
4. Reranking（如已安装 sentence-transformers）
5. 完整 RAG 问答链路对比
"""

import time
from ai_interview_assistant.rag.vector_store import VectorStoreService, BM25Retriever
from ai_interview_assistant.rag.rag_service import RagService
from ai_interview_assistant.utils.logger_handler import logger


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_bm25_tokenize():
    """测试 jieba 分词效果。"""
    separator("1. BM25 分词测试")
    texts = [
        "RAG检索增强生成的原理是什么",
        "大模型工程师面试常见问题",
        "请介绍一下Agent开发的核心技术",
    ]
    for text in texts:
        tokens = BM25Retriever._tokenize(text)
        print(f"  原文: {text}")
        print(f"  分词: {tokens}")
        print()


def test_bm25_retrieval():
    """测试 BM25 检索器。"""
    separator("2. BM25 检索测试")
    service = VectorStoreService()
    all_docs = service._get_all_documents_from_store()
    if not all_docs:
        print("  [SKIP] 向量库中无文档，请先运行 kb_builder 构建知识库")
        return

    print(f"  文档总数: {len(all_docs)}")
    retriever = BM25Retriever.from_documents(all_docs, k=4)

    queries = ["RAG 原理", "Agent 开发", "面试准备建议"]
    for q in queries:
        docs = retriever.invoke(q)
        print(f"\n  查询: {q}")
        print(f"  命中: {len(docs)} 条")
        for i, doc in enumerate(docs[:2], 1):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    [{i}] {preview}...")


def test_hybrid_retriever():
    """测试混合检索器。"""
    separator("3. 混合检索测试（BM25 + Vector）")
    service = VectorStoreService()
    try:
        retriever = service.get_hybrid_retriever()
    except Exception as e:
        print(f"  [FAIL] 混合检索器创建失败: {e}")
        return

    queries = ["RAG 技术原理", "大模型面试题", "如何准备AI岗位面试"]
    for q in queries:
        start = time.time()
        docs = retriever.invoke(q)
        elapsed = time.time() - start
        print(f"\n  查询: {q}")
        print(f"  命中: {len(docs)} 条 | 耗时: {elapsed:.2f}s")
        for i, doc in enumerate(docs[:2], 1):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    [{i}] {preview}...")


def test_hybrid_with_reranker():
    """测试混合检索 + Reranking。"""
    separator("4. 混合检索 + Reranking 测试")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("  [SKIP] sentence-transformers 未安装，跳过 Reranking 测试")
        print("  安装命令: pip install sentence-transformers")
        return

    service = VectorStoreService()
    try:
        retriever = service.get_hybrid_retriever_with_reranker()
    except Exception as e:
        print(f"  [FAIL] Reranker 检索器创建失败: {e}")
        return

    q = "RAG 检索增强生成的核心流程"
    start = time.time()
    docs = retriever.invoke(q)
    elapsed = time.time() - start
    print(f"  查询: {q}")
    print(f"  精排后命中: {len(docs)} 条 | 耗时: {elapsed:.2f}s")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i}] {preview}...")


def test_query_rewrite():
    """测试查询改写。"""
    separator("5. 查询改写测试")
    print("  正在初始化 RagService（含 Reranker 模型加载）...")
    service = RagService()
    if not service.query_rewrite_enabled:
        print("  [SKIP] 查询改写未启用（config/chroma.yml 中 query_rewriting.enabled=false）")
        return

    queries = [
        "帮我介绍一下RAG是什么东西",
        "大模型工程师面试一般会问啥",
        "我想练习一下面试",
        "Agent和RAG有什么区别",
    ]
    for q in queries:
        start = time.time()
        rewritten = service._rewrite_query(q)
        elapsed = time.time() - start
        print(f"  原始: {q}")
        print(f"  改写: {rewritten}")
        print(f"  耗时: {elapsed:.2f}s")
        print()


def test_full_rag_comparison():
    """完整 RAG 问答对比：用原始 query vs 改写后 query。"""
    separator("6. 完整 RAG 问答对比")
    print("  正在初始化 RagService（复用已有实例可跳过模型加载）...")
    service = RagService()

    queries = [
        "RAG技术怎么用",
        "面试的时候怎么准备算法题",
    ]

    for q in queries:
        print(f"\n  原始问题: {q}")

        # 改写
        rewritten = service._rewrite_query(q)
        if rewritten != q:
            print(f"  改写后:   {rewritten}")

        # 检索
        docs = service.retrieve_docs(rewritten)
        print(f"  检索命中: {len(docs)} 条")

        # 生成回答
        start = time.time()
        result = service.answer_with_rag(q)
        elapsed = time.time() - start
        print(f"  回答模式: {result.mode}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  回答预览: {result.answer[:150]}...")


if __name__ == "__main__":
    print("RAG 检索优化测试")
    print("请确保已安装: pip install jieba rank-bm25")
    print("可选安装: pip install sentence-transformers (用于 Reranking)")

    test_bm25_tokenize()
    test_bm25_retrieval()
    test_hybrid_retriever()
    test_hybrid_with_reranker()
    test_query_rewrite()
    test_full_rag_comparison()
