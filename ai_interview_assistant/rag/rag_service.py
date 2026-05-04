"""
RAG 服务模块（面试助手版）

能力：
1) 知识库命中时：基于检索上下文回答（rag_hit）
2) 知识库未命中时：先提示，再走大模型通用兜底（fallback）
3) 无简历/JD的问答也先检索，再决定 rag_hit/fallback（qa_no_context 语义标签保留）
4) 支持面试题 + 参考答案检索（interview_qa）

说明：
- 普通知识问答继续走原有 RAG 问答逻辑
- 面试题/参考答案检索作为新增能力单独提供
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_interview_assistant.model.factory import chat_model
from ai_interview_assistant.rag.vector_store import VectorStoreService
from ai_interview_assistant.utils.config_handler import chroma_conf
from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.prompt_loader import load_rag_query_prompt, load_query_rewrite_prompt


@dataclass
class RagAnswerResult:
    """
    RAG 返回结构（用于普通问答）。
    """
    mode: str  # rag_hit / fallback / qa_no_context / reject
    answer: str
    notice: str
    retrieved_count: int
    retrieved_context: str
    meta: dict[str, Any]


@dataclass
class InterviewQAItem:
    """
    面试题检索结果项。
    """
    question: str
    reference_answer: str
    qtype: str = ""
    focus: str = ""
    source: str = ""


@dataclass
class InterviewQAResult:
    """
    面试题 + 参考答案检索结果。
    """
    mode: str  # interview_qa_hit / interview_qa_miss / reject
    notice: str
    items: list[dict[str, Any]]
    retrieved_count: int
    retrieved_context: str
    meta: dict[str, Any]


class RagService:
    def __init__(self) -> None:
        self.vector_store = VectorStoreService()

        # 根据配置选择检索器：混合检索（BM25+Vector+Reranker）或纯向量检索
        hybrid_conf = chroma_conf.get("hybrid_search", {})
        rerank_conf = chroma_conf.get("reranking", {})
        use_hybrid = hybrid_conf.get("enabled", False)

        if use_hybrid:
            if rerank_conf.get("enabled", False):
                self.retriever = self.vector_store.get_hybrid_retriever_with_reranker()
                logger.info("[RAG] 使用混合检索 + Reranking 精排")
            else:
                self.retriever = self.vector_store.get_hybrid_retriever()
                logger.info("[RAG] 使用混合检索（BM25 + Vector）")
        else:
            self.retriever = self.vector_store.get_retriever()
            logger.info("[RAG] 使用纯向量检索")

        # 模型和输出解析器
        self.model = chat_model
        self.output_parser = StrOutputParser()

        # RAG 问答 prompt
        self.rag_prompt_text = load_rag_query_prompt()
        self.rag_prompt = PromptTemplate.from_template(self.rag_prompt_text)

        self.rag_chain = self.rag_prompt | self.model | self.output_parser

        # 查询改写 chain（LLM 将口语化问题改写为更适合检索的形式）
        self.query_rewrite_enabled = chroma_conf.get("query_rewriting", {}).get("enabled", False)
        if self.query_rewrite_enabled:
            rewrite_prompt_text = load_query_rewrite_prompt()
            rewrite_prompt = PromptTemplate.from_template(rewrite_prompt_text)
            self.query_rewrite_chain = rewrite_prompt | self.model | StrOutputParser()
            logger.info("[RAG] 查询改写已启用")
        else:
            self.query_rewrite_chain = None

    def _rewrite_query(self, query: str) -> str:
        """
        用 LLM 将用户口语化问题改写为更适合检索的形式。
        改写失败时静默降级，返回原始 query。
        """
        if not self.query_rewrite_enabled or not self.query_rewrite_chain:
            return query
        try:
            rewritten = self.query_rewrite_chain.invoke({"query": query}).strip()
            # 校验：改写结果过短或为空则降级
            if not rewritten or len(rewritten) < 3:
                logger.info(f"[RAG] query_rewrite 结果过短，降级为原始 query")
                return query
            logger.info(f"[RAG] query_rewrite 原始={query} 改写={rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"[RAG] query_rewrite 失败，降级为原始 query: {e}")
            return query

    def retrieve_docs(self, query: str) -> list[Document]:
        """检索知识库文档。"""
        return self.retriever.invoke(query)

    @staticmethod
    def _format_context(docs: list[Document], max_docs: int = 4) -> str:
        """把检索文档整理为可喂给模型的上下文字符串。"""
        if not docs:
            return ""

        blocks: list[str] = []
        for idx, doc in enumerate(docs[:max_docs], 1):
            blocks.append(
                f"【参考资料{idx}】\n"
                f"内容: {doc.page_content}\n"
                f"元数据: {doc.metadata}\n"
            )
        return "\n".join(blocks).strip()

    @staticmethod
    def _is_retrieval_insufficient(
        docs: list[Document],
        min_docs: int = 1,
        min_chars: int = 100,
    ) -> bool:
        """
        粗粒度判断检索是否不足：
        - 命中文档数少于阈值
        - 或命中内容太短（信息量不足）
        """
        if len(docs) < min_docs:
            return True

        total_chars = sum(len(d.page_content or "") for d in docs)
        return total_chars < min_chars

    @staticmethod
    def _extract_query_keywords(query: str) -> list[str]:
        """
        从 query 提取更“领域化”的关键词，用于粗粒度相关性判断。
        """
        stop_words = {
            "面试", "问题", "常见", "如何", "哪些", "什么", "相关", "准备", "建议",
            "岗位", "工程师", "技术", "知识", "请问", "请", "一下", "介绍",
            "技能", "能力", "要求", "需要", "ai", "产品", "经理", "大", "模型",
            "参考答案", "答案", "题目", "面试题",
        }

        en_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", query)
        zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,8}", query)

        raw_tokens = [t.lower().strip() for t in (en_tokens + zh_tokens)]
        keywords: list[str] = []
        for t in raw_tokens:
            if not t or t in stop_words:
                continue
            if t not in keywords:
                keywords.append(t)

        # 额外处理一些常见组合词
        joined_query = query.lower().replace(" ", "")
        if "大模型工程师" in joined_query and "大模型工程师" not in keywords:
            keywords.append("大模型工程师")
        if "rag" in joined_query and "rag" not in keywords:
            keywords.append("rag")
        if "llm" in joined_query and "llm" not in keywords:
            keywords.append("llm")
        if "agent" in joined_query and "agent" not in keywords:
            keywords.append("agent")

        return keywords

    def _is_retrieval_relevant(self, query: str, docs: list[Document]) -> bool:
        """
        判断检索是否“相关”：
        - 只要命中 1 个 query 关键词即可
        - 不再使用岗位词硬校验，避免误伤真正相关的内容
        """
        if not docs:
            return False

        keywords = self._extract_query_keywords(query)
        if not keywords:
            return True

        haystack = "\n".join((d.page_content or "") for d in docs).lower()
        hit_count = sum(1 for k in keywords if k in haystack)

        logger.info(
            f"[RAG] relevance_check query={query} keywords={keywords} hit_count={hit_count}"
        )

        return hit_count >= 1

    @staticmethod
    def _is_out_of_scope_query(query: str) -> bool:
        """
        判断是否不应进入业务处理流程（无关问题或非目标场景）。
        返回 True 表示统一走“非目标场景处理”分支，不进入 RAG 流程。
        """
        q = (query or "").lower().strip()
        if not q:
            return True

        in_scope_keywords = {
            "面试", "简历", "jd", "岗位", "求职", "评估", "追问", "题目",
            "算法", "开发", "工程", "运维", "产品", "经理", "大模型", "llm",
            "rag", "agent", "prompt", "机器学习", "深度学习", "nlp", "python",
            "java", "后端", "前端", "数据库", "系统设计", "八股", "技术",
            "参考答案", "答案", "面试题",
        }

        out_scope_keywords = {
            "天气", "下雨", "温度", "穿什么", "穿搭", "衣服", "吃什么", "美食",
            "电影", "电视剧", "八卦", "旅游", "星座", "情感", "睡眠", "健身",
            "减肥", "笑话", "聊天", "闲聊", "周末", "购物", "外卖",
        }

        if any(k in q for k in in_scope_keywords):
            return False

        if any(k in q for k in out_scope_keywords):
            return True

        return False

    def _answer_with_retrieval(self, query: str, request_mode: str) -> RagAnswerResult:
        """
        普通问答统一执行逻辑：
        1) 查询改写（可选）
        2) 混合检索 + 精排
        3) 检索足够 -> rag_hit
        4) 检索不足 -> fallback
        """
        if self._is_out_of_scope_query(query):
            reject_answer = (
                "我主要用于面试准备与专业知识问答（如简历/JD分析、面试题、"
                "技术知识与答题建议）。\n"
                "我当前无法回答你的问题。\n"
                "如果你是想问专业知识或面试相关的问题，请描述得更清楚些。"
            )
            logger.info(
                f"[RAG] mode=reject request_mode={request_mode} query={query}"
            )
            return RagAnswerResult(
                mode="reject",
                answer=reject_answer,
                notice="当前问题不属于面试或专业知识场景。",
                retrieved_count=0,
                retrieved_context="",
                meta={"reason": ["out_of_scope"], "request_mode": request_mode},
            )

        # 查询改写：将口语化问题转为更适合检索的形式
        rewritten_query = self._rewrite_query(query)

        docs = self.retrieve_docs(rewritten_query)
        context = self._format_context(docs)
        retrieval_insufficient = self._is_retrieval_insufficient(docs)
        retrieval_relevant = self._is_retrieval_relevant(query, docs)

        logger.info(
            f"[RAG] retrieval_stats request_mode={request_mode} "
            f"query={query} docs={len(docs)} insufficient={retrieval_insufficient} "
            f"relevant={retrieval_relevant}"
        )

        if retrieval_insufficient or not retrieval_relevant:
            notice = "当前知识库中未检索到足够相关内容，以下回答基于通用面试经验。"
            answer = self.rag_chain.invoke(
                {
                    "answer_mode": "fallback",
                    "query": query,
                    "context": "",
                }
            )

            reason = []
            if retrieval_insufficient:
                reason.append("retrieval_insufficient")
            if not retrieval_relevant:
                reason.append("retrieval_irrelevant")

            logger.info(
                f"[RAG] mode=fallback request_mode={request_mode} "
                f"query={query} retrieved_count={len(docs)}"
            )
            return RagAnswerResult(
                mode="fallback",
                answer=answer,
                notice=notice,
                retrieved_count=len(docs),
                retrieved_context=context,
                meta={
                    "reason": reason,
                    "request_mode": request_mode,
                },
            )

        answer = self.rag_chain.invoke(
            {
                "answer_mode": "rag_hit",
                "query": query,
                "context": context,
            }
        )

        logger.info(
            f"[RAG] mode=rag_hit request_mode={request_mode} "
            f"query={query} retrieved_count={len(docs)}"
        )
        return RagAnswerResult(
            mode="rag_hit",
            answer=answer,
            notice="",
            retrieved_count=len(docs),
            retrieved_context=context,
            meta={"request_mode": request_mode},
        )

    @staticmethod
    def _extract_interview_items_from_docs(docs: list[Document], query: str) -> list[dict[str, Any]]:
        """
        从检索到的文档中提取“面试题 + 参考答案”条目。
        这里做一个尽量稳健的过渡实现：
        - 优先从 metadata 中读取结构化字段
        - 否则用文档内容做兜底提取
        """
        items: list[dict[str, Any]] = []

        for doc in docs:
            meta = doc.metadata or {}

            question = str(
                meta.get("question")
                or meta.get("title")
                or meta.get("q")
                or ""
            ).strip()

            reference_answer = str(
                meta.get("reference_answer")
                or meta.get("answer")
                or meta.get("reference")
                or ""
            ).strip()

            qtype = str(meta.get("qtype") or meta.get("type") or "").strip()
            focus = str(meta.get("focus") or meta.get("exam_focus") or "").strip()

            # 如果 metadata 没有，就尝试从正文中抽一部分
            content = (doc.page_content or "").strip()
            if not question and content:
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                if lines:
                    question = lines[0]
                if not reference_answer and len(lines) > 1:
                    reference_answer = "\n".join(lines[1:]).strip()

            if not question and not reference_answer:
                continue

            items.append(
                {
                    "question": question or query,
                    "reference_answer": reference_answer or content,
                    "qtype": qtype,
                    "focus": focus,
                    "source": meta.get("source", ""),
                }
            )

        return items

    def search_interview_qa(self, query: str, use_file_mode: bool = True) -> InterviewQAResult:
        """
        检索面试题 + 参考答案。

        策略：
        1) 先检索知识库
        2) 优先使用知识库中已有的题目/参考答案
        3) 如果没有结构化题库结果，再用普通 RAG 的检索结果做兜底
        """
        q = (query or "").strip()
        if not q:
            return InterviewQAResult(
                mode="reject",
                notice="请输入有效的问题。",
                items=[],
                retrieved_count=0,
                retrieved_context="",
                meta={"reason": "empty_query"},
            )

        if self._is_out_of_scope_query(q):
            return InterviewQAResult(
                mode="reject",
                notice="当前问题不属于面试或专业知识场景。",
                items=[],
                retrieved_count=0,
                retrieved_context="",
                meta={"reason": "out_of_scope"},
            )

        try:
            rewritten_query = self._rewrite_query(q)
            docs = self.retrieve_docs(rewritten_query)
            context = self._format_context(docs)

            items = self._extract_interview_items_from_docs(docs, q)

            # 如果没有结构化条目，但有检索内容，则先尝试让模型整理成题库样式
            if not items and docs:
                answer_result = self._answer_with_retrieval(query=q, request_mode="interview_qa")
                if answer_result.mode in {"rag_hit", "fallback"}:
                    items = [
                        {
                            "question": q,
                            "reference_answer": answer_result.answer,
                            "qtype": "",
                            "focus": "",
                            "source": "rag_answer_fallback",
                        }
                    ]

            # 如果还是没有，返回空结果，让上层走大模型兜底
            if not items:
                return InterviewQAResult(
                    mode="interview_qa_miss",
                    notice="知识库中未检索到明确的面试题/参考答案。",
                    items=[],
                    retrieved_count=len(docs),
                    retrieved_context=context,
                    meta={"reason": ["no_interview_qa_items"]},
                )

            return InterviewQAResult(
                mode="interview_qa_hit",
                notice="已从知识库检索到面试题/参考答案相关内容。",
                items=items,
                retrieved_count=len(docs),
                retrieved_context=context,
                meta={"reason": [], "query": q, "use_file_mode": use_file_mode},
            )

        except Exception as e:
            logger.error(f"[RAG] search_interview_qa failed: {str(e)}", exc_info=True)
            return InterviewQAResult(
                mode="reject",
                notice="RAG 服务调用失败，请稍后重试。",
                items=[],
                retrieved_count=0,
                retrieved_context="",
                meta={"reason": "exception", "error": str(e)},
            )

    def answer_with_rag(self, query: str) -> RagAnswerResult:
        return self._answer_with_retrieval(query=query, request_mode="with_file")

    def answer_general_question(self, query: str) -> RagAnswerResult:
        result = self._answer_with_retrieval(query=query, request_mode="no_file")

        if result.mode == "rag_hit":
            result.notice = "当前为无文件问答模式（已结合固定知识库检索结果）。"
        elif result.mode == "reject":
            result.notice = "当前输入未触发面试助手的业务处理流程。"
        else:
            result.notice = (
                "当前为无文件问答模式，且知识库未检索到足够相关内容，"
                "以下回答基于通用面试经验。"
            )
        return result

    def rag_summarize(self, query: str) -> str:
        return self.answer_with_rag(query).answer


if __name__ == "__main__":
    service = RagService()

    test_query_1 = "请给我一些大模型工程师面试准备建议"
    result_1 = service.answer_with_rag(test_query_1)
    print("=== [WITH FILE MODE TEST] ===")
    print("mode:", result_1.mode)
    print("notice:", result_1.notice)
    print("retrieved_count:", result_1.retrieved_count)
    print("answer:\n", result_1.answer)

    test_query_2 = "简单介绍一下RAG技术"
    result_2 = service.answer_general_question(test_query_2)
    print("\n=== [NO FILE MODE TEST] ===")
    print("mode:", result_2.mode)
    print("notice:", result_2.notice)
    print("retrieved_count:", result_2.retrieved_count)
    print("answer:\n", result_2.answer)

    test_query_3 = "给我一些AI Agent面试题和参考答案"
    result_3 = service.search_interview_qa(test_query_3)
    print("\n=== [INTERVIEW QA TEST] ===")
    print("mode:", result_3.mode)
    print("notice:", result_3.notice)
    print("retrieved_count:", result_3.retrieved_count)
    print("items:", len(result_3.items))
    for idx, item in enumerate(result_3.items, 1):
        print(f"{idx}. Q: {item.get('question', '')}")
        print(f"   A: {item.get('reference_answer', '')}")