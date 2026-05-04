"""
RAG 检索回答工具（面试助手版）

作用：
- 接收用户问题
- 调用 rag 目录下的 RAG 实现
- 返回结构化结果，供 Agent / App 层使用
- 支持普通知识问答
- 支持面试题/参考答案检索

说明：
- 复用 ai_interview_assistant.rag.rag_service.RagService
- 主检索回答逻辑已经在 rag_service.py 中实现
- 该工具仅作为 agent/tools 层的统一封装入口
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from ai_interview_assistant.rag.rag_service import RagService, RagAnswerResult
from ai_interview_assistant.utils.logger_handler import logger


@dataclass
class RagToolResult:
    """RAG 工具返回结构。"""
    mode: str
    answer: str
    notice: str
    retrieved_count: int
    retrieved_context: str
    meta: dict[str, Any]


@dataclass
class InterviewQAItem:
    """面试题检索结果项。"""
    question: str
    reference_answer: str
    qtype: str = ""
    focus: str = ""
    source: str = ""


@dataclass
class InterviewQAResult:
    """面试题+参考答案检索结果。"""
    mode: str
    notice: str
    items: list[dict[str, Any]]
    retrieved_count: int
    retrieved_context: str
    meta: dict[str, Any]


class RagTool:
    """
    RAG 工具封装。

    对外提供统一的问答接口：
    - 有文件/上下文场景：answer_with_rag
    - 无文件场景：answer_general_question
    - 面试题/参考答案检索：search_interview_qa
    """

    def __init__(self) -> None:
        self.service = RagService()

    @staticmethod
    def _convert_result(result: RagAnswerResult) -> dict[str, Any]:
        """
        将 RagService 的返回结果转为工具层统一结构。
        """
        tool_result = RagToolResult(
            mode=result.mode,
            answer=result.answer,
            notice=result.notice,
            retrieved_count=result.retrieved_count,
            retrieved_context=result.retrieved_context,
            meta=result.meta,
        )
        return asdict(tool_result)

    @staticmethod
    def _build_empty_result(reason: str, notice: str) -> dict[str, Any]:
        return asdict(
            RagToolResult(
                mode="reject",
                answer="",
                notice=notice,
                retrieved_count=0,
                retrieved_context="",
                meta={"reason": reason},
            )
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    def answer(
        self,
        query: str,
        use_file_mode: bool = True,
    ) -> dict[str, Any]:
        """
        统一问答入口。

        Args:
            query: 用户问题
            use_file_mode: True 表示按“上传简历/JD后的业务模式”处理；
                           False 表示按“无文件问答模式”处理

        Returns:
            dict: 结构化回答结果
        """
        query = self._normalize_text(query)
        if not query:
            return self._build_empty_result("empty_query", "请输入有效的问题。")

        try:
            if use_file_mode:
                result = self.service.answer_with_rag(query)
            else:
                result = self.service.answer_general_question(query)

            logger.info(
                f"[RagTool] answer done | mode={result.mode} "
                f"retrieved_count={result.retrieved_count}"
            )
            return self._convert_result(result)

        except Exception as e:
            logger.error(f"[RagTool] answer failed: {str(e)}", exc_info=True)
            return asdict(
                RagToolResult(
                    mode="reject",
                    answer="",
                    notice="RAG 服务调用失败，请稍后重试。",
                    retrieved_count=0,
                    retrieved_context="",
                    meta={"reason": "exception", "error": str(e)},
                )
            )

    def search_interview_qa(
        self,
        query: str,
        use_file_mode: bool = True,
    ) -> dict[str, Any]:
        """
        检索面试题 + 参考答案。

        说明：
        - 这里优先尝试从知识库中检索“题目、答案、考察点”相关内容
        - 仅当底层返回明确的结构化题库条目（meta.items）时，才判定为命中
        - 普通 RAG 的 answer / retrieved_context 不再被包装成题目
        - 这样可以避免把“普通检索结果”误当成“面试题库命中”

        Returns:
            dict:
                {
                    mode,
                    notice,
                    items,
                    retrieved_count,
                    retrieved_context,
                    meta
                }
        """
        query = self._normalize_text(query)
        if not query:
            return asdict(
                InterviewQAResult(
                    mode="reject",
                    notice="请输入有效的问题。",
                    items=[],
                    retrieved_count=0,
                    retrieved_context="",
                    meta={"reason": "empty_query"},
                )
            )

        try:
            rag_result = self.service.answer_with_rag(query) if use_file_mode else self.service.answer_general_question(query)

            items: list[dict[str, Any]] = []
            base_mode = str(getattr(rag_result, "mode", "")).strip()
            base_notice = str(getattr(rag_result, "notice", "")).strip()

            meta = rag_result.meta or {}
            raw_items = meta.get("items", []) if isinstance(meta, dict) else []
            if isinstance(raw_items, list) and raw_items:
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    q = self._normalize_text(item.get("question", ""))
                    a = self._normalize_text(item.get("reference_answer", ""))
                    if not q and not a:
                        continue
                    items.append(
                        asdict(
                            InterviewQAItem(
                                question=q,
                                reference_answer=a,
                                qtype=self._normalize_text(item.get("qtype", "")),
                                focus=self._normalize_text(item.get("focus", "")),
                                source=self._normalize_text(item.get("source", "")),
                            )
                        )
                    )

            hit = bool(items)

            if hit:
                notice = "已从知识库检索到面试题/参考答案相关内容。"
            else:
                notice = "知识库中未检索到明确的面试题/参考答案，已返回普通检索结果。"

            result = InterviewQAResult(
                mode="interview_qa_hit" if hit else "interview_qa_miss",
                notice=notice,
                items=items,
                retrieved_count=rag_result.retrieved_count,
                retrieved_context=rag_result.retrieved_context,
                meta={
                    "source": "rag",
                    "base_mode": base_mode,
                    "base_notice": base_notice,
                    "result_kind": "structured_kb_hit" if hit else "rag_fallback",
                    "has_structured_items": hit,
                },
            )

            logger.info(
                f"[RagTool] search_interview_qa done | "
                f"mode={result.mode} items={len(result.items)}"
            )
            return asdict(result)

        except Exception as e:
            logger.error(f"[RagTool] search_interview_qa failed: {str(e)}", exc_info=True)
            return asdict(
                InterviewQAResult(
                    mode="reject",
                    notice="RAG 服务调用失败，请稍后重试。",
                    items=[],
                    retrieved_count=0,
                    retrieved_context="",
                    meta={"reason": "exception", "error": str(e)},
                )
            )

    def rag_summarize(self, query: str, use_file_mode: bool = True) -> str:
        """
        兼容旧风格接口：直接返回回答文本。
        """
        return self.answer(query, use_file_mode=use_file_mode).get("answer", "")


def answer_with_rag(query: str, use_file_mode: bool = True) -> dict[str, Any]:
    """
    便捷函数：直接调用 RAG 工具。
    """
    tool = RagTool()
    return tool.answer(query=query, use_file_mode=use_file_mode)


def search_interview_qa(query: str, use_file_mode: bool = True) -> dict[str, Any]:
    """
    便捷函数：直接检索面试题 + 参考答案。
    """
    tool = RagTool()
    return tool.search_interview_qa(query=query, use_file_mode=use_file_mode)


if __name__ == "__main__":
    demo_query_1 = "请给我一些大模型工程师面试准备建议"
    demo_query_2 = "简单介绍一下RAG技术"
    demo_query_3 = "给我一些AI Agent面试题和参考答案"

    tool = RagTool()

    def print_simple_result(title: str, result: dict[str, Any]) -> None:
        print(f"\n=== {title} ===")
        print("结果模式:", result.get("mode"))
        print("提示:", result.get("notice"))
        if "items" in result:
            print("检索条数:", len(result.get("items", [])))
            for idx, item in enumerate(result.get("items", []), 1):
                print(f"{idx}. 问题: {item.get('question', '')}")
                print(f"   参考答案: {item.get('reference_answer', '')}")
        else:
            print("模型回复结果:")
            print(result.get("answer", "").strip() or "[empty]")
        print("-" * 60)

    result_1 = tool.answer(demo_query_1, use_file_mode=True)
    print_simple_result("WITH FILE MODE", result_1)

    result_2 = tool.answer(demo_query_2, use_file_mode=False)
    print_simple_result("NO FILE MODE", result_2)

    result_3 = tool.search_interview_qa(demo_query_3, use_file_mode=True)
    print_simple_result("INTERVIEW QA", result_3)