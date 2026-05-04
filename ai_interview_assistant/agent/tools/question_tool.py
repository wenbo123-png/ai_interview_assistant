"""
面试题生成工具（最终版）

作用：
- 输入简历信息、JD 信息、RAG 上下文
- 生成结构化面试题列表（3~10题）
- 支持优先检索题库文本中的面试题/参考答案
- 检索不到时再调用大模型生成
- 为每道题补充参考答案示例，便于前端/Agent 展示

说明：
当前版本改为从 prompts/question_generation_prompt.txt 加载提示词，
避免将 prompt 直接写死在代码中。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_interview_assistant.agent.tools.rag_tool import RagTool
from ai_interview_assistant.model.factory import chat_model
from ai_interview_assistant.utils.json_utils import (
    parse_llm_json_response,
    ensure_dict,
    ensure_list,
)
from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.prompt_loader import load_question_generation_prompt, load_mock_interview_prompt


@dataclass
class InterviewQuestion:
    """单道面试题结构。"""
    question: str
    qtype: str
    focus: str
    reference_points: list[str]
    approach: str = ""  # 答题思路
    reference_answer: str = ""
    source: str = ""  # kb / llm / fallback


@dataclass
class QuestionGenerationResult:
    """出题结果结构。"""
    mode: str  # rag_based / fallback / kb_hit
    notice: str
    questions: list[dict[str, Any]]
    meta: dict[str, Any]


class QuestionTool:
    """
    面试题生成工具。
    """

    def __init__(self) -> None:
        self.model = chat_model
        self.parser = StrOutputParser()
        self.rag_tool = RagTool()

        # 从 prompts/question_generation_prompt.txt 加载主提示词
        self.prompt_text = load_question_generation_prompt()
        self.prompt = PromptTemplate.from_template(self.prompt_text)

        # 同一份 prompt 文件同时支持 RAG 命中和 fallback 场景
        self.chain = self.prompt | self.model | self.parser

        # 模拟面试专用轻量 prompt（只要题目，不要参考答案）
        self.mock_prompt_text = load_mock_interview_prompt()
        self.mock_prompt = PromptTemplate.from_template(self.mock_prompt_text)
        self.mock_chain = self.mock_prompt | self.model | self.parser

    @staticmethod
    def _normalize_qtype(item_dict: dict[str, Any]) -> str:
        """
        统一题型字段名。
        兼容：
        - qtype
        - type
        """
        qtype = str(item_dict.get("qtype", "") or item_dict.get("type", "")).strip()
        return qtype or "综合能力类"

    @staticmethod
    def _normalize_focus(item_dict: dict[str, Any]) -> str:
        """
        统一考察点字段。
        """
        focus = str(item_dict.get("focus", "")).strip()
        return focus or "综合能力评估"

    @staticmethod
    def _normalize_reference_points(item_dict: dict[str, Any]) -> list[str]:
        """
        规范化参考要点。
        """
        points = ensure_list(item_dict.get("reference_points", []), default=[])
        ref_points = [str(p).strip() for p in points if str(p).strip()]

        return ref_points

    @staticmethod
    def _build_reference_answer(question: str, qtype: str, focus: str, ref_points: list[str]) -> str:
        """
        基于参考要点构造一个简洁的参考答案示例。
        目标是给前端和 react_agent 一个可直接展示的答案，而不是长篇分析。
        """
        q = question.strip()
        if not q:
            return ""

        if qtype in {"自我介绍类", "项目经历类", "岗位匹配类", "行为面试类", "技术基础类", "问题排查类"}:
            answer_lines = [
                f"回答时可围绕“{focus}”展开。",
                "建议先给出结论，再补充具体经历或方法，最后回到岗位匹配点。",
            ]
        else:
            answer_lines = [
                "建议先说明核心思路，再补充一个具体例子。",
            ]

        if ref_points:
            answer_lines.append("可覆盖以下要点：")
            for idx, point in enumerate(ref_points[:5], 1):
                answer_lines.append(f"{idx}. {point}")

        return "\n".join(answer_lines).strip()

    @staticmethod
    def _normalize_questions(raw_questions: list[Any], source: str = "llm") -> list[dict[str, Any]]:
        """
        规范化问题列表，避免字段缺失。
        """
        normalized: list[dict[str, Any]] = []

        for item in raw_questions:
            item_dict = ensure_dict(item, default={})
            q = str(item_dict.get("question", "")).strip()
            if not q:
                continue

            qtype = QuestionTool._normalize_qtype(item_dict)
            focus = QuestionTool._normalize_focus(item_dict)
            ref_points = QuestionTool._normalize_reference_points(item_dict)

            approach = str(item_dict.get("approach", "") or "").strip()
            reference_answer = str(item_dict.get("reference_answer", "") or "").strip()
            if not reference_answer:
                reference_answer = QuestionTool._build_reference_answer(
                    question=q,
                    qtype=qtype,
                    focus=focus,
                    ref_points=ref_points,
                )

            normalized.append(
                asdict(
                    InterviewQuestion(
                        question=q,
                        qtype=qtype,
                        focus=focus,
                        reference_points=ref_points,
                        approach=approach,
                        reference_answer=reference_answer,
                        source=str(item_dict.get("source", "") or source).strip() or source,
                    )
                )
            )

        return normalized

    @staticmethod
    def _normalize_text(text: str) -> str:
        """压缩空白并去除首尾空格。"""
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _build_fallback_notice() -> str:
        return "知识库相关内容不足，已基于通用面试经验生成题目。"

    @staticmethod
    def _looks_like_question_text(text: str) -> bool:
        """
        判断文本是否像一道面试题。

        目的：
        - 过滤掉摘要、参考资料、元数据、答案块等非题目文本
        - 避免它们被误包装成“题库命中”中的题目
        """
        q = QuestionTool._normalize_text(text)
        if not q:
            return False

        # 明显不是题目的内容直接过滤
        blocked_markers = (
            "参考资料", "元数据", "内容:", "内容：", "摘要", "职责:", "职责：",
            "要求:", "要求：", "说明:", "说明：", "可覆盖以下要点", "建议先", "回答时可",
        )
        if any(marker in q for marker in blocked_markers):
            return False

        # 题目通常较短，且具备一定的提问特征
        if len(q) > 160:
            return False

        question_markers = (
            "什么是", "如何", "为什么", "请", "介绍", "解释", "区别", "谈谈",
            "举例", "描述", "分析", "你会如何", "你如何", "怎么", "怎样",
        )
        return any(marker in q for marker in question_markers) or q.endswith(("?", "？"))

    def _parse_questions(self, text: str) -> list[dict[str, Any]]:
        """
        从 LLM 输出中解析 questions。
        """
        parsed = parse_llm_json_response(text, default={})
        data = ensure_dict(parsed, default={})
        raw_questions = ensure_list(data.get("questions", []), default=[])
        return self._normalize_questions(raw_questions, source="llm")

    @staticmethod
    def _question_key(item: dict[str, Any]) -> str:
        """用于去重的题目键。"""
        return QuestionTool._normalize_text(str(ensure_dict(item, default={}).get("question", ""))).lower()

    @staticmethod
    def _merge_unique_questions(
        base_questions: list[dict[str, Any]],
        new_questions: list[dict[str, Any]],
        desired_count: int,
    ) -> list[dict[str, Any]]:
        """按题目文本去重并保留顺序，限制到目标数量。"""
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for question_list in (base_questions, new_questions):
            for item in question_list:
                key = QuestionTool._question_key(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(item)
                if len(merged) >= desired_count:
                    return merged

        return merged

    @staticmethod
    def _normalize_items_from_kb_items(kb_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        把 RagService.search_interview_qa 返回的普通文本题库结果，转换成统一结构。
        """
        normalized: list[dict[str, Any]] = []

        for item in kb_items:
            item_dict = ensure_dict(item, default={})
            question = str(item_dict.get("question", "")).strip()
            reference_answer = str(item_dict.get("reference_answer", "")).strip()
            qtype = str(item_dict.get("qtype", "")).strip() or "题库题"
            focus = str(item_dict.get("focus", "")).strip() or "题库参考"
            source = str(item_dict.get("source", "")).strip() or "kb"

            # 只保留真正像“面试题”的文本；
            # 纯答案、纯摘要、纯元数据不应被当成题目展示
            if not question:
                continue

            if not QuestionTool._looks_like_question_text(question):
                continue

            if not reference_answer:
                reference_answer = "请参考知识库原文或结合题目上下文进行回答。"

            normalized.append(
                asdict(
                    InterviewQuestion(
                        question=question or "题库检索结果",
                        qtype=qtype,
                        focus=focus,
                        reference_points=[focus] if focus else [],
                        approach="",
                        reference_answer=reference_answer,
                        source=source,
                    )
                )
            )

        return normalized

    def _get_questions_from_kb(
        self,
        resume_context: str,
        jd_context: str,
        rag_context: str,
        use_rag: bool,
    ) -> tuple[list[dict[str, Any]], str, str]:
        """
        先检索题库文本中的面试题/参考答案。
        命中则直接返回。
        """
        query_parts = [resume_context.strip(), jd_context.strip(), rag_context.strip()]
        query = " ".join([p for p in query_parts if p])

        if not query:
            query = "面试题 参考答案"

        kb_result = self.rag_tool.search_interview_qa(query=query, use_file_mode=use_rag)
        mode = str(kb_result.get("mode", "")).strip()

        # 先尝试从 meta.items / items 中获取结构化题目
        if mode == "interview_qa_hit":
            items = kb_result.get("items", []) or []
            normalized_items = self._normalize_items_from_kb_items(items)
            if normalized_items:
                notice = kb_result.get("notice", "") or "已从知识库检索到面试题和参考答案。"
                retrieved_context = str(kb_result.get("retrieved_context", "") or "").strip()
                return normalized_items, notice, retrieved_context

        # 如果没有结构化题目，但底层返回了检索上下文（纯文本），
        # 把这个 retrieved_context 作为 LLM 的参考上下文返回，避免直接走 fallback
        retrieved_context = str(kb_result.get("retrieved_context", "") or "").strip()
        if retrieved_context:
            notice = kb_result.get("notice", "") or "已从知识库检索到参考上下文，已用于题目生成。"
            return [], notice, retrieved_context

        return [], "", ""

    def _invoke_mode(
        self,
        resume_context: str,
        jd_context: str,
        rag_context: str,
        extra_notice: str = "",
        use_rag: bool = True,
    ) -> QuestionGenerationResult:
        """
        统一调用 prompt + LLM 生成题目。

        通过 extra_notice 控制当前是否为 fallback 场景。
        """
        try:
            text = self.chain.invoke(
                {
                    "resume_context": resume_context,
                    "jd_context": jd_context,
                    "rag_context": rag_context,
                    "extra_notice": extra_notice,
                }
            )
            questions = self._parse_questions(text)

            if not questions:
                logger.warning("[QuestionTool] LLM 输出解析为空，返回空题目列表。")
                return QuestionGenerationResult(
                    mode="fallback",
                    notice=self._build_fallback_notice(),
                    questions=[],
                    meta={"source": "llm_empty_output"},
                )

            mode = "rag_based" if (rag_context or "").strip() else "fallback"
            notice = (
                "已结合固定知识库检索结果生成面试题。"
                if mode == "rag_based"
                else self._build_fallback_notice()
            )

            return QuestionGenerationResult(
                mode=mode,
                notice=notice,
                questions=questions,
                meta={"source": "rag" if mode == "rag_based" else "fallback"},
            )
        except Exception as e:
            logger.error(f"[QuestionTool] 调用出题链失败: {str(e)}", exc_info=True)
            return QuestionGenerationResult(
                mode="fallback",
                notice=self._build_fallback_notice(),
                questions=[],
                meta={"source": "error", "error": str(e)},
            )

    def generate_questions(
        self,
        resume_context: str,
        jd_context: str,
        rag_context: str = "",
        use_rag: bool = True,
        desired_count: int = 6,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        """
        统一出题入口。

        优先级：
        1) 先检索题库文本中的面试题/参考答案
        2) 如果检索不到，再让大模型生成
        3) 如果模型没给 reference_answer，再由工具层兜底生成

        Returns:
            dict: 结构化出题结果
        """
        resume_context = resume_context or ""
        jd_context = jd_context or ""
        rag_context = rag_context or ""
        desired_count = max(1, int(desired_count or 6))
        max_attempts = max(1, int(max_attempts or 1))

        def build_missing_notice(base_notice: str) -> str:
            prefix = str(base_notice or "").strip()
            extra = f"请确保输出至少 {desired_count} 道不重复的面试题。"
            return f"{prefix} {extra}".strip() if prefix else extra

        def enrich_and_limit(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return self._merge_unique_questions([], questions, desired_count)

        combined_rag_context = "\n".join(
            [p for p in (rag_context.strip(),) if p]
        )

        # 先查题库（结构化题目或纯文本检索上下文）
        kb_questions, kb_notice, kb_retrieved_context = self._get_questions_from_kb(
            resume_context=resume_context,
            jd_context=jd_context,
            rag_context=rag_context,
            use_rag=use_rag,
        )

        # 如果直接从 KB 得到结构化题目，优先返回
        if kb_questions:
            questions = enrich_and_limit(kb_questions)

            if len(questions) < desired_count:
                supplement_context = "\n".join(
                    [p for p in (combined_rag_context.strip(), kb_retrieved_context.strip()) if p]
                )
                for attempt in range(max_attempts):
                    supplement = self._invoke_mode(
                        resume_context=resume_context,
                        jd_context=jd_context,
                        rag_context=supplement_context,
                        extra_notice=build_missing_notice(
                            f"已有 {len(questions)} 道题，请补足到 {desired_count} 道并避免重复。"
                        ),
                        use_rag=use_rag,
                    )
                    questions = self._merge_unique_questions(questions, supplement.questions, desired_count)
                    if len(questions) >= desired_count:
                        break

            result = QuestionGenerationResult(
                mode="kb_hit" if len(questions) >= desired_count else "rag_based",
                notice=kb_notice or "已从知识库检索到面试题和参考答案。",
                questions=questions,
                meta={
                    "source": "kb",
                    "question_count": len(questions),
                    "desired_count": desired_count,
                },
            )
            logger.info(
                f"[QuestionTool] generate_questions done | mode={result.mode} "
                f"question_count={len(result.questions)}"
            )
            return asdict(result)

        # 如果没有结构化题目，但检索到纯文本上下文，仍然把该上下文作为 rag_context
        # 喂给 LLM，让模型基于检索文本生成更贴合的题目（视为 rag_based）
        if kb_retrieved_context:
            # 合并外部传入的 rag_context 与 KB 检索到的上下文
            merged_rag_context = "\n".join([p for p in (combined_rag_context.strip(), kb_retrieved_context.strip()) if p])
            result = self._invoke_mode(
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context=merged_rag_context,
                extra_notice=build_missing_notice(
                    f"请基于知识库参考内容生成至少 {desired_count} 道不重复的面试题。"
                ),
                use_rag=use_rag,
            )
            # 标注来源为 KB 文本参考
            result.meta = getattr(result, "meta", {}) or {}
            result.meta["source"] = "kb_text_rag"
            result.questions = self._merge_unique_questions([], result.questions, desired_count)
            logger.info(
                f"[QuestionTool] generate_questions done | mode={result.mode} "
                f"question_count={len(result.questions)} (from kb_text_rag)"
            )
            return asdict(result)

        # 题库与检索上下文均无，则根据传入的 rag_context（如果有）或直接 fallback
        if use_rag and rag_context.strip():
            result = self._invoke_mode(
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context=rag_context,
                extra_notice=build_missing_notice(f"请生成至少 {desired_count} 道不重复的面试题。"),
                use_rag=use_rag,
            )
            result.questions = self._merge_unique_questions([], result.questions, desired_count)
        else:
            result = self._invoke_mode(
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context="",
                extra_notice="未提供有效检索上下文，请基于通用经验和候选人信息出题。",
                use_rag=use_rag,
            )
            result.questions = self._merge_unique_questions([], result.questions, desired_count)

        if len(result.questions) < desired_count and use_rag:
            # 最后兜底：如果题目仍不足，则再尝试一次基于简历/JD 的补题，尽量凑足目标数量。
            supplement = self._invoke_mode(
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context="\n".join([p for p in (rag_context.strip(), kb_retrieved_context.strip()) if p]),
                extra_notice=build_missing_notice(
                    f"目前只有 {len(result.questions)} 道题，请继续补充到 {desired_count} 道并避免重复。"
                ),
                use_rag=use_rag,
            )
            result.questions = self._merge_unique_questions(result.questions, supplement.questions, desired_count)
            if len(result.questions) >= desired_count and not str(result.mode).strip():
                result.mode = supplement.mode
            result.meta = getattr(result, "meta", {}) or {}
            result.meta["supplemented"] = True
            result.meta["desired_count"] = desired_count

        logger.info(
            f"[QuestionTool] generate_questions done | mode={result.mode} "
            f"question_count={len(result.questions)}"
        )
        return asdict(result)

    def generate_mock_questions(
        self,
        resume_context: str,
        jd_context: str,
        rag_context: str = "",
        desired_count: int = 6,
    ) -> dict[str, Any]:
        """
        模拟面试专用出题：只要题目，不要参考答案。
        使用轻量 prompt，输出体积小，不易被截断。
        """
        resume_context = resume_context or ""
        jd_context = jd_context or ""
        rag_context = rag_context or ""

        try:
            text = self.mock_chain.invoke(
                {
                    "resume_context": resume_context,
                    "jd_context": jd_context,
                    "rag_context": rag_context,
                }
            )
            parsed = parse_llm_json_response(text, default={})
            data = ensure_dict(parsed, default={})
            raw_questions = ensure_list(data.get("questions", []), default=[])

            # 轻量规范化：只保留 question / type / focus
            questions = []
            for item in raw_questions:
                item_dict = ensure_dict(item, default={})
                q = str(item_dict.get("question", "")).strip()
                if not q:
                    continue
                questions.append({
                    "question": q,
                    "qtype": str(item_dict.get("type", "") or item_dict.get("qtype", "")).strip() or "综合能力类",
                    "focus": str(item_dict.get("focus", "")).strip() or "综合能力评估",
                    "reference_points": [],
                    "approach": "",
                    "reference_answer": "",
                    "source": "llm",
                })

            # 去重
            seen = set()
            unique = []
            for q in questions:
                key = q["question"].lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(q)

            questions = unique[:desired_count]

            if len(questions) < desired_count:
                logger.warning(
                    f"[QuestionTool] mock 只生成了 {len(questions)} 道题，目标 {desired_count} 道"
                )

            result = QuestionGenerationResult(
                mode="mock_interview",
                notice="",
                questions=questions,
                meta={"source": "mock_llm", "question_count": len(questions)},
            )
            logger.info(
                f"[QuestionTool] generate_mock_questions done | count={len(questions)}"
            )
            return asdict(result)

        except Exception as e:
            logger.error(f"[QuestionTool] 模拟面试出题失败: {e}", exc_info=True)
            return asdict(QuestionGenerationResult(
                mode="fallback",
                notice="出题失败，请重试。",
                questions=[],
                meta={"source": "error", "error": str(e)},
            ))


def generate_interview_questions(
    resume_context: str,
    jd_context: str,
    rag_context: str = "",
    use_rag: bool = True,
) -> dict[str, Any]:
    """
    便捷函数：直接生成面试题。
    """
    tool = QuestionTool()
    return tool.generate_questions(
        resume_context=resume_context,
        jd_context=jd_context,
        rag_context=rag_context,
        use_rag=use_rag,
    )


if __name__ == "__main__":
    demo_resume = "技能：Python、RAG、Agent；项目：AI面试准备助手"
    demo_jd = "岗位：AI Agent 应用开发工程师；要求：熟悉RAG、Prompt、工具调用"
    demo_rag = "参考：Agent架构、ReAct、工具调用与检索增强是常见考点。"

    result = generate_interview_questions(
        resume_context=demo_resume,
        jd_context=demo_jd,
        rag_context=demo_rag,
        use_rag=True,
    )
    print("=== 面试题生成结果 ===")
    print(result)