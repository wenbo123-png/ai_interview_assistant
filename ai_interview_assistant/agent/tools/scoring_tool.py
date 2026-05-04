"""
答案评分工具（最终版）

作用：
- 输入原始问题、用户回答、简历信息、JD 信息、RAG 上下文
- 对回答进行结构化评分
- 评分结果仅用于后台记录和最终总结评估
- 面试过程不直接展示评分内容

说明：
当前版本适配 prompts/answer_scoring_prompt.txt 的固定变量名：
- question
- answer
- context
- extra_notice

提示词文件保持不变，代码在内部将 resume_context / jd_context / rag_context 合并为 context。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_interview_assistant.model.factory import chat_model
from ai_interview_assistant.utils.json_utils import (
    parse_llm_json_response,
    ensure_dict,
    ensure_list,
)
from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.prompt_loader import load_answer_scoring_prompt


@dataclass
class AnswerScoreDetail:
    """单项评分详情。"""
    criterion: str
    score: int
    comment: str


@dataclass
class AnswerScoringResult:
    """答案评分结果。"""
    mode: str  # rag_based / fallback / skip / session
    notice: str
    total_score: int
    dimension_scores: dict[str, int]
    comment: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    details: list[dict[str, Any]]
    meta: dict[str, Any]


class ScoringTool:
    """
    答案评分工具。

    输入：
    - original_question: 原始题目
    - user_answer: 用户回答
    - resume_context: 简历结构化信息
    - jd_context: JD结构化信息
    - rag_context: 检索上下文（可为空）

    输出：
    - 结构化评分结果（仅用于后台和最终总结）
    """

    DIMENSION_KEYS: tuple[str, ...] = ("completeness", "logic", "professionalism", "job_fit")
    PROMPT_VERSION: str = "answer_scoring_prompt"
    GENERIC_FEEDBACK_PHRASES: tuple[str, ...] = (
        "回答能够覆盖问题核心。",
        "回答基本覆盖了问题要点。",
        "回答还可以进一步补充关键细节。",
        "需要进一步补充说明。",
        "建议使用 STAR 方法补充回答。",
    )

    def __init__(self) -> None:
        self.model = chat_model
        self.parser = StrOutputParser()

        # 从 prompts/answer_scoring_prompt.txt 加载评分提示词
        self.prompt_text = load_answer_scoring_prompt()
        self.prompt = PromptTemplate.from_template(self.prompt_text)

        self.chain = self.prompt | self.model | self.parser

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            text = ScoringTool._clean_text(item)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    def _normalize_feedback_items(self, value: Any, default_item: str, limit: int = 4) -> list[str]:
        """将反馈条目规范化、去重并过滤常见套话。"""
        items = ensure_list(value, default=[])
        normalized: list[str] = []

        for item in items:
            text = self._clean_text(item)
            if not text:
                continue
            if any(phrase in text for phrase in self.GENERIC_FEEDBACK_PHRASES):
                continue
            if text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break

        if not normalized and default_item:
            normalized = [default_item]

        return normalized

    @staticmethod
    def _clamp_score(value: Any, default: int = 0) -> int:
        """将分数限制在 0~100 之间。"""
        try:
            score = int(float(value))
        except Exception:
            score = default
        return max(0, min(100, score))

    def _normalize_dimension_scores(self, raw: Any) -> dict[str, int]:
        """规范化维度分数。"""
        data = ensure_dict(raw, default={})
        return {
            "completeness": self._clamp_score(data.get("completeness", 0), default=0),
            "logic": self._clamp_score(data.get("logic", 0), default=0),
            "professionalism": self._clamp_score(data.get("professionalism", 0), default=0),
            "job_fit": self._clamp_score(data.get("job_fit", 0), default=0),
        }

    @staticmethod
    def _normalize_details(raw_details: Any) -> list[dict[str, Any]]:
        """
        规范化评分细节列表。
        兼容模型额外输出 details 时的情况；如果没有则返回空列表。
        """
        details: list[dict[str, Any]] = []
        items = ensure_list(raw_details, default=[])

        for item in items:
            item_dict = ensure_dict(item, default={})
            criterion = str(item_dict.get("criterion", "")).strip()
            if not criterion:
                continue

            score = ScoringTool._clamp_score(item_dict.get("score", 0), default=0)
            comment = str(item_dict.get("comment", "")).strip()
            if not comment:
                comment = "需要进一步补充说明。"

            details.append(
                asdict(
                    AnswerScoreDetail(
                        criterion=criterion,
                        score=score,
                        comment=comment,
                    )
                )
            )
        return details

    @staticmethod
    def _merge_context(
        resume_context: str,
        jd_context: str,
        rag_context: str,
    ) -> str:
        """
        将多个上下文合并成 prompt 需要的单一 context 变量。
        """
        parts: list[str] = []

        if resume_context.strip():
            parts.append(f"简历信息：\n{resume_context.strip()}")

        if jd_context.strip():
            parts.append(f"JD信息：\n{jd_context.strip()}")

        if rag_context.strip():
            parts.append(f"参考上下文：\n{rag_context.strip()}")

        return "\n\n".join(parts).strip()

    @staticmethod
    def _extract_transcript_stats(user_answer: str) -> dict[str, Any]:
        """提取整场模拟面试 transcript 的基本统计信息。"""
        text = ScoringTool._clean_text(user_answer)
        question_count = len(re.findall(r"第\s*\d+\s*题：", text))
        answer_count = len(re.findall(r"用户回答：", text))
        followup_count = len(re.findall(r"追问回答\s*\d+：", text)) + len(re.findall(r"追问\s*\d+：", text))
        total_lines = len([line for line in text.splitlines() if line.strip()])
        avg_line_length = round(len(text) / total_lines, 1) if total_lines else 0.0

        tech_keywords = (
            "python", "rag", "agent", "llm", "langchain", "prompt", "数据库", "sql",
            "redis", "mysql", "性能", "优化", "调试", "排查", "系统设计", "工程",
        )
        tech_hits = sum(1 for kw in tech_keywords if kw in text.lower())

        structure_keywords = ("首先", "其次", "然后", "最后", "总结", "结论", "第一", "第二", "第三")
        structure_hits = sum(1 for kw in structure_keywords if kw in text)

        return {
            "question_count": question_count,
            "answer_count": answer_count,
            "followup_count": followup_count,
            "total_lines": total_lines,
            "avg_line_length": avg_line_length,
            "text_length": len(text),
            "tech_hits": tech_hits,
            "structure_hits": structure_hits,
        }

    @staticmethod
    def _looks_like_session_transcript(original_question: str, user_answer: str) -> bool:
        """判断是否是整场模拟面试 transcript。"""
        q = ScoringTool._clean_text(original_question)
        a = ScoringTool._clean_text(user_answer)
        if not a:
            return False
        session_markers = ("整场", "模拟面试", "总体评估", "第1题：", "用户回答：")
        return any(marker in q for marker in session_markers) or len(re.findall(r"第\s*\d+\s*题：", a)) >= 2

    def _build_transcript_heuristic_payload(
        self,
        user_answer: str,
        transcript_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """当 LLM 失败时，构造 transcript-aware 的兜底评分结果。"""
        question_count = int(transcript_stats.get("question_count", 0) or 0)
        answer_count = int(transcript_stats.get("answer_count", 0) or 0)
        followup_count = int(transcript_stats.get("followup_count", 0) or 0)
        text_length = int(transcript_stats.get("text_length", 0) or 0)
        avg_line_length = float(transcript_stats.get("avg_line_length", 0) or 0)
        tech_hits = int(transcript_stats.get("tech_hits", 0) or 0)
        structure_hits = int(transcript_stats.get("structure_hits", 0) or 0)

        completeness = 60 + min(15, question_count * 2) + min(10, answer_count * 2)
        if avg_line_length >= 60:
            completeness += 5
        if text_length < 200:
            completeness -= 8

        logic = 58 + min(12, structure_hits * 3)
        if question_count >= 6:
            logic += 5

        professionalism = 58 + min(15, tech_hits * 2)
        if followup_count:
            professionalism += 4

        job_fit = 55 + min(15, tech_hits * 2)
        if question_count >= 6 and answer_count >= max(1, question_count - 1):
            job_fit += 5

        dimension_scores = {
            "completeness": self._clamp_score(completeness, default=60),
            "logic": self._clamp_score(logic, default=60),
            "professionalism": self._clamp_score(professionalism, default=60),
            "job_fit": self._clamp_score(job_fit, default=60),
        }

        total_score = round(sum(dimension_scores.values()) / len(dimension_scores))

        strengths: list[str] = []
        weaknesses: list[str] = []
        suggestions: list[str] = []

        if question_count >= 6 and answer_count >= max(1, question_count - 1):
            strengths.append("能够完成较完整的模拟面试流程，答题覆盖度较高。")
        if structure_hits:
            strengths.append("回答中具备一定结构化表达意识，能够使用分层思路展开。")
        if tech_hits:
            strengths.append("能够结合技术关键词展开，体现了一定的岗位相关性。")

        if avg_line_length < 45 or text_length < 300:
            weaknesses.append("部分回答仍偏短，建议进一步补充过程、细节和结果。")
        if structure_hits < 2:
            weaknesses.append("回答结构还可以更清晰，建议继续强化结论先行和分点展开。")
        if tech_hits < 2:
            weaknesses.append("与岗位相关的技术细节仍可进一步补强，避免过于概括。")

        suggestions.extend([
            "每道题优先使用“结论-过程-结果”的结构回答，减少长段无重点描述。",
            "准备 1~2 个可量化的项目例子，补充技术选型、难点处理和最终效果。",
            "围绕岗位关键词补齐高频概念、边界场景与常见追问点。",
        ])

        if followup_count:
            suggestions.append("对于追问题，可继续补充实现细节、权衡取舍和经验总结。")

        return {
            "total_score": total_score,
            "dimension_scores": dimension_scores,
            "comment": "整场模拟面试整体表现较为稳定，但仍可继续提升答题结构、技术细节与岗位贴合度。",
            "strengths": self._dedupe_preserve_order(strengths),
            "weaknesses": self._dedupe_preserve_order(weaknesses),
            "suggestions": self._dedupe_preserve_order(suggestions),
            "details": [],
        }

    def _parse_scoring_response(self, text: str) -> dict[str, Any]:
        """
        从 LLM 输出中解析评分 JSON。
        """
        parsed = parse_llm_json_response(text, default={})
        return ensure_dict(parsed, default={})

    def _build_result(
        self,
        parsed: dict[str, Any],
        mode: str,
        notice: str,
        source: str,
        scope: str = "single_answer",
        input_type: str = "answer",
        transcript_stats: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> AnswerScoringResult:
        """
        将解析结果构造为统一结构。
        """
        total_score = self._clamp_score(parsed.get("total_score", 0), default=0)
        dimension_scores = self._normalize_dimension_scores(parsed.get("dimension_scores", {}))

        comment = str(parsed.get("comment", "")).strip()
        if not comment:
            comment = "整体回答基本符合题意，但仍有提升空间。"

        strengths = self._normalize_feedback_items(
            parsed.get("strengths", []),
            default_item="回答中存在一定亮点，但仍需进一步展开。",
        )
        weaknesses = self._normalize_feedback_items(
            parsed.get("weaknesses", []),
            default_item="回答还可以进一步补充关键细节。",
        )
        suggestions = self._normalize_feedback_items(
            parsed.get("suggestions", []),
            default_item="建议结合项目经历补充具体做法和结果。",
        )

        details = self._normalize_details(parsed.get("details", []))

        meta: dict[str, Any] = {
            "source": source,
            "scope": scope,
            "input_type": input_type,
            "prompt_version": self.PROMPT_VERSION,
            "transcript_stats": transcript_stats or {},
        }
        if extra_meta:
            meta.update(extra_meta)

        return AnswerScoringResult(
            mode=mode,
            notice=notice,
            total_score=total_score,
            dimension_scores=dimension_scores,
            comment=comment,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            details=details,
            meta=meta,
        )

    def _invoke_mode(
        self,
        original_question: str,
        user_answer: str,
        resume_context: str,
        jd_context: str,
        rag_context: str,
        scope: str = "single_answer",
        input_type: str = "answer",
        transcript_stats: dict[str, Any] | None = None,
        extra_notice: str = "",
    ) -> AnswerScoringResult:
        """
        统一调用 prompt + LLM 进行评分。
        """
        try:
            context = self._merge_context(
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context=rag_context,
            )

            text = self.chain.invoke(
                {
                    "question": original_question,
                    "answer": user_answer,
                    "context": context,
                    "extra_notice": extra_notice,
                }
            )

            parsed = self._parse_scoring_response(text)
            transcript_stats = transcript_stats or (self._extract_transcript_stats(user_answer) if scope == "session_transcript" else None)

            if not parsed and scope == "session_transcript":
                logger.warning("[ScoringTool] session transcript 输出解析为空，使用 transcript-aware 兜底结果。")
                return self._build_result(
                    parsed=self._build_transcript_heuristic_payload(
                        user_answer=user_answer,
                        transcript_stats=transcript_stats or {},
                    ),
                    mode="session",
                    notice="已基于整场模拟面试生成兜底总评。",
                    source="heuristic_session",
                    scope=scope,
                    input_type=input_type,
                    transcript_stats=transcript_stats,
                    extra_meta={"raw_empty": True},
                )

            if not parsed:
                logger.warning("[ScoringTool] 评分输出解析为空，使用默认兜底结果。")
                return self._build_result(
                    parsed={
                        "total_score": 60,
                        "dimension_scores": {
                            "completeness": 60,
                            "logic": 60,
                            "professionalism": 60,
                            "job_fit": 60,
                        },
                        "comment": "回答基本符合题意，但细节和深度仍可加强。",
                        "strengths": ["回答能够覆盖问题核心。"],
                        "weaknesses": ["缺少更具体的项目细节或数据支撑。"],
                        "suggestions": ["建议使用 STAR 方法补充回答。"],
                        "details": [],
                    },
                    mode="fallback",
                    notice="评分结果解析失败，已使用默认兜底结果。",
                    source="llm_empty_output",
                    scope=scope,
                    input_type=input_type,
                    transcript_stats=transcript_stats,
                    extra_meta={"raw_empty": True},
                )

            if scope == "session_transcript":
                heuristic = self._build_transcript_heuristic_payload(
                    user_answer=user_answer,
                    transcript_stats=transcript_stats or {},
                )
                current_dim_scores = self._normalize_dimension_scores(parsed.get("dimension_scores", {}))
                current_total_score = self._clamp_score(parsed.get("total_score", 0), default=0)
                if not any(current_dim_scores.values()) or current_total_score <= 0:
                    parsed = {**heuristic, **parsed}
                    parsed["total_score"] = heuristic["total_score"]
                    parsed["dimension_scores"] = heuristic["dimension_scores"]
                    parsed["strengths"] = self._dedupe_preserve_order(
                        self._normalize_feedback_items(parsed.get("strengths", []), default_item="", limit=4)
                        + heuristic["strengths"]
                    )
                    parsed["weaknesses"] = self._dedupe_preserve_order(
                        self._normalize_feedback_items(parsed.get("weaknesses", []), default_item="", limit=4)
                        + heuristic["weaknesses"]
                    )
                    parsed["suggestions"] = self._dedupe_preserve_order(
                        self._normalize_feedback_items(parsed.get("suggestions", []), default_item="", limit=4)
                        + heuristic["suggestions"]
                    )

            return self._build_result(
                parsed=parsed,
                mode="session" if scope == "session_transcript" else ("rag_based" if rag_context.strip() else "fallback"),
                notice=(
                    "已完成整场模拟面试总结评分。"
                    if scope == "session_transcript"
                    else ("已完成后台答案评分。" if rag_context.strip() else "已基于通用面试经验完成后台答案评分。")
                ),
                source="rag" if rag_context.strip() else ("session" if scope == "session_transcript" else "fallback"),
                scope=scope,
                input_type=input_type,
                transcript_stats=transcript_stats,
            )
        except Exception as e:
            logger.error(f"[ScoringTool] 调用评分链失败: {str(e)}", exc_info=True)
            transcript_stats = transcript_stats or (self._extract_transcript_stats(user_answer) if scope == "session_transcript" else None)
            if scope == "session_transcript":
                heuristic = self._build_transcript_heuristic_payload(
                    user_answer=user_answer,
                    transcript_stats=transcript_stats or {},
                )
                return self._build_result(
                    parsed=heuristic,
                    mode="session",
                    notice="评分过程出现异常，已返回整场模拟面试的兜底总结。",
                    source="error_session",
                    scope=scope,
                    input_type=input_type,
                    transcript_stats=transcript_stats,
                    extra_meta={"error": str(e)},
                )
            return self._build_result(
                parsed={
                    "total_score": 60,
                    "dimension_scores": {
                        "completeness": 60,
                        "logic": 60,
                        "professionalism": 60,
                        "job_fit": 60,
                    },
                    "comment": "评分过程出现异常，已返回默认兜底结果。",
                    "strengths": ["回答基本覆盖了问题要点。"],
                    "weaknesses": ["缺少更具体的项目细节或数据支撑。"],
                    "suggestions": ["建议使用 STAR 方法补充回答。"],
                    "details": [],
                },
                mode="fallback",
                notice="评分过程出现异常，已返回默认兜底结果。",
                source="error",
                scope=scope,
                input_type=input_type,
                transcript_stats=transcript_stats,
                extra_meta={"error": str(e)},
            )

    def _score_with_scope(
        self,
        original_question: str,
        user_answer: str,
        resume_context: str = "",
        jd_context: str = "",
        rag_context: str = "",
        use_rag: bool = True,
        scope: str = "single_answer",
        input_type: str = "answer",
        transcript_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        original_question = self._clean_text(original_question)
        user_answer = self._clean_text(user_answer)
        resume_context = resume_context or ""
        jd_context = jd_context or ""
        rag_context = rag_context or ""

        if not original_question or not user_answer:
            result = self._build_result(
                parsed={
                    "total_score": 0,
                    "dimension_scores": {
                        "completeness": 0,
                        "logic": 0,
                        "professionalism": 0,
                        "job_fit": 0,
                    },
                    "comment": "题目或回答为空，无法进行有效评分。",
                    "strengths": [],
                    "weaknesses": ["输入内容不足，无法评估。"],
                    "suggestions": ["请先补充完整的题目和回答。"],
                    "details": [],
                },
                mode="skip",
                notice="题目或回答为空，已跳过评分。",
                source="empty_input",
                scope=scope,
                input_type=input_type,
                transcript_stats=transcript_stats or {},
                extra_meta={"reason": "empty_question_or_answer"},
            )
            logger.info("[ScoringTool] score_answer skipped due to empty input")
            return asdict(result)

        result = self._invoke_mode(
            original_question=original_question,
            user_answer=user_answer,
            resume_context=resume_context,
            jd_context=jd_context,
            rag_context=rag_context if use_rag and rag_context.strip() else "",
            scope=scope,
            input_type=input_type,
            transcript_stats=transcript_stats,
            extra_notice=(
                "未提供有效检索上下文，请基于通用评分标准进行评价。"
                if not (use_rag and rag_context.strip())
                else ""
            ),
        )

        logger.info(
            f"[ScoringTool] score_answer done | mode={result.mode} total_score={result.total_score}"
        )
        return asdict(result)

    def score_answer(
        self,
        original_question: str,
        user_answer: str,
        resume_context: str = "",
        jd_context: str = "",
        rag_context: str = "",
        use_rag: bool = True,
    ) -> dict[str, Any]:
        """
        统一评分入口。

        说明：
        - 评分结果只用于后台记录和最终总评
        - 面试过程不应直接展示评分内容
        - 上层 ReactAgent 只需要保存该结果并在最后统一汇总
        """
        is_session_transcript = self._looks_like_session_transcript(original_question, user_answer)
        transcript_stats = self._extract_transcript_stats(user_answer) if is_session_transcript else None
        scope = "session_transcript" if is_session_transcript else "single_answer"
        input_type = "transcript" if is_session_transcript else "answer"
        return self._score_with_scope(
            original_question=original_question,
            user_answer=user_answer,
            resume_context=resume_context,
            jd_context=jd_context,
            rag_context=rag_context,
            use_rag=use_rag,
            scope=scope,
            input_type=input_type,
            transcript_stats=transcript_stats,
        )

    def score_session_transcript(
        self,
        transcript: str,
        resume_context: str = "",
        jd_context: str = "",
        rag_context: str = "",
        use_rag: bool = False,
    ) -> dict[str, Any]:
        """
        直接对整场模拟面试 transcript 做一次总评。
        """
        transcript = self._clean_text(transcript)
        transcript_stats = self._extract_transcript_stats(transcript)
        return self._score_with_scope(
            original_question="整场模拟面试总体评估",
            user_answer=transcript,
            resume_context=resume_context,
            jd_context=jd_context,
            rag_context=rag_context,
            use_rag=use_rag,
            scope="session_transcript",
            input_type="transcript",
            transcript_stats=transcript_stats,
        )


def score_interview_answer(
    original_question: str,
    user_answer: str,
    resume_context: str = "",
    jd_context: str = "",
    rag_context: str = "",
    use_rag: bool = True,
) -> dict[str, Any]:
    """
    便捷函数：直接评分。
    """
    tool = ScoringTool()
    return tool.score_answer(
        original_question=original_question,
        user_answer=user_answer,
        resume_context=resume_context,
        jd_context=jd_context,
        rag_context=rag_context,
        use_rag=use_rag,
    )


if __name__ == "__main__":
    demo_question = "请介绍一下你对 RAG 的理解"
    demo_answer = "RAG 就是先检索再生成，可以提升回答的相关性。"
    demo_resume = "技能：Python、RAG、Agent；项目：AI面试准备助手"
    demo_jd = "岗位：AI Agent 应用开发工程师；要求：熟悉RAG、Prompt、工具调用"
    demo_rag = "参考：RAG 的核心是检索召回与生成结合，评估重点包括召回率与忠实度。"

    result = score_interview_answer(
        original_question=demo_question,
        user_answer=demo_answer,
        resume_context=demo_resume,
        jd_context=demo_jd,
        rag_context=demo_rag,
        use_rag=True,
    )
    print("=== 答案评分结果 ===")
    print(result)

