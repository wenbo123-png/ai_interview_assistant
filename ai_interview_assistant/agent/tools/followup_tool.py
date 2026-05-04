"""
追问生成工具（最终版）

作用：
- 输入用户回答、原始问题、简历信息、JD 信息、RAG 上下文
- 生成 1~3 个针对性追问
- 用于模拟面试中的深入追问环节

最终要求：
- 追问时只展示问题本身
- 不展示 reason / focus 等内部字段给前端主链路
- 每道题最多追问两轮
- 只对一部分题目触发追问，不是所有题都追问
- 与 ReactAgent 的模拟面试闭环衔接
"""

from __future__ import annotations

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
from ai_interview_assistant.utils.prompt_loader import load_followup_prompt


@dataclass
class FollowUpQuestion:
    """单个追问结构。"""
    question: str
    focus: str
    reason: str


@dataclass
class FollowUpResult:
    """追问生成结果。"""
    mode: str  # rag_based / fallback / skip
    notice: str
    followups: list[dict[str, Any]]
    meta: dict[str, Any]


class FollowUpTool:
    """
    追问生成工具。

    输入：
    - original_question: 原始题目
    - user_answer: 用户回答
    - resume_context: 简历结构化信息
    - jd_context: JD结构化信息
    - rag_context: 检索上下文（可为空）

    输出：
    - 1~3 个追问问题
    """

    def __init__(self) -> None:
        self.model = chat_model
        self.parser = StrOutputParser()

        # 从 prompts/followup_prompt.txt 加载提示词
        self.prompt_text = load_followup_prompt()
        self.prompt = PromptTemplate.from_template(self.prompt_text)

        # 同一份 prompt 文件同时支持 RAG 命中和 fallback 场景
        self.chain = self.prompt | self.model | self.parser

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _should_generate_followup(original_question: str, user_answer: str, allow_followup: bool) -> bool:
        """
        是否允许生成追问。

        说明：
        - 追问是否触发的核心控制权交给上层 ReactAgent
        - 这里仅做最小校验，避免空题、空回答或显式禁止时仍继续生成
        """
        if not allow_followup:
            return False
        if not str(original_question or "").strip():
            return False
        if not str(user_answer or "").strip():
            return False

        return True

    @staticmethod
    def _build_safe_followup_question(original_question: str, user_answer: str) -> str:
        """
        当模型输出为空时，构造一个兜底追问。
        """
        q = (original_question or "").lower().strip()
        a = (user_answer or "").strip()

        if len(a) < 25:
            return "能否结合你的具体实践，再补充一下思路、步骤和结果？"

        if any(k in q for k in ("项目", "实现", "优化", "排查", "调试", "设计", "落地")):
            return "你能进一步说明这里最关键的实现细节、难点以及你的处理方式吗？"

        if any(k in q for k in ("agent", "rag", "llm", "langchain", "python", "sql", "数据库")):
            return "能否结合一个具体场景，说明你是如何设计和落地这个方案的？"

        return "请结合一个具体例子，继续展开你的思路、动作和最终效果。"

    @staticmethod
    def _normalize_followups(raw_followups: list[Any]) -> list[dict[str, Any]]:
        """
        规范化追问列表，避免字段缺失。
        只保留前一条，确保每次只输出一个追问问题。
        """
        normalized: list[dict[str, Any]] = []
        for item in raw_followups:
            item_dict = ensure_dict(item, default={})
            question = str(item_dict.get("question", "")).strip()
            if not question:
                continue

            focus = str(item_dict.get("focus", "")).strip()
            reason = str(item_dict.get("reason", "")).strip()

            normalized.append(
                asdict(
                    FollowUpQuestion(
                        question=question,
                        focus=focus,
                        reason=reason,
                    )
                )
            )

            if len(normalized) >= 1:
                break

        return normalized

    def _parse_followups(self, text: str) -> list[dict[str, Any]]:
        """
        从 LLM 输出中解析 followups。
        """
        parsed = parse_llm_json_response(text, default={})
        data = ensure_dict(parsed, default={})
        raw_followups = ensure_list(data.get("followups", []), default=[])
        return self._normalize_followups(raw_followups)

    def _invoke_rag_mode(
        self,
        original_question: str,
        user_answer: str,
        resume_context: str,
        jd_context: str,
        rag_context: str,
    ) -> FollowUpResult:
        """
        检索命中场景下生成追问。
        """
        text = self.chain.invoke(
            {
                "question": original_question,
                "answer": user_answer,
                "resume_context": resume_context,
                "jd_context": jd_context,
                "rag_context": rag_context,
            }
        )
        followups = self._parse_followups(text)

        if not followups:
            logger.warning("[FollowUpTool] RAG mode 解析为空，自动降级 fallback。")
            return self._invoke_fallback_mode(
                original_question=original_question,
                user_answer=user_answer,
                resume_context=resume_context,
                jd_context=jd_context,
                extra_notice="RAG输出为空，自动切换通用追问模式。",
            )

        return FollowUpResult(
            mode="rag_based",
            notice="已结合检索上下文生成追问。",
            followups=followups,
            meta={"source": "rag"},
        )

    def _invoke_fallback_mode(
        self,
        original_question: str,
        user_answer: str,
        resume_context: str,
        jd_context: str,
        extra_notice: str,
    ) -> FollowUpResult:
        """
        通用兜底场景下生成追问。
        """
        text = self.chain.invoke(
            {
                "question": original_question,
                "answer": user_answer,
                "resume_context": resume_context,
                "jd_context": jd_context,
                "rag_context": "",
            }
        )
        followups = self._parse_followups(text)

        return FollowUpResult(
            mode="fallback",
            notice="已基于通用面试经验生成追问。",
            followups=followups,
            meta={"source": "fallback", "extra_notice": extra_notice},
        )

    def generate_followups(
        self,
        original_question: str,
        user_answer: str,
        resume_context: str = "",
        jd_context: str = "",
        rag_context: str = "",
        use_rag: bool = True,
        allow_followup: bool = True,
    ) -> dict[str, Any]:
        """
        统一追问生成入口。

        规则：
        1) 由上层 ReactAgent 决定是否允许本次追问
        2) 这里只做最小校验：空题、空回答或显式禁止则跳过
        3) 适合的话，再根据 RAG / fallback 生成一条追问
        """
        original_question = self._clean_text(original_question)
        user_answer = self._clean_text(user_answer)
        resume_context = resume_context or ""
        jd_context = jd_context or ""
        rag_context = rag_context or ""

        if not self._should_generate_followup(original_question, user_answer, allow_followup):
            result = FollowUpResult(
                mode="skip",
                notice="当前题目不触发追问。",
                followups=[],
                meta={"source": "rule_skip", "allow_followup": allow_followup},
            )
            logger.info(
                f"[FollowUpTool] generate_followups done | mode={result.mode} followup_count=0"
            )
            return asdict(result)

        if use_rag and rag_context.strip():
            result = self._invoke_rag_mode(
                original_question=original_question,
                user_answer=user_answer,
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context=rag_context,
            )
        else:
            result = self._invoke_fallback_mode(
                original_question=original_question,
                user_answer=user_answer,
                resume_context=resume_context,
                jd_context=jd_context,
                extra_notice="未提供有效检索上下文，请基于通用经验生成追问。",
            )

        # 再做一次保险截断，确保最多一条
        result.followups = (result.followups or [])[:1]

        if not result.followups:
            safe_question = self._build_safe_followup_question(original_question, user_answer)
            result.followups = [
                asdict(
                    FollowUpQuestion(
                        question=safe_question,
                        focus="深入理解",
                        reason="兜底追问，保证追问链路稳定。",
                    )
                )
            ]
            result.mode = "fallback"
            result.notice = "已生成兜底追问。"
            result.meta = getattr(result, "meta", {}) or {}
            result.meta["source"] = "fallback_safe"

        logger.info(
            f"[FollowUpTool] generate_followups done | mode={result.mode} "
            f"followup_count={len(result.followups)}"
        )
        return asdict(result)


def generate_followup_questions(
    original_question: str,
    user_answer: str,
    resume_context: str = "",
    jd_context: str = "",
    rag_context: str = "",
    use_rag: bool = True,
) -> dict[str, Any]:
    """
    便捷函数：直接生成追问。
    """
    tool = FollowUpTool()
    return tool.generate_followups(
        original_question=original_question,
        user_answer=user_answer,
        resume_context=resume_context,
        jd_context=jd_context,
        rag_context=rag_context,
        use_rag=use_rag,
    )


if __name__ == "__main__":
    demo_question = "请介绍一下 RAG 的基本原理"
    demo_answer = "RAG 就是检索增强生成，先检索再生成。"
    demo_resume = "技能：Python、RAG、Agent；项目：AI面试准备助手"
    demo_jd = "岗位：AI Agent 应用开发工程师；要求：熟悉RAG、Prompt、工具调用"
    demo_rag = "参考：RAG 的核心是检索召回与生成结合，评估重点包括召回率与忠实度。"

    result = generate_followup_questions(
        original_question=demo_question,
        user_answer=demo_answer,
        resume_context=demo_resume,
        jd_context=demo_jd,
        rag_context=demo_rag,
        use_rag=True,
    )
    print("=== 追问生成结果 ===")
    print(result)