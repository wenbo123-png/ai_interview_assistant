from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_interview_assistant.agent.tools.resume_tool import ResumeTool
from ai_interview_assistant.agent.tools.jd_tool import JDTool
from ai_interview_assistant.agent.tools.question_tool import QuestionTool
from ai_interview_assistant.agent.tools.followup_tool import FollowUpTool
from ai_interview_assistant.agent.tools.scoring_tool import ScoringTool
from ai_interview_assistant.agent.tools.rag_tool import RagTool
from ai_interview_assistant.model.factory import chat_model
from ai_interview_assistant.utils.json_utils import parse_llm_json_response, ensure_dict, ensure_list
from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.prompt_loader import load_intent_classify_prompt, load_final_evaluation_prompt


@dataclass
class AgentSessionState:
    """
    会话状态：
    - 支持多轮对话
    - 支持工作流编排
    - 支持模拟面试闭环
    """
    has_resume: bool = False
    has_jd: bool = False

    # idle / mock_interview / qa / non_target / suggestion / question_generation
    current_mode: str = "idle"
    last_intent: str = ""
    last_query: str = ""
    last_answer: str = ""
    runtime_meta: dict[str, Any] = field(default_factory=dict)

    # 解析后的文件内容
    resume_data: dict[str, Any] | None = None
    jd_data: dict[str, Any] | None = None

    # 最近一次工作流结果
    last_questions: list[dict[str, Any]] = field(default_factory=list)
    last_followups: list[dict[str, Any]] = field(default_factory=list)
    last_score: dict[str, Any] | None = None

    # 模拟面试状态
    mock_interview_started: bool = False
    current_question_index: int = -1
    current_question: dict[str, Any] | None = None
    current_answer: str = ""
    awaiting_followup_answer: bool = False
    pending_followup_round: int = 0

    # 最终总评收口
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)
    total_followup_count: int = 0
    asked_question_count: int = 0
    answered_question_count: int = 0
    followup_question_indices: set[int] = field(default_factory=set)
    # 当前题目已触发的追问轮次（最多 2 轮）
    current_question_followup_count: int = 0
    final_summary_ready: bool = False


class ReactAgent:
    """
    面试准备助手 Agent

    核心职责：
    1) 维护会话状态
    2) 接收并解析简历 / JD
    3) 识别用户目标
    4) 调用对应工具
    5) 支持模拟面试逐题推进、最多两次追问、后台评分
    6) 所有题目结束后统一输出最终总评
    7) 保留无文件问答 / 拒答逻辑
    """

    # 意图分类的有效标签集合
    VALID_INTENTS: set[str] = {
        "greeting", "mock_interview", "question_generation", "suggestion", "qa", "non_target",
    }

    def __init__(self):
        self.resume_tool = ResumeTool()
        self.jd_tool = JDTool()
        self.question_tool = QuestionTool()
        self.followup_tool = FollowUpTool()
        self.scoring_tool = ScoringTool()
        self.rag_tool = RagTool()

        self.session_state = AgentSessionState()

        # LLM 意图分类 chain
        intent_prompt_text = load_intent_classify_prompt()
        intent_prompt = PromptTemplate.from_template(intent_prompt_text)
        self.intent_chain = intent_prompt | chat_model | StrOutputParser()

        # 最终总评 chain
        eval_prompt_text = load_final_evaluation_prompt()
        eval_prompt = PromptTemplate.from_template(eval_prompt_text)
        self.eval_chain = eval_prompt | chat_model | StrOutputParser()

        # 意图分类缓存：相同输入直接返回上次结果
        self._intent_cache: dict[str, str] = {}

    def _reset_mock_session(self) -> None:
        """
        结束一轮模拟面试后，重置与模拟面试相关的状态。
        """
        self.session_state.mock_interview_started = False
        self.session_state.current_question_index = -1
        self.session_state.current_question = None
        self.session_state.current_answer = ""
        self.session_state.awaiting_followup_answer = False
        self.session_state.pending_followup_round = 0
        self.session_state.current_question_followup_count = 0
        self.session_state.current_mode = "idle"
        self.session_state.last_questions = []
        self.session_state.last_followups = []
        self.session_state.last_score = None
        self.session_state.evaluation_history = []
        self.session_state.total_followup_count = 0
        self.session_state.asked_question_count = 0
        self.session_state.answered_question_count = 0
        self.session_state.followup_question_indices = set()
        self.session_state.final_summary_ready = False

    def _update_session_state(
        self,
        query: str,
        runtime_context: dict[str, Any] | None = None,
        intent: str | None = None,
    ) -> None:
        """
        将当前轮输入合并到会话状态中。
        """
        ctx = runtime_context or {}

        if "has_resume" in ctx:
            self.session_state.has_resume = bool(ctx.get("has_resume", False))
        if "has_jd" in ctx:
            self.session_state.has_jd = bool(ctx.get("has_jd", False))

        self.session_state.last_query = query or ""
        if intent is not None:
            self.session_state.last_intent = intent
            self.session_state.current_mode = intent

        self.session_state.runtime_meta.update(ctx)

        resume_input = ctx.get("resume_input")
        jd_input = ctx.get("jd_input")

        if resume_input:
            try:
                self.session_state.resume_data = self.resume_tool.parse(str(resume_input))
                self.session_state.has_resume = True
                logger.info("[AGENT] resume parsed and stored in session state")
            except Exception as e:
                logger.warning(f"[AGENT] 简历解析失败: {e}")

        if jd_input:
            try:
                self.session_state.jd_data = self.jd_tool.parse(str(jd_input))
                self.session_state.has_jd = True
                logger.info("[AGENT] JD parsed and stored in session state")
            except Exception as e:
                logger.warning(f"[AGENT] JD解析失败: {e}")

    def _classify_intent(self, query: str) -> str:
        """
        意图识别：优先 LLM 分类，失败时降级为关键词匹配。
        LLM 能理解语义（如"我想练习一下"→ mock_interview），
        关键词匹配只作为兜底保障。相同输入直接返回缓存结果。
        """
        q = (query or "").strip()
        if not q:
            return "non_target"

        # 缓存命中：相同 query 直接返回
        if q in self._intent_cache:
            logger.info(f"[AGENT] intent_cache 命中: {q} → {self._intent_cache[q]}")
            return self._intent_cache[q]

        # LLM 分类
        try:
            result = self.intent_chain.invoke({"query": q}).strip().lower()
            # 提取标签：兼容 LLM 输出 "mock_interview\n" 或 "类别：mock_interview"
            for intent in self.VALID_INTENTS:
                if intent in result:
                    logger.info(f"[AGENT] intent_llm query={q} → {intent}")
                    self._intent_cache[q] = intent
                    return intent
            logger.warning(f"[AGENT] intent_llm 返回无效标签: {result}，降级为关键词匹配")
        except Exception as e:
            logger.warning(f"[AGENT] intent_llm 调用失败: {e}，降级为关键词匹配")

        # 关键词兜底
        intent = self._classify_intent_by_keywords(q)
        self._intent_cache[q] = intent
        return intent

    @staticmethod
    def _classify_intent_by_keywords(q: str) -> str:
        """关键词兜底意图识别，确保 LLM 不可用时系统仍能运行。"""
        q = q.lower().strip()

        mock_keywords = {
            "模拟面试", "面试我", "评估", "追问", "评分", "测评", "mock interview",
        }
        question_keywords = {
            "出题", "面试题", "题目", "题库", "生成题", "给我题", "参考答案",
        }
        suggestion_keywords = {
            "建议", "准备建议", "面试准备", "复习建议", "准备方向", "学习建议",
        }
        qa_keywords = {
            "什么是", "如何", "为什么", "介绍", "原理", "区别", "面试", "岗位",
            "技术", "算法", "rag", "agent", "llm", "知识", "问题",
        }

        if any(k in q for k in mock_keywords):
            return "mock_interview"
        if any(k in q for k in question_keywords):
            return "question_generation"
        if any(k in q for k in suggestion_keywords):
            return "suggestion"
        if any(k in q for k in qa_keywords):
            return "qa"

        return "non_target"

    @staticmethod
    def _has_effective_file_context(session_state: AgentSessionState) -> bool:
        return bool(session_state.has_resume or session_state.has_jd)

    @staticmethod
    def _non_target_message() -> str:
        return (
            "我主要用于面试准备与专业知识问答，无法回答业务能力之外的问题。\n"
            "如果你是想问专业知识或面试相关的问题，请描述得更清楚些。"
        )

    @staticmethod
    def _need_file_message() -> str:
        return "缺少必要的参考资料，请先提供简历或JD相关文件，我才能为你提供对应服务。"

    @staticmethod
    def _format_resume_jd_context(session_state: AgentSessionState) -> tuple[str, str]:
        """
        把解析后的简历/JD结构化内容转成工具可用文本。
        """
        resume_context = ""
        jd_context = ""

        if session_state.resume_data:
            resume_context = (
                f"简历摘要：{session_state.resume_data.get('summary', '')}\n"
                f"技能：{', '.join(session_state.resume_data.get('skills', []))}\n"
                f"项目：{', '.join(session_state.resume_data.get('projects', []))}\n"
                f"实习：{', '.join(session_state.resume_data.get('internship', []))}\n"
                f"优势：{', '.join(session_state.resume_data.get('strengths', []))}\n"
                f"关键词：{', '.join(session_state.resume_data.get('keywords', []))}"
            )

        if session_state.jd_data:
            jd_context = (
                f"JD摘要：{session_state.jd_data.get('summary', '')}\n"
                f"职责：{', '.join(session_state.jd_data.get('responsibilities', []))}\n"
                f"要求：{', '.join(session_state.jd_data.get('requirements', []))}\n"
                f"加分项：{', '.join(session_state.jd_data.get('plus_points', []))}\n"
                f"面试重点：{', '.join(session_state.jd_data.get('interview_focus', []))}\n"
                f"关键词：{', '.join(session_state.jd_data.get('keywords', []))}"
            )

        return resume_context, jd_context

    @staticmethod
    def _rag_dict_to_text(result: dict[str, Any]) -> str:
        """
        将 RAG 返回结果整理成适合展示的文本。
        """
        parts: list[str] = []
        notice = str(result.get("notice", "")).strip()
        answer = str(result.get("answer", "")).strip()
        if notice:
            parts.append(notice)
        if answer:
            parts.append(answer)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _to_markdown(parts: list[str]) -> str:
        """将段落列表转换为 st.markdown 兼容文本（非空行末尾加两个空格做硬换行）。"""
        md_lines: list[str] = []
        for part in parts:
            if part == "":
                md_lines.append("")
            else:
                md_lines.append(part + "  ")
        return "\n".join(md_lines)

    @staticmethod
    def _clean_question_text(text: str) -> str:
        """
        清理题干文本，去掉多余空白。
        """
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _unique_nonempty(items: list[str]) -> list[str]:
        """去重并保留顺序。"""
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(text)
        return unique

    @staticmethod
    def _rating_and_prediction(total_score: int) -> tuple[str, str]:
        """根据总分给出评级与通关预判。"""
        if total_score >= 90:
            return "优秀", "大概率通过"
        if total_score >= 80:
            return "良好", "较有希望通过"
        if total_score >= 70:
            return "中等", "边缘待定"
        if total_score >= 60:
            return "待加强", "通过率较低"
        return "基础薄弱", "建议重点备考"

    @staticmethod
    def _score_band_comment(score: int, strong: str, medium: str, weak: str) -> str:
        if score >= 85:
            return strong
        if score >= 70:
            return medium
        return weak

    def _build_followup_question(self, original_question: str, user_answer: str) -> str:
        """当模型未返回有效追问时，使用一个安全兜底追问。"""
        _ = original_question, user_answer
        return "请进一步结合你的项目经历、实现细节或优化结果展开说明。"

    def _create_followup_question(
        self,
        original_question: str,
        user_answer: str,
        round_index: int,
        resume_context: str = "",
        jd_context: str = "",
        rag_context: str = "",
    ) -> str:
        """
        基于当前回答生成一个追问问题。

        仅提取一条有效追问；如果模型没有返回可用内容，则使用兜底追问。
        """
        followup_result = self.followup_tool.generate_followups(
            original_question=original_question,
            user_answer=user_answer,
            resume_context=resume_context,
            jd_context=jd_context,
            rag_context=rag_context,
            use_rag=True,
        )

        for item in followup_result.get("followups", []) or []:
            q_text = self._clean_question_text(item.get("question", ""))
            if q_text:
                return q_text

        if round_index >= 2:
            return "请继续补充更具体的实现细节、关键风险以及最终效果。"
        return self._build_followup_question(original_question=original_question, user_answer=user_answer)

    @staticmethod
    def _get_reference_answer(item: dict[str, Any]) -> str:
        """
        获取参考答案：
        1) 优先使用工具返回的 reference_answer
        2) 没有的话，用 reference_points 兜底拼一个简洁答案
        """
        reference_answer = str(item.get("reference_answer", "") or "").strip()
        if reference_answer:
            return reference_answer

        ref_points = item.get("reference_points", []) or []
        if isinstance(ref_points, list) and ref_points:
            lines = [
                f"{idx}. {str(point).strip()}"
                for idx, point in enumerate(ref_points, 1)
                if str(point).strip()
            ]
            if lines:
                return "\n".join(lines)

        return "暂无参考答案。"

    @staticmethod
    def _question_list_to_text(
        title: str,
        notice: str,
        questions: list[dict[str, Any]],
        show_meta: bool = True,
        include_reference_answer: bool = False,
    ) -> str:
        """
        格式化题目列表输出。

        show_meta=True:
            展示题型/考察点/参考要点等元信息（用于“面试题生成模式”）
        show_meta=False:
            仅展示题目本身（用于“模拟面试模式”）
        include_reference_answer=True:
            在题目列表后统一给出参考答案示例
        """
        answer_parts: list[str] = [title.strip()]

        if notice.strip():
            answer_parts.append(notice.strip())

        if not questions:
            answer_parts.append("暂无可生成的题目。")
            return ReactAgent._to_markdown(answer_parts)

        for idx, item in enumerate(questions, 1):
            if idx > 1:
                answer_parts.append("")  # 题目间空行分隔

            question_text = ReactAgent._clean_question_text(item.get("question", ""))

            if show_meta:
                answer_parts.append(f"{idx}. {question_text}")

                qtype = str(item.get("qtype", "")).strip()
                focus = str(item.get("focus", "")).strip()
                ref_points = item.get("reference_points", []) or []
                reference_answer = ReactAgent._get_reference_answer(item)

                if qtype:
                    answer_parts.append(f"  题型：{qtype}")
                if focus:
                    answer_parts.append(f"  考察点：{focus}")

                if include_reference_answer:
                    approach = str(item.get("approach", "")).strip()
                    answer_parts.append("  答题思路/参考答案：")
                    if approach:
                        answer_parts.append(f"    【答题思路】{approach}")
                    if reference_answer.strip():
                        label = "    【参考答案】" if approach else "    "
                        for line in reference_answer.splitlines():
                            line_text = str(line).rstrip()
                            if line_text.strip():
                                answer_parts.append(f"{label}{line_text}")
                                label = "    "
                    elif not approach:
                        answer_parts.append("    暂无答题思路和参考答案。")
            else:
                # 模拟面试模式：只展示当前题，不展示题型/考察点/参考要点
                answer_parts.append(f"【当前题目 {idx}/{len(questions)}】")
                answer_parts.append(question_text)

        return ReactAgent._to_markdown([part for part in answer_parts if str(part).strip() or part == ""])

    def _build_transcript(self, history: list[dict[str, Any]]) -> str:
        """从面试历史中构建 transcript 文本。"""
        transcript_parts: list[str] = []
        for idx, item in enumerate(history, 1):
            q = str(item.get("question", "") or "").strip()
            a = str(item.get("answer", "") or "").strip()
            followups = item.get("followups", []) or []

            transcript_parts.append(f"第{idx}题：{q}")
            transcript_parts.append(f"用户回答：{a}")

            if followups:
                for f_idx, followup in enumerate(followups, 1):
                    f_q = self._clean_question_text(followup.get("question", ""))
                    f_a = self._clean_question_text(followup.get("answer", ""))
                    if f_q:
                        transcript_parts.append(f"追问{f_idx}：{f_q}")
                    if f_a:
                        transcript_parts.append(f"追问回答{f_idx}：{f_a}")
            transcript_parts.append("")
        return "\n".join(transcript_parts).strip()

    def _format_llm_evaluation(self, parsed: dict[str, Any]) -> str:
        """将 LLM 返回的总评 JSON 格式化为 markdown 输出。"""
        total_score = int(parsed.get("total_score", 0) or 0)
        rating = str(parsed.get("rating", "待定") or "待定")
        prediction = str(parsed.get("prediction", "待定") or "待定")
        overall_comment = str(parsed.get("overall_comment", "") or "").strip()

        dimensions = ensure_dict(parsed.get("dimensions", {}))
        dim_prof = ensure_dict(dimensions.get("professionalism", {}))
        dim_logic = ensure_dict(dimensions.get("logic", {}))
        dim_content = ensure_dict(dimensions.get("content_quality", {}))
        dim_fit = ensure_dict(dimensions.get("job_fit", {}))

        strengths = ensure_list(parsed.get("strengths", []))
        weaknesses = ensure_list(parsed.get("weaknesses", []))
        suggestions = ensure_list(parsed.get("suggestions", []))

        parts: list[str] = [
            "【最终总评】",
            "一、整体总评",
            overall_comment or "暂无整体评语。",
            "",
            "二、核心维度评估",
            f"1. 专业知识（{dim_prof.get('score', 0)}分）：{dim_prof.get('comment', '暂无评语。')}",
            f"2. 答题逻辑（{dim_logic.get('score', 0)}分）：{dim_logic.get('comment', '暂无评语。')}",
            f"3. 内容质量（{dim_content.get('score', 0)}分）：{dim_content.get('comment', '暂无评语。')}",
            f"4. 岗位匹配（{dim_fit.get('score', 0)}分）：{dim_fit.get('comment', '暂无评语。')}",
            "",
            "三、核心短板",
        ]

        for item in weaknesses:
            text = str(item or "").strip()
            if text:
                parts.append(f"- {text}")

        parts.extend(["", "四、提升建议"])
        for item in suggestions:
            text = str(item or "").strip()
            if text:
                parts.append(f"- {text}")

        parts.extend([
            "",
            "五、综合评定",
            f"评级：{rating}",
            f"模拟面试通关预判：{prediction}",
            "",
            "说明：模拟面试不提供参考答案，若想知道答案和答题思路，请在无文件模式下单独询问。",
        ])
        return ReactAgent._to_markdown(parts)

    def _build_rule_based_evaluation(
        self,
        history: list[dict[str, Any]],
        dim_scores: dict[str, int],
    ) -> str:
        """
        规则模板兜底：当 LLM 总评失败时，使用预写模板生成总评。
        逻辑与原版一致。
        """
        professional_score = dim_scores.get("professionalism", 0)
        logic_score = dim_scores.get("logic", 0)
        content_score = dim_scores.get("completeness", 0)
        job_fit_score = dim_scores.get("job_fit", 0)

        score_values = [s for s in [professional_score, logic_score, content_score, job_fit_score] if s > 0]
        total_score = round(sum(score_values) / len(score_values)) if score_values else 0

        rating, prediction = self._rating_and_prediction(total_score)

        strongest_dim = max(
            [(professional_score, "专业知识"), (logic_score, "答题逻辑"),
             (content_score, "内容质量"), (job_fit_score, "岗位匹配")],
            key=lambda x: x[0],
        )[1]
        weakest_dim = min(
            [(professional_score, "专业知识"), (logic_score, "答题逻辑"),
             (content_score, "内容质量"), (job_fit_score, "岗位匹配")],
            key=lambda x: x[0],
        )[1]

        overall_summary = self._score_band_comment(
            total_score,
            strong=f"本次模拟面试整体表现扎实稳健，能够全面且精准地覆盖岗位核心能力要求，回答问题逻辑严谨、结构清晰条理分明，且在 {strongest_dim} 相关维度方面表现尤为亮眼突出。",
            medium=f"本次模拟面试整体表现平稳稳健，能够紧密围绕岗位核心要求展开条理清晰的回答，基础发挥较为稳定，但在 {weakest_dim} 相关专业内容上仍有较大的优化提升空间。",
            weak=f"本次模拟面试具备一定的知识储备和表达基础，整体完成了基础作答，但在岗位核心知识掌握、语言表达逻辑和岗位需求贴合度方面仍需进行系统的学习与加强。",
        )

        def dimension_note(score: int, strong: str, medium: str, weak: str) -> str:
            return self._score_band_comment(score, strong=strong, medium=medium, weak=weak)

        core_shortboards: list[str] = []
        if professional_score < 80:
            core_shortboards.append("专业知识的系统性构建和知识边界的深度理解仍需进一步加强，尤其要重点补齐核心关键概念、底层原理以及面试官高频常见追问内容。")
        if logic_score < 80:
            core_shortboards.append("整体答题的逻辑结构还可以更加清晰聚焦，建议有意识减少无关的发散式表达，重点强化结论先行的答题习惯和逻辑分层展开的表达方式。")
        if content_score < 80:
            core_shortboards.append("面试回答的干货知识密度和实际案例支撑力度还可以进一步提升，尽量避免仅停留在基础概念的浅层描述层面。")
        if job_fit_score < 80:
            core_shortboards.append("个人能力与岗位要求的匹配映射还可以表现得更加主动，建议将自身项目经历、专业能力与岗位JD核心重点进行更紧密的精准对齐。")
        core_shortboards = self._unique_nonempty(core_shortboards)[:4]
        if not core_shortboards:
            core_shortboards.append("当前面试整体表现均衡核心短板不突出，但仍建议持续打磨语言表达精度、项目案例完整度和岗位需求贴合度。")

        improvement_suggestions: list[str] = []
        if professional_score < 80:
            improvement_suggestions.append("正式面试前按照目标岗位方向系统梳理核心专业概念、高频常见问题与边界应用场景，搭建形成可复用的系统化知识框架。")
        if logic_score < 80:
            improvement_suggestions.append("现场回答问题时尽量采用结论先行-阐述原因-讲解过程-说明结果的标准逻辑结构，有效提升回答的条理性和面试官的可听性。")
        if content_score < 80:
            improvement_suggestions.append("针对每道面试题目至少准备1个贴合的具体项目实战例子，同时补充关键技术决策、详细实现细节和量化的结果数据支撑。")
        if job_fit_score < 80:
            improvement_suggestions.append("面试回答中主动精准对齐岗位JD核心关键词，重点强调自身能力能为目标岗位解决实际问题、创造核心价值。")
        improvement_suggestions.append("日常模拟面试训练时优先打磨先说核心结论再逐步展开细节的答题节奏，避免长篇段落内容堆砌导致逻辑混乱。")
        improvement_suggestions = self._unique_nonempty(improvement_suggestions)[:5]

        parts: list[str] = [
            "【最终总评】",
            "一、整体总评",
            overall_summary,
            "",
            "二、核心维度评估",
            f"1. 专业知识：{dimension_note(professional_score, '对岗位核心概念和底层技术原理掌握扎实稳固，能够精准全面地回应各类关键问题。', '对核心专业概念有基础掌握，但在知识边界理解和内容深度展开上仍可进一步补强。', '专业知识体系掌握不够稳定，部分核心关键概念和技术原理仍需系统化复习巩固。')}",
            f"2. 答题逻辑：{dimension_note(logic_score, '回答条理清晰分明，能够熟练做到结论先行并按照逻辑层次有序展开阐述。', '整体答题逻辑基本清晰顺畅，但个别题目仍存在内容展开顺序不够合理稳定的问题。', '答题逻辑较为松散，建议重点加强结构化表达能力和核心重点的收束总结能力。')}",
            f"3. 内容质量：{dimension_note(content_score, '回答内容干货储备充足，能够结合实战案例和方法展开阐述，整体信息密度较高。', '内容质量总体达标可用，但案例细节打磨、结果数据量化或问题解决过程还可进一步补充。', '回答内容偏空泛单薄，建议多增加真实项目事实、核心技术细节和可量化验证的实际结果。')}",
            f"4. 岗位匹配：{dimension_note(job_fit_score, '与岗位要求匹配度较高，能够主动精准呼应JD重点并充分体现对目标岗位的深度理解。', '岗位匹配度较为明确清晰，但与JD核心关键词的对应关联还可以更加突出明显。', '岗位匹配表达力度偏弱，建议更主动将个人实战经历与岗位核心要求建立紧密对应。')}",
            "",
            "三、核心短板",
        ]

        for item in core_shortboards:
            parts.append(f"- {item}")

        parts.extend(["", "四、提升建议"])
        for item in improvement_suggestions:
            parts.append(f"- {item}")

        parts.extend([
            "",
            "五、综合评定",
            f"评级：{rating}",
            f"模拟面试通关预判：{prediction}",
            "",
            "说明：模拟面试不提供参考答案，若想知道答案和答题思路，请在无文件模式下单独询问。",
        ])
        return ReactAgent._to_markdown(parts)

    def _summarize_final_evaluation(
        self,
        history: list[dict[str, Any]],
        resume_context: str = "",
        jd_context: str = "",
    ) -> str:
        """
        基于整场模拟面试的问答历史生成最终总评。

        优先使用 LLM 生成个性化评语，失败时降级到规则模板。
        """
        if not history:
            return "本轮模拟面试已结束，但没有可汇总的答题记录。"

        transcript = self._build_transcript(history)

        # 优先尝试 LLM 生成个性化总评
        try:
            llm_output = self.eval_chain.invoke(
                {
                    "transcript": transcript,
                    "resume_context": resume_context or "未提供简历。",
                    "jd_context": jd_context or "未提供岗位要求。",
                }
            )
            parsed = parse_llm_json_response(llm_output, default=None)

            if isinstance(parsed, dict) and parsed.get("total_score"):
                logger.info(f"[AGENT] LLM 总评生成成功，total_score={parsed.get('total_score')}")
                return self._format_llm_evaluation(parsed)

            logger.warning("[AGENT] LLM 总评输出解析为空或无效，降级为规则模板兜底")
        except Exception as e:
            logger.warning(f"[AGENT] LLM 总评生成失败，降级为规则模板兜底: {e}")

        # 兜底：使用规则模板 + scoring_tool 的维度分数
        dim_scores: dict[str, int] = {}
        try:
            final_score_result = self.scoring_tool.score_answer(
                original_question="整场模拟面试总体评估",
                user_answer=transcript,
                resume_context="",
                jd_context="",
                rag_context="",
                use_rag=False,
            )
            if isinstance(final_score_result, dict):
                raw_dims = final_score_result.get("dimension_scores", {})
                if isinstance(raw_dims, dict):
                    dim_scores = {k: int(v or 0) for k, v in raw_dims.items()}
        except Exception as e:
            logger.warning(f"[AGENT] 兜底评分也失败: {e}")

        return self._build_rule_based_evaluation(history, dim_scores)

    def _generate_questions(
        self,
        resume_context: str,
        jd_context: str,
        rag_context: str,
    ) -> dict[str, Any]:
        """
        统一生成面试题。
        返回 question_tool 的原始结果，避免丢失 mode / meta / reference_answer 等信息。
        """
        questions_result = self.question_tool.generate_questions(
            resume_context=resume_context,
            jd_context=jd_context,
            rag_context=rag_context,
            use_rag=True,
        )
        return questions_result

    def _evaluate_current_answer(
        self,
        original_question: str,
        user_answer: str,
        resume_context: str,
        jd_context: str,
        rag_context: str,
        allow_followup: bool,
    ) -> tuple[str, str, bool]:
        """
        对当前回答进行后台追问。

        返回：
        - followup_question
        - 展示文本
        - 是否需要追问
        """
        followup_question = ""
        parts: list[str] = []
        if allow_followup and self.session_state.current_question_followup_count < 2:
            followup_question = self._create_followup_question(
                original_question=original_question,
                user_answer=user_answer,
                round_index=self.session_state.current_question_followup_count + 1,
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context=rag_context,
            )

        followup_question = self._clean_question_text(followup_question)

        self.session_state.last_followups = [{"question": followup_question}] if followup_question else []
        self.session_state.last_score = None
        self.session_state.evaluation_history.append(
            {
                "question": original_question,
                "answer": user_answer,
                "followups": [{"question": followup_question, "answer": ""}] if followup_question else [],
            }
        )

        need_followup = bool(followup_question)
        if need_followup:
            parts.append("【追问】")
            parts.append(followup_question)

        return followup_question, ReactAgent._to_markdown(parts), need_followup

    def _next_mock_question_text(self, questions: list[dict[str, Any]], index: int) -> str:
        """
        格式化当前题目输出。
        只展示题目本身，不展示题型、考察点、参考要点。
        """
        if not questions or index < 0 or index >= len(questions):
            return "当前没有可继续推进的面试题。"

        item = questions[index]
        question_text = self._clean_question_text(item.get("question", ""))
        return ReactAgent._to_markdown([
            f"【当前题目 {index + 1}/{len(questions)}】",
            question_text,
        ])

    def _advance_mock_interview(self, resume_context: str = "", jd_context: str = "") -> str:
        """
        将模拟面试推进到下一题。
        """
        questions = self.session_state.last_questions
        if not questions:
            self.session_state.mock_interview_started = False
            self.session_state.current_question = None
            self.session_state.current_question_index = -1
            return "当前没有可继续推进的面试题，模拟面试已结束。"

        next_index = self.session_state.current_question_index + 1
        if next_index >= len(questions):
            self.session_state.mock_interview_started = False
            self.session_state.current_question = None
            self.session_state.current_question_index = -1
            self.session_state.final_summary_ready = True
            self.session_state.current_mode = "idle"
            summary = self._summarize_final_evaluation(
                self.session_state.evaluation_history,
                resume_context=resume_context,
                jd_context=jd_context,
            )
            self._reset_mock_session()
            self.session_state.final_summary_ready = True
            return summary

        self.session_state.current_question_index = next_index
        self.session_state.current_question = questions[next_index]
        self.session_state.current_question_followup_count = 0
        return self._next_mock_question_text(questions, next_index)

    def execute(self, query: str, runtime_context: dict[str, Any] | None = None) -> str:
        """
        非流式执行入口。
        """
        ctx = runtime_context or {}

        self._update_session_state(query=query, runtime_context=ctx)

        has_file_context = self._has_effective_file_context(self.session_state)
        intent = self._classify_intent(query)
        resume_context, jd_context = self._format_resume_jd_context(self.session_state)
        rag_context = ""

        self.session_state.last_intent = intent
        logger.info(
            f"[AGENT] intent={intent} has_resume={self.session_state.has_resume} "
            f"has_jd={self.session_state.has_jd} query={query}"
        )

        # 后置修正：用户已上传文件 + 查询含建议类关键词 → 升级为 suggestion
        # 因为 LLM 可能把"给我面试准备建议"分成 qa（没有明确说"根据简历"），
        # 但用户上传了文件说明就是要个性化建议
        if intent == "qa" and has_file_context:
            suggestion_triggers = {"建议", "准备", "怎么准备", "如何准备", "复习", "方向"}
            if any(kw in query for kw in suggestion_triggers):
                intent = "suggestion"
                self.session_state.last_intent = intent
                logger.info(f"[AGENT] intent 修正: qa → suggestion（已上传文件 + 含建议关键词）")

        # 模拟面试进行中时，优先把所有输入当作对当前题目的回答处理，避免被重新分类打断流程。
        if self.session_state.mock_interview_started and self.session_state.current_question:
            self.session_state.current_mode = "mock_interview"

            original_question = str(self.session_state.current_question.get("question", "")).strip()
            if not original_question:
                self.session_state.last_answer = "当前没有可继续推进的面试题，模拟面试已结束。"
                self._reset_mock_session()
                return self.session_state.last_answer

            if not query.strip():
                self.session_state.last_answer = "请先回答当前题目。"
                return self.session_state.last_answer

            # 追问答案：先写回上一轮追问的答案，再判断是否继续追问或推进到下一题。
            if self.session_state.awaiting_followup_answer:
                current_index = self.session_state.current_question_index
                if 0 <= current_index < len(self.session_state.evaluation_history):
                    record = self.session_state.evaluation_history[current_index]
                    followups = record.setdefault("followups", [])
                    pending_round = max(1, self.session_state.pending_followup_round)
                    while len(followups) < pending_round:
                        followups.append({"question": "", "answer": ""})
                    followups[pending_round - 1]["answer"] = query

                self.session_state.current_answer = query
                self.session_state.awaiting_followup_answer = False
                self.session_state.pending_followup_round = 0

                can_continue_followup = (
                    self.session_state.current_question_index in self.session_state.followup_question_indices
                    and self.session_state.current_question_followup_count < 2
                )
                if can_continue_followup:
                    next_followup_question = self._create_followup_question(
                        original_question=original_question,
                        user_answer=query,
                        round_index=self.session_state.current_question_followup_count + 1,
                        resume_context=resume_context,
                        jd_context=jd_context,
                        rag_context=rag_context,
                    )
                    next_followup_question = self._clean_question_text(next_followup_question)
                    if next_followup_question:
                        if 0 <= current_index < len(self.session_state.evaluation_history):
                            record = self.session_state.evaluation_history[current_index]
                            record.setdefault("followups", []).append({"question": next_followup_question, "answer": ""})

                        self.session_state.current_question_followup_count += 1
                        self.session_state.awaiting_followup_answer = True
                        self.session_state.pending_followup_round = self.session_state.current_question_followup_count
                        self.session_state.last_followups = [{"question": next_followup_question}]
                        self.session_state.total_followup_count += 1

                        followup_text = self._rag_dict_to_text(
                            {
                                "notice": "基于你的回答，我再追问一个问题：",
                                "answer": next_followup_question,
                            }
                        )
                        self.session_state.last_answer = followup_text
                        return followup_text

                self.session_state.current_question_followup_count = 0
                next_question_text = self._advance_mock_interview(resume_context=resume_context, jd_context=jd_context)
                self.session_state.last_answer = next_question_text
                return next_question_text

            allow_followup = (
                self.session_state.current_question_index in self.session_state.followup_question_indices
                and self.session_state.current_question_followup_count < 2
            )

            followup_question, followup_text, need_followup = self._evaluate_current_answer(
                original_question=original_question,
                user_answer=query,
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context=rag_context,
                allow_followup=allow_followup,
            )
            self.session_state.current_answer = query
            self.session_state.answered_question_count += 1

            if need_followup:
                self.session_state.current_question_followup_count += 1
                self.session_state.awaiting_followup_answer = True
                self.session_state.pending_followup_round = self.session_state.current_question_followup_count
                self.session_state.total_followup_count += 1
                self.session_state.last_followups = [{"question": followup_question}] if followup_question else []
                self.session_state.last_answer = followup_text
                return followup_text

            self.session_state.awaiting_followup_answer = False
            self.session_state.pending_followup_round = 0
            next_question_text = self._advance_mock_interview(resume_context=resume_context, jd_context=jd_context)
            self.session_state.last_answer = next_question_text
            return next_question_text

        # 问候/感谢场景：LLM 根据用户输入生成自然回复，不依赖文件也不走 RAG
        if intent == "greeting":
            self.session_state.current_mode = "greeting"
            greeting_prompt = PromptTemplate.from_template(
                "你是一个友好的 AI 面试准备助手。根据用户的消息，用简洁自然的中文回复。\n"
                "- 如果是打招呼/问候：简短问候并介绍自己的能力（生成面试题、模拟面试追问评分、回答面试知识、给面试建议）\n"
                "- 如果是感谢：礼貌回应，鼓励用户继续提问\n"
                "- 如果是询问身份/功能：简要说明自己的四大功能\n"
                "保持 2-4 句话，语气亲切，不要过度啰嗦。\n\n"
                "用户消息：{query}\n回复："
            )
            greeting_chain = greeting_prompt | chat_model | StrOutputParser()
            greeting_answer = greeting_chain.invoke({"query": query}).strip()
            self.session_state.last_answer = greeting_answer
            return greeting_answer

        # 非目标场景统一处理
        if intent == "non_target":
            self.session_state.current_mode = "non_target"
            self.session_state.last_answer = self._non_target_message()
            return self.session_state.last_answer

        # 模拟面试 / 出题 依赖文件上下文；建议 无文件时走 RAG 兜底
        if intent in {"mock_interview", "question_generation"} and not has_file_context:
            self.session_state.current_mode = intent
            self.session_state.last_answer = self._need_file_message()
            return self.session_state.last_answer

        # 1) 模拟面试
        if intent == "mock_interview":
            self.session_state.current_mode = "mock_interview"

            # 模拟面试：使用轻量出题（只要题目，不要参考答案）
            questions_result = self.question_tool.generate_mock_questions(
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context="",
                desired_count=6,
            )
            questions = (questions_result.get("questions", []) or [])[:6]

            if len(questions) < 6:
                self.session_state.last_answer = "当前可生成的面试题不足 6 道，请补充资料后重试。"
                self._reset_mock_session()
                return self.session_state.last_answer

            self.session_state.last_questions = questions
            self.session_state.mock_interview_started = True
            self.session_state.current_question_index = 0
            self.session_state.current_question = questions[0] if questions else None
            self.session_state.current_question_followup_count = 0
            self.session_state.awaiting_followup_answer = False
            self.session_state.pending_followup_round = 0
            self.session_state.asked_question_count = len(questions)
            self.session_state.followup_question_indices = {int(i) for i in random.sample(range(6), 2)}

            if self.session_state.current_question:
                answer_text = self._next_mock_question_text(
                    self.session_state.last_questions,
                    self.session_state.current_question_index,
                )
            else:
                answer_text = "当前没有可生成的面试题。"

            header_parts = [
                "【模拟面试模式】",
                "",
                answer_text,
                "",
                "说明：请先回答当前题，我会根据你的回答决定是否追问或进入下一题。",
            ]

            self.session_state.last_answer = ReactAgent._to_markdown(header_parts)
            return self.session_state.last_answer

        # 2) 直接出题：返回面试题
        if intent == "question_generation":
            self.session_state.current_mode = "question_generation"

            questions_result = self._generate_questions(
                resume_context=resume_context,
                jd_context=jd_context,
                rag_context="",
            )
            notice = str(questions_result.get("notice", "")).strip()
            questions = (questions_result.get("questions", []) or [])[:6]
            mode = str(questions_result.get("mode", "")).strip()

            self.session_state.last_questions = questions

            answer_text = self._question_list_to_text(
                title="",
                notice="",
                questions=questions,
                show_meta=True,
                include_reference_answer=True,
            )


            self.session_state.last_answer = answer_text
            return answer_text

        # 3) 准备建议：先 RAG，再兜底
        if intent == "suggestion":
            self.session_state.current_mode = "suggestion"

            # 用户明确提到"文件/简历/JD"但没上传 → 直接提示，不走 RAG
            if not has_file_context:
                file_triggers = {"上传", "简历", "文件", "资料", "JD", "jd", "岗位描述"}
                if any(kw in query for kw in file_triggers):
                    self.session_state.last_answer = self._need_file_message()
                    return self.session_state.last_answer

            # 将简历/JD上下文拼入检索query，否则纯自然语言提问与
            # 知识库技术文档之间语义差距过大，导致向量检索命中率极低
            enriched_query_parts = [query]
            if resume_context.strip():
                enriched_query_parts.append(resume_context.strip())
            if jd_context.strip():
                enriched_query_parts.append(jd_context.strip())
            enriched_query = " ".join(enriched_query_parts)

            rag_result = self.rag_tool.answer(enriched_query, use_file_mode=has_file_context)
            rag_text = self._rag_dict_to_text(rag_result)

            if rag_text.strip():
                self.session_state.last_answer = rag_text
                return rag_text

            self.session_state.last_answer = self._need_file_message()
            return self.session_state.last_answer

        # 4) 普通问答
        if intent == "qa":
            self.session_state.current_mode = "qa"

            rag_result = self.rag_tool.answer(query, use_file_mode=has_file_context)
            answer_text = self._rag_dict_to_text(rag_result)
            self.session_state.last_answer = answer_text
            return answer_text

        self.session_state.current_mode = "idle"
        self.session_state.last_answer = self._non_target_message()
        return self.session_state.last_answer

    def execute_stream(self, query: str, runtime_context: dict[str, Any] | None = None):
        """
        流式输出入口：按字符逐个返回，实现“一个字一个字”的效果。
        """
        text = self.execute(query, runtime_context=runtime_context)

        for ch in text:
            yield ch


