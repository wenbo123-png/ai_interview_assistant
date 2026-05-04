"""
简历解析工具（prompt 驱动版）

作用：
- 读取用户上传的简历文本/文件
- 通过 prompts/resume_parse_prompt.txt 引导大模型结构化解析
- 提取教育背景、技能栈、项目经历、实习经历等信息
- 为后续面试题生成、追问生成、评分提供上下文

说明：
- 当前版本以 LLM 解析为主
- 保留少量规则处理作为兜底
- 解析结果统一输出为结构化字典，便于后续 Agent 调用
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_interview_assistant.model.factory import chat_model
from ai_interview_assistant.utils.file_handler import txt_loader, pdf_loader
from ai_interview_assistant.utils.json_utils import (
    parse_llm_json_response,
    ensure_dict,
    ensure_list,
    ensure_text,
)
from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.prompt_loader import load_resume_parse_prompt
from ai_interview_assistant.utils.text_utils import clean_text, truncate_text


@dataclass
class ResumeParseResult:
    """简历解析结果。"""
    raw_text: str
    summary: str
    education: list[str]
    skills: list[str]
    projects: list[str]
    internship: list[str]
    strengths: list[str]
    weaknesses: list[str]
    keywords: list[str]


class ResumeTool:
    """
    简历解析工具。

    输入可以是：
    - 纯文本简历
    - txt 文件路径
    - pdf 文件路径

    输出是结构化字典，便于后续 Agent 调用。
    """

    def __init__(self, max_text_chars: int = 12000) -> None:
        self.max_text_chars = max_text_chars

        # 从 prompts/ 目录读取提示词，而不是在代码里硬编码
        self.prompt_text = load_resume_parse_prompt()
        self.prompt = PromptTemplate.from_template(self.prompt_text)

        # 简历解析链：prompt -> model -> string
        self.chain = self.prompt | chat_model | StrOutputParser()

        # 缓存：相同简历文本只解析一次
        self._cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _is_file_path(text: str) -> bool:
        """简单判断是否像文件路径。"""
        if not text:
            return False
        p = Path(text)
        return p.suffix.lower() in {".txt", ".pdf"} and p.exists()

    @staticmethod
    def _load_text_from_file(file_path: str) -> str:
        """从 txt/pdf 文件中加载简历文本。"""
        lower_path = file_path.lower()
        if lower_path.endswith(".txt"):
            docs = txt_loader(file_path)
        elif lower_path.endswith(".pdf"):
            docs = pdf_loader(file_path)
        else:
            raise ValueError(f"不支持的简历文件类型: {file_path}")

        # 将文档内容拼接为纯文本
        text = "\n".join(doc.page_content for doc in docs if doc.page_content)
        return text

    @staticmethod
    def _fallback_extract_keywords(text: str) -> list[str]:
        """
        规则兜底：从简历中抽取一些高频技术关键词。
        当模型输出不完整时，用于补足 keywords。
        """
        if not text:
            return []

        candidate_keywords = [
            "python", "java", "go", "c++", "sql", "mysql", "postgresql", "redis",
            "docker", "kubernetes", "k8s", "linux", "git", "shell", "nginx",
            "pytorch", "tensorflow", "paddle", "llm", "rag", "agent", "prompt",
            "nlp", "cv", "transformer", "spring", "spring boot", "flask", "fastapi",
            "django", "vue", "react", "vue3", "mongodb", "elasticsearch",
        ]

        lower_text = text.lower()
        found: list[str] = []
        for kw in candidate_keywords:
            if kw in lower_text and kw not in found:
                found.append(kw)

        return found

    @staticmethod
    def _fallback_build_summary(
        education: list[str],
        skills: list[str],
        projects: list[str],
        internship: list[str],
        strengths: list[str],
    ) -> str:
        """规则兜底生成简历摘要。"""
        parts: list[str] = []

        if education:
            parts.append(f"教育背景：{education[0]}")
        if skills:
            parts.append(f"技能栈：{', '.join(skills[:8])}")
        if projects:
            parts.append(f"项目经历：{projects[0]}")
        if internship:
            parts.append(f"实习经历：{internship[0]}")
        if strengths:
            parts.append(f"优势：{', '.join(strengths[:5])}")

        return "；".join(parts) if parts else "未能从简历中提取出明显结构化信息。"

    @staticmethod
    def _normalize_list(value: Any) -> list[str]:
        """把任意输入尽量规范为字符串列表。"""
        items = ensure_list(value, default=[])
        result: list[str] = []
        for item in items:
            text = ensure_text(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _parse_by_llm(self, resume_text: str) -> dict[str, Any]:
        """
        通过大模型解析简历文本。
        返回原始 JSON 结构。
        """
        prompt_input = {"resume_text": resume_text}
        text = self.chain.invoke(prompt_input)
        parsed = parse_llm_json_response(text, default={})
        return ensure_dict(parsed, default={})

    def parse_text(self, resume_text: str) -> ResumeParseResult:
        """
        解析简历文本。相同文本直接返回缓存结果。
        """
        cleaned = clean_text(resume_text)
        cleaned = truncate_text(cleaned, self.max_text_chars)

        # 缓存命中：相同简历文本直接返回
        cache_key = cleaned[:500]  # 取前 500 字符作为缓存 key（避免过长）
        if cache_key in self._cache:
            logger.info("[ResumeTool] 缓存命中，跳过 LLM 调用")
            cached = self._cache[cache_key]
            return ResumeParseResult(**cached)

        # 先走模型解析
        model_data = {}
        try:
            model_data = self._parse_by_llm(cleaned)
        except Exception as e:
            logger.warning(f"[ResumeTool] LLM 解析失败，启用兜底规则抽取: {e}")

        # 从模型结果中取字段
        raw_text = ensure_text(model_data.get("raw_text", cleaned))
        summary = ensure_text(model_data.get("summary", "")).strip()
        education = self._normalize_list(model_data.get("education", []))
        skills = self._normalize_list(model_data.get("skills", []))
        projects = self._normalize_list(model_data.get("projects", []))
        internship = self._normalize_list(model_data.get("internship", []))
        strengths = self._normalize_list(model_data.get("strengths", []))
        weaknesses = self._normalize_list(model_data.get("weaknesses", []))
        keywords = self._normalize_list(model_data.get("keywords", []))

        # 兜底补齐：如果模型没提取出来，就用规则补
        if not keywords:
            keywords = self._fallback_extract_keywords(cleaned)

        if not summary:
            summary = self._fallback_build_summary(
                education=education,
                skills=skills,
                projects=projects,
                internship=internship,
                strengths=strengths,
            )

        result = ResumeParseResult(
            raw_text=raw_text or cleaned,
            summary=summary,
            education=education,
            skills=skills,
            projects=projects,
            internship=internship,
            strengths=strengths,
            weaknesses=weaknesses,
            keywords=keywords,
        )

        logger.info(
            f"[ResumeTool] parse_text done | "
            f"education={len(education)} skills={len(skills)} "
            f"projects={len(projects)} internship={len(internship)} "
            f"keywords={len(keywords)}"
        )

        # 写入缓存
        self._cache[cache_key] = asdict(result)
        return result

    def parse_file(self, file_path: str) -> ResumeParseResult:
        """
        解析简历文件。
        """
        if not self._is_file_path(file_path):
            # 这里允许用户直接传入 txt/pdf 路径；否则当成文本处理
            logger.info(f"[ResumeTool] 输入不是标准文件路径，改为按文本解析: {file_path[:50]}")
            return self.parse_text(file_path)

        text = self._load_text_from_file(file_path)
        return self.parse_text(text)

    def parse(self, input_data: str) -> dict[str, Any]:
        """
        统一入口：
        - 如果是文件路径，则按文件解析
        - 否则按文本解析
        """
        if self._is_file_path(input_data):
            result = self.parse_file(input_data)
        else:
            result = self.parse_text(input_data)

        return asdict(result)


def parse_resume(input_data: str) -> dict[str, Any]:
    """
    便捷函数：直接调用简历解析工具。
    """
    tool = ResumeTool()
    return tool.parse(input_data)


if __name__ == "__main__":
    demo_resume = """
    张三
    教育背景：XX大学 本科 软件工程
    技能：Python、Java、Linux、MySQL、Redis、FastAPI、RAG、Agent
    项目经历：基于RAG的智能问答系统；AI面试准备助手
    实习经历：某互联网公司后端实习
    自我评价：学习能力强，沟通能力较好，具备一定工程实践经验
    """

    result = parse_resume(demo_resume)
    print("=== 简历解析结果 ===")
    for k, v in result.items():
        print(f"{k}: {v}")