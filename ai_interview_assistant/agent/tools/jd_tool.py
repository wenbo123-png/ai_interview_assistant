"""
JD 解析工具（增强版）

作用：
- 读取用户输入的岗位 JD / 面试主题（文本或文件）
- 使用 prompts/jd_parse_prompt.txt 做 LLM 结构化解析
- 在 LLM 失败时回退到规则抽取
- 为出题、追问、评分模块提供岗位上下文

说明：
当前版本优先使用 LLM + 提示词文件解析，必要时回退到规则抽取。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_interview_assistant.model.factory import chat_model
from ai_interview_assistant.utils.prompt_loader import load_jd_parse_prompt
from ai_interview_assistant.utils.file_handler import txt_loader, pdf_loader
from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.text_utils import clean_text, truncate_text


@dataclass
class JDParseResult:
    """JD 解析结果。"""
    raw_text: str
    summary: str
    responsibilities: list[str]
    requirements: list[str]
    plus_points: list[str]
    keywords: list[str]
    interview_focus: list[str]


class JDTool:
    """
    JD 解析工具。

    支持输入：
    - JD 文本
    - txt 文件路径
    - pdf 文件路径
    """

    def __init__(self, max_text_chars: int = 12000) -> None:
        self.max_text_chars = max_text_chars
        self.prompt_text = load_jd_parse_prompt()
        self.prompt = PromptTemplate.from_template(self.prompt_text)
        self.chain = self.prompt | chat_model | StrOutputParser()

        # 缓存：相同 JD 文本只解析一次
        self._cache: dict[str, JDParseResult] = {}

    @staticmethod
    def _is_file_path(text: str) -> bool:
        """简单判断输入是否为存在的 txt/pdf 文件路径。"""
        if not text:
            return False
        p = Path(text)
        return p.suffix.lower() in {".txt", ".pdf"} and p.exists()

    @staticmethod
    def _load_text_from_file(file_path: str) -> str:
        """从 txt/pdf 文件读取 JD 文本。"""
        lower_path = file_path.lower()
        if lower_path.endswith(".txt"):
            docs = txt_loader(file_path)
        elif lower_path.endswith(".pdf"):
            docs = pdf_loader(file_path)
        else:
            raise ValueError(f"不支持的 JD 文件类型: {file_path}")

        text = "\n".join(doc.page_content for doc in docs if doc.page_content)
        return text

    @staticmethod
    def _extract_section_text(text: str, section_markers: list[str]) -> list[str]:
        """
        按关键词行做规则抽取（MVP）。
        """
        if not text:
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        results: list[str] = []

        for line in lines:
            lower_line = line.lower()
            if any(marker in lower_line for marker in section_markers):
                results.append(line)

        return results

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """
        提取岗位关键词（规则版）。
        """
        if not text:
            return []

        candidate_keywords = [
            # AI / LLM / Agent / RAG
            "ai", "llm", "agent", "rag", "prompt", "transformer", "nlp", "embedding",
            # 常见工程栈
            "python", "java", "go", "sql", "mysql", "redis", "docker", "kubernetes",
            "linux", "git", "fastapi", "flask", "spring", "django",
            # 检索与数据
            "向量数据库", "chroma", "faiss", "milvus", "elasticsearch",
            # 系统能力
            "系统设计", "高并发", "微服务", "消息队列", "监控", "日志",
        ]

        lower_text = text.lower()
        found: list[str] = []
        for kw in candidate_keywords:
            if kw in lower_text and kw not in found:
                found.append(kw)

        return found

    @staticmethod
    def _build_interview_focus(
        responsibilities: list[str],
        requirements: list[str],
        keywords: list[str],
    ) -> list[str]:
        """
        生成面试重点方向（规则版）。
        """
        focus: list[str] = []

        # 基于职责抽取方向
        for line in responsibilities[:5]:
            if "架构" in line or "设计" in line:
                focus.append("系统设计与架构能力")
            if "模型" in line or "大模型" in line:
                focus.append("模型应用与落地能力")
            if "检索" in line or "rag" in line.lower():
                focus.append("RAG 检索与知识库能力")
            if "agent" in line.lower() or "智能体" in line:
                focus.append("Agent 任务编排与工具调用能力")

        # 基于关键词补充方向
        if any(k in keywords for k in ["python", "java", "go"]):
            focus.append("编程与工程实现能力")
        if any(k in keywords for k in ["docker", "kubernetes", "linux"]):
            focus.append("工程部署与运维基础能力")
        if any(k in keywords for k in ["mysql", "redis", "sql"]):
            focus.append("数据层设计与优化能力")

        # 去重保持顺序
        unique_focus: list[str] = []
        for f in focus:
            if f not in unique_focus:
                unique_focus.append(f)

        return unique_focus[:8]

    @staticmethod
    def _build_summary(
        responsibilities: list[str],
        requirements: list[str],
        plus_points: list[str],
        keywords: list[str],
    ) -> str:
        """构建 JD 摘要。"""
        parts: list[str] = []

        if responsibilities:
            parts.append(f"核心职责：{responsibilities[0]}")
        if requirements:
            parts.append(f"技能要求：{requirements[0]}")
        if plus_points:
            parts.append(f"加分项：{plus_points[0]}")
        if keywords:
            parts.append(f"关键词：{', '.join(keywords[:8])}")

        return "；".join(parts) if parts else "未能从 JD 中提取出明显结构化信息。"

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any] | None:
        """
        尝试从模型输出中解析 JSON。
        兼容：
        - 纯 JSON
        - ```json ... ```
        - ``` ... ```
        """
        if not text:
            return None

        raw = text.strip()

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        codeblock_patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]

        for pattern in codeblock_patterns:
            match = re.search(pattern, raw, re.S | re.I)
            if match:
                content = match.group(1).strip()
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue

        return None

    def _parse_with_llm(self, jd_text: str) -> JDParseResult | None:
        """
        先尝试用 prompt + LLM 解析 JD。
        如果输出不能解析为结构化结果，返回 None，交给规则抽取兜底。
        """
        try:
            # 与 prompt 文件变量名保持一致：建议 jd_parse_prompt.txt 使用 {jd_text}
            output = self.chain.invoke({"jd_text": jd_text})
            parsed = self._safe_parse_json(output)

            if not parsed:
                logger.warning("[JDTool] LLM 输出无法解析为 JSON，回退规则抽取")
                return None

            result = JDParseResult(
                raw_text=jd_text,
                summary=str(parsed.get("summary", "") or ""),
                responsibilities=list(parsed.get("responsibilities", []) or []),
                requirements=list(parsed.get("requirements", []) or []),
                plus_points=list(parsed.get("plus_points", []) or []),
                keywords=list(parsed.get("keywords", []) or []),
                interview_focus=list(parsed.get("interview_focus", []) or []),
            )

            return result
        except Exception as e:
            logger.warning(f"[JDTool] LLM 解析失败，回退规则抽取: {str(e)}")
            return None

    def parse_text(self, jd_text: str) -> JDParseResult:
        """
        解析 JD 文本。相同文本直接返回缓存结果。
        """
        cleaned = clean_text(jd_text)
        cleaned = truncate_text(cleaned, self.max_text_chars)

        # 缓存命中：相同 JD 文本直接返回
        cache_key = cleaned[:500]
        if cache_key in self._cache:
            logger.info("[JDTool] 缓存命中，跳过 LLM 调用")
            return self._cache[cache_key]

        llm_result = self._parse_with_llm(cleaned)
        if llm_result:
            logger.info("[JDTool] parse_text done by LLM")
            self._cache[cache_key] = llm_result
            return llm_result

        # fallback：规则抽取
        responsibilities = self._extract_section_text(
            cleaned, ["岗位职责", "职责", "responsibility", "工作内容", "你将负责"]
        )
        requirements = self._extract_section_text(
            cleaned, ["任职要求", "要求", "requirement", "资格", "我们希望你", "任职资格"]
        )
        plus_points = self._extract_section_text(
            cleaned, ["加分", "优先", "bonus", "plus", "优先考虑", "加分项"]
        )

        keywords = self._extract_keywords(cleaned)
        interview_focus = self._build_interview_focus(responsibilities, requirements, keywords)
        summary = self._build_summary(responsibilities, requirements, plus_points, keywords)

        result = JDParseResult(
            raw_text=cleaned,
            summary=summary,
            responsibilities=responsibilities,
            requirements=requirements,
            plus_points=plus_points,
            keywords=keywords,
            interview_focus=interview_focus,
        )

        logger.info(
            f"[JDTool] parse_text done by rules | "
            f"responsibilities={len(responsibilities)} requirements={len(requirements)} "
            f"plus_points={len(plus_points)} focus={len(interview_focus)}"
        )

        # 写入缓存
        self._cache[cache_key] = result
        return result

    def parse_file(self, file_path: str) -> JDParseResult:
        """
        解析 JD 文件。
        """
        if not self._is_file_path(file_path):
            # 若不是标准文件路径，按纯文本处理
            logger.info(f"[JDTool] 输入不是标准文件路径，改为按文本解析: {file_path[:50]}")
            return self.parse_text(file_path)

        text = self._load_text_from_file(file_path)
        return self.parse_text(text)

    def parse(self, input_data: str) -> dict[str, Any]:
        """
        统一入口：
        - 文件路径 -> parse_file
        - 纯文本 -> parse_text
        """
        if self._is_file_path(input_data):
            result = self.parse_file(input_data)
        else:
            result = self.parse_text(input_data)

        return asdict(result)


def parse_jd(input_data: str) -> dict[str, Any]:
    """
    便捷函数：直接调用 JD 解析。
    """
    tool = JDTool()
    return tool.parse(input_data)


if __name__ == "__main__":
    demo_jd = """
    岗位名称：AI Agent 应用开发工程师

    岗位职责：
    1. 负责 AI Agent 应用的架构设计与开发；
    2. 负责大模型调用、提示词优化、工具调用链路建设；
    3. 参与 RAG 检索与知识库系统优化。

    任职要求：
    1. 熟悉 Python，具备良好的工程能力；
    2. 熟悉大模型应用开发，了解 Prompt、RAG、Agent；
    3. 熟悉 MySQL、Redis、Docker、Linux。

    加分项：
    1. 有 LangChain / LangGraph 实战经验；
    2. 有线上 AI 应用稳定性优化经验。
    """

    result = parse_jd(demo_jd)
    print("=== JD解析结果 ===")
    for k, v in result.items():
        print(f"{k}: {v}")