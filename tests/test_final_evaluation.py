"""
模拟面试最终总评测试

覆盖：
- LLM 正常路径：个性化评语格式化与内容检查
- LLM 失败兜底：降级到规则模板
- LLM 返回无效 JSON 兜底
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_interview_assistant.agent.wf_agent import ReactAgent


# ---------- 共享测试数据 ----------

MOCK_HISTORY: list[dict[str, Any]] = [
    {
        "question": "请介绍一下你对 RAG（检索增强生成）的理解及其核心流程。",
        "answer": "RAG 是将信息检索与大语言模型生成相结合的技术框架。核心流程包括索引构建、查询检索、上下文增强生成三个阶段。我在实际项目中基于 LangChain 实现了 RAG 管道，使用 FAISS 作为向量库，配合 BGE 嵌入模型做 dense retrieval。",
        "followups": [
            {
                "question": "你在项目中如何评估检索召回的质量？",
                "answer": "主要使用命中率（Hit Rate）和 MRR 两个指标。我们构建了 200 条标注查询，检索 Top-K 取 5 时命中率达到 87%，MRR 为 0.72。",
            },
        ],
    },
    {
        "question": "请解释一下向量数据库的工作原理，以及它与传统数据库的区别。",
        "answer": "向量数据库的核心是将非结构化数据通过嵌入模型映射到高维向量空间，用 ANN 算法做近似最近邻检索。与传统数据库主要区别在于：传统库基于精确匹配，向量库基于语义相似度做模糊匹配。",
        "followups": [],
    },
    {
        "question": "假设你要设计一个支持百万级用户的面试准备平台，你会如何做系统架构？",
        "answer": "接入层用 Nginx 做负载均衡和限流。应用层采用微服务架构，将会话管理、题目生成、评分计算拆分为独立服务，通过 Kafka 异步解耦。数据层 MySQL 做用户和题库管理，Redis 做会话缓存，Elasticsearch 做题目全文检索，向量检索单独部署 Milvus 集群。",
        "followups": [
            {
                "question": "追问1：如果遇到用户量突增，你如何保证系统不被击穿？",
                "answer": "核心思路是限流降级+弹性扩容。接入层用令牌桶做限流，对于 LLM 调用这种高成本操作提前预热缓存到 Redis，同时基于 K8s HPA 根据 CPU/内存指标自动水平扩容。",
            },
            {
                "question": "追问2：请进一步展开你在高并发场景下的具体经验。",
                "answer": "",
            },
        ],
    },
]

MOCK_RESUME = "技能：Python、RAG、Agent、LangChain；项目：AI面试准备助手"
MOCK_JD = "岗位：AI Agent 应用开发工程师；要求：熟悉RAG、Prompt Engineering、工具调用、系统设计"

STRUCTURE_CHECKS = [
    "【最终总评】",
    "一、整体总评",
    "二、核心维度评估",
    "三、核心短板",
    "四、提升建议",
    "五、综合评定",
    "评级：",
    "模拟面试通关预判：",
]

# LLM 正常返回的模拟 JSON
MOCK_LLM_RESPONSE = json.dumps({
    "total_score": 82,
    "rating": "良好",
    "prediction": "较有希望通过",
    "overall_comment": (
        "你在专业知识和内容质量方面表现较好，能够结合实际项目经验展开回答，体现了一定的技术深度。"
        "但在系统设计和岗位匹配方面仍有提升空间，整体处于中等偏上水平。"
    ),
    "dimensions": {
        "professionalism": {
            "score": 85,
            "comment": "第 1 题提到了 FAISS、BGE 等具体技术栈，对 RAG 核心流程掌握较好。第 2 题对向量数据库与传统数据库的区别理解正确但偏概念化。"
        },
        "logic": {
            "score": 78,
            "comment": "第 1 题回答层次清晰（三阶段拆解），第 3 题的架构设计从接入层到数据层逐层展开。但追问环节对限流降级和弹性扩容的展开略显混杂。"
        },
        "content_quality": {
            "score": 82,
            "comment": "第 1 题提到了 LangChain、FAISS、BGE 等具体技术栈，追问中给出了 Hit Rate 87%、MRR 0.72 的量化指标，信息密度较高。"
        },
        "job_fit": {
            "score": 80,
            "comment": "RAG 和向量检索能力与岗位要求匹配度较高，但系统设计方面未主动呼应 JD 中对 Agent 工具调用能力的要求。"
        }
    },
    "strengths": [
        "第 1 题对 RAG 流程的拆解完整，且能结合 LangChain + FAISS + BGE 的实际项目经验展开",
        "追问环节能给出 Hit Rate 87%、MRR 0.72 等量化指标，体现了实际数据意识",
    ],
    "weaknesses": [
        "第 3 题系统设计回答偏架构图式罗列，缺少对具体容灾方案的深入分析",
        "追问2 未能补充高并发场景的具体经验，暴露了实际项目经验的深度不足",
    ],
    "suggestions": [
        "准备 1-2 个高并发场景的完整案例，包含限流策略、缓存预热、扩容方案的具体参数",
        "回答向量数据库问题时，补充 Milvus vs FAISS 的实际性能对比和选型决策依据",
    ]
}, ensure_ascii=False)


# ---------- 辅助函数 ----------

def _check_structure(result: str) -> list[str]:
    """返回缺失的结构字段列表，空列表 = 全部通过。"""
    return [c for c in STRUCTURE_CHECKS if c not in result]


def _check_markdown_hard_breaks(result: str) -> list[str]:
    """返回格式有问题的行描述列表，空列表 = 全部通过。"""
    errors: list[str] = []
    for i, line in enumerate(result.split("\n"), 1):
        if line == "" or line.startswith("```"):
            continue
        if not line.endswith("  "):
            errors.append(f"L{i}: {line[:60]!r}")
    return errors


def _count_generic_phrases(result: str) -> int:
    """统计模板套话命中数。"""
    generic_phrases = [
        "扎实稳健",
        "平稳稳健",
        "知识储备和表达基础",
        "系统性构建和知识边界的深度理解",
        "整体答题的逻辑结构还可以更加清晰聚焦",
    ]
    return sum(1 for p in generic_phrases if p in result)


def _make_agent() -> ReactAgent:
    return ReactAgent()


# ---------- 测试用例 ----------

def _mock_eval_chain(return_value: str = "", side_effect: Exception | None = None) -> MagicMock:
    """创建一个 mock 的 eval_chain，替换 RunnableSequence 以绕过 Pydantic frozen 限制。"""
    mock = MagicMock()
    if side_effect is not None:
        mock.invoke.side_effect = side_effect
    else:
        mock.invoke.return_value = return_value
    return mock


def _mock_score_answer(scores: dict[str, int] | None = None) -> MagicMock:
    """创建一个 mock 的 scoring_tool，返回指定维度分数。"""
    default_scores = {
        "professionalism": 75, "logic": 70,
        "completeness": 73, "job_fit": 68,
    }
    mock = MagicMock()
    mock.score_answer.return_value = {
        "total_score": 72,
        "dimension_scores": scores or default_scores,
    }
    return mock


class TestLLMEvaluation:
    """LLM 正常路径：个性化评语。"""

    def test_llm_evaluation_structure(self):
        """LLM 返回有效 JSON 时，输出应包含完整五段式结构。"""
        agent = _make_agent()
        agent.eval_chain = _mock_eval_chain(return_value=MOCK_LLM_RESPONSE)

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        missing = _check_structure(result)
        assert not missing, f"缺少结构字段: {missing}"

    def test_llm_evaluation_markdown_format(self):
        """LLM 路径输出的非空行应以两个空格结尾（markdown 硬换行）。"""
        agent = _make_agent()
        agent.eval_chain = _mock_eval_chain(return_value=MOCK_LLM_RESPONSE)

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        errors = _check_markdown_hard_breaks(result)
        assert not errors, f"硬换行格式问题: {errors}"

    def test_llm_evaluation_personalization(self):
        """LLM 路径评语应引用具体内容，不应全是模板套话。"""
        agent = _make_agent()
        agent.eval_chain = _mock_eval_chain(return_value=MOCK_LLM_RESPONSE)

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        generic_count = _count_generic_phrases(result)
        assert generic_count < 3, f"命中 {generic_count} 条模板套话，评语可能未个性化"

    def test_llm_evaluation_contains_dimension_scores(self):
        """LLM 路径输出应包含各维度分数。"""
        agent = _make_agent()
        agent.eval_chain = _mock_eval_chain(return_value=MOCK_LLM_RESPONSE)

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        assert "85分" in result or "专业知识" in result
        assert "78分" in result or "答题逻辑" in result


class TestFallbackEvaluation:
    """LLM 失败时的兜底路径。"""

    def test_llm_exception_falls_back_to_rules(self):
        """LLM 调用抛异常时，应降级到规则模板并输出完整结构。"""
        agent = _make_agent()
        agent.eval_chain = _mock_eval_chain(side_effect=RuntimeError("模拟 LLM 失败"))
        agent.scoring_tool = _mock_score_answer()

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        missing = _check_structure(result)
        assert not missing, f"兜底路径缺少结构字段: {missing}"

    def test_llm_bad_json_falls_back_to_rules(self):
        """LLM 返回非 JSON 文本时，应降级到规则模板。"""
        agent = _make_agent()
        agent.eval_chain = _mock_eval_chain(return_value="抱歉，我无法生成有效的评估结果。")
        agent.scoring_tool = _mock_score_answer()

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        missing = _check_structure(result)
        assert not missing, f"兜底路径缺少结构字段: {missing}"

    def test_fallback_markdown_format(self):
        """兜底路径输出的非空行也应以两个空格结尾。"""
        agent = _make_agent()
        agent.eval_chain = _mock_eval_chain(side_effect=RuntimeError("模拟失败"))
        agent.scoring_tool = _mock_score_answer()

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        errors = _check_markdown_hard_breaks(result)
        assert not errors, f"兜底路径硬换行格式问题: {errors}"


class TestEdgeCases:
    """边界场景。"""

    def test_empty_history(self):
        """空面试历史应返回提示文本。"""
        agent = _make_agent()
        result = agent._summarize_final_evaluation([])
        assert "没有可汇总" in result

    def test_llm_returns_partial_json(self):
        """LLM 返回缺少字段的 JSON 时，应能格式化而不报错。"""
        agent = _make_agent()
        partial = json.dumps({
            "total_score": 75,
            "rating": "中等",
            "overall_comment": "整体表现一般。",
        }, ensure_ascii=False)
        agent.eval_chain = _mock_eval_chain(return_value=partial)

        result = agent._summarize_final_evaluation(
            MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
        )

        assert "【最终总评】" in result
        assert "整体表现一般" in result


if __name__ == "__main__":
    """手动运行：打印 LLM 个性化总评 和 规则模板兜底总评，方便直观对比效果。"""

    def _print_section(title: str, content: str) -> None:
        print()
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
        print(content)

    # ---- 1. LLM 个性化总评 ----
    agent1 = _make_agent()
    agent1.eval_chain = _mock_eval_chain(return_value=MOCK_LLM_RESPONSE)
    llm_result = agent1._summarize_final_evaluation(
        MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
    )
    _print_section("LLM 个性化总评", llm_result)

    # ---- 2. 规则模板兜底总评（LLM 失败） ----
    agent2 = _make_agent()
    agent2.eval_chain = _mock_eval_chain(side_effect=RuntimeError("模拟 LLM 失败"))
    agent2.scoring_tool = _mock_score_answer()
    fallback_result = agent2._summarize_final_evaluation(
        MOCK_HISTORY, resume_context=MOCK_RESUME, jd_context=MOCK_JD,
    )
    _print_section("规则模板兜底总评（LLM 失败时）", fallback_result)
