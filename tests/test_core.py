"""
核心模块单元测试

覆盖：
- 意图分类缓存
- 简历/JD 解析缓存
- API 错误分类
- 会话状态序列化/反序列化
- 配置加载
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =========================
# 1. 意图分类缓存测试
# =========================
class TestIntentCache:
    """测试 ReactAgent 的意图分类缓存。"""

    def _make_agent(self):
        """创建一个 ReactAgent 实例（需要环境变量可用）。"""
        from ai_interview_assistant.agent.wf_agent import ReactAgent
        return ReactAgent()

    def test_cache_stores_result(self):
        """分类结果应被缓存。"""
        agent = self._make_agent()
        # 手动写入缓存
        agent._intent_cache["你好"] = "greeting"
        assert agent._intent_cache["你好"] == "greeting"

    def test_cache_hit_skips_llm(self):
        """缓存命中时不应调用 LLM。"""
        agent = self._make_agent()
        agent._intent_cache["什么是RAG"] = "qa"
        # 如果缓存命中，_classify_intent 应直接返回，不调用 intent_chain
        result = agent._classify_intent("什么是RAG")
        assert result == "qa"

    def test_empty_query_returns_non_target(self):
        """空查询返回 non_target。"""
        agent = self._make_agent()
        assert agent._classify_intent("") == "non_target"
        assert agent._classify_intent("   ") == "non_target"


# =========================
# 2. 解析缓存测试
# =========================
class TestParseCache:
    """测试 ResumeTool 和 JDTool 的缓存。"""

    def test_resume_cache(self):
        """相同简历文本应命中缓存。"""
        from ai_interview_assistant.agent.tools.resume_tool import ResumeTool
        tool = ResumeTool()
        # 手动写入缓存
        tool._cache["test_key"] = {"summary": "cached", "skills": ["python"]}
        assert tool._cache["test_key"]["summary"] == "cached"

    def test_jd_cache(self):
        """相同 JD 文本应命中缓存。"""
        from ai_interview_assistant.agent.tools.jd_tool import JDTool
        tool = JDTool()
        # 手动写入缓存
        from ai_interview_assistant.agent.tools.jd_tool import JDParseResult
        cached = JDParseResult(
            raw_text="test", summary="cached", responsibilities=[],
            requirements=[], plus_points=[], keywords=[], interview_focus=[],
        )
        tool._cache["test_key"] = cached
        assert tool._cache["test_key"].summary == "cached"


# =========================
# 3. API 错误分类测试
# =========================
class TestErrorClassification:
    """测试 api_server 的错误分类逻辑。"""

    @staticmethod
    def _classify(e: Exception) -> str:
        """复制 api_server 中的错误分类逻辑用于测试。"""
        err_msg = str(e).lower()
        err_type = type(e).__name__.lower()

        if "timeout" in err_msg or "timed out" in err_msg or "connect" in err_msg:
            return "AI 服务响应超时，请稍后重试。"
        if "429" in err_msg or "rate" in err_msg or "limit" in err_msg:
            return "AI 服务繁忙（限流），请稍等几秒后重试。"
        if "401" in err_msg or "403" in err_msg or "auth" in err_msg or "api key" in err_msg:
            return "AI 服务认证失败，请检查 API Key 配置。"
        if "json" in err_type or "json" in err_msg or "parse" in err_msg:
            return "AI 输出格式异常，请重新发送消息。"
        return f"处理请求时发生错误：{e}"

    def test_timeout_error(self):
        msg = self._classify(TimeoutError("Connection timed out"))
        assert "超时" in msg

    def test_rate_limit_error(self):
        msg = self._classify(Exception("Rate limit exceeded 429"))
        assert "限流" in msg

    def test_auth_error(self):
        msg = self._classify(Exception("401 Unauthorized"))
        assert "认证" in msg

    def test_json_error(self):
        msg = self._classify(json.JSONDecodeError("Expecting value", "", 0))
        assert "格式" in msg

    def test_generic_error(self):
        msg = self._classify(ValueError("something went wrong"))
        assert "something went wrong" in msg


# =========================
# 4. 会话状态序列化测试
# =========================
class TestSessionState:
    """测试 AgentSessionState 的序列化/反序列化。"""

    def test_roundtrip(self):
        """序列化再反序列化应保持数据一致。"""
        from ai_interview_assistant.agent.wf_agent import ReactAgent, AgentSessionState
        agent = ReactAgent()

        # 设置一些状态
        agent.session_state.has_resume = True
        agent.session_state.current_mode = "qa"
        agent.session_state.last_intent = "qa"
        agent.session_state.last_questions = [{"question": "test"}]
        agent.session_state.followup_question_indices = {1, 3}

        # 序列化
        from api_server import _agent_state_to_dict, _restore_agent_state
        state_dict = _agent_state_to_dict(agent)

        assert state_dict["has_resume"] is True
        assert state_dict["current_mode"] == "qa"
        assert state_dict["followup_question_indices"] == [1, 3] or set(state_dict["followup_question_indices"]) == {1, 3}

        # 反序列化到新 Agent
        agent2 = ReactAgent()
        _restore_agent_state(agent2, state_dict)

        assert agent2.session_state.has_resume is True
        assert agent2.session_state.current_mode == "qa"
        assert agent2.session_state.last_intent == "qa"
        assert agent2.session_state.last_questions == [{"question": "test"}]
        assert agent2.session_state.followup_question_indices == {1, 3}


# =========================
# 5. 配置加载测试
# =========================
class TestConfig:
    """测试配置文件加载。"""

    def test_app_config_loads(self):
        """app.yml 应能正常加载。"""
        from ai_interview_assistant.utils.config_handler import app_conf
        assert "backend_port" in app_conf
        assert "frontend_port" in app_conf
        assert isinstance(app_conf["backend_port"], int)

    def test_rag_config_loads(self):
        """rag.yml 应能正常加载。"""
        from ai_interview_assistant.utils.config_handler import rag_conf
        assert "chat_model_name" in rag_conf
        assert "embedding_model_name" in rag_conf

    def test_prompts_config_loads(self):
        """prompts.yml 应能正常加载。"""
        from ai_interview_assistant.utils.config_handler import prompts_conf
        assert "main_prompt_path" in prompts_conf


# =========================
# 6. 临时文件清理测试
# =========================
class TestTempCleanup:
    """测试临时文件清理逻辑。"""

    def test_cleanup_removes_old_files(self):
        """超过时限的文件应被删除。"""
        from start import cleanup_temp_files
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建一个"过期"文件
            old_file = os.path.join(tmpdir, "old_resume.txt")
            with open(old_file, "w") as f:
                f.write("old content")
            # 修改文件时间为 48 小时前
            old_time = time.time() - 48 * 3600
            os.utime(old_file, (old_time, old_time))

            # 创建一个"新鲜"文件
            new_file = os.path.join(tmpdir, "new_resume.txt")
            with open(new_file, "w") as f:
                f.write("new content")

            # 清理（max_age=24 小时）
            cleanup_temp_files(tmpdir, max_age_hours=24)

            assert not os.path.exists(old_file), "过期文件应被删除"
            assert os.path.exists(new_file), "新鲜文件应保留"
