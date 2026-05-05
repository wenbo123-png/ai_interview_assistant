"""
FastAPI 后端服务

将原本耦合在 Streamlit 前端中的业务逻辑（Agent 调用、会话管理、文件上传）
抽离为独立的 REST API，供前端通过 HTTP 调用。

启动方式：
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_interview_assistant.agent.wf_agent import ReactAgent
from ai_interview_assistant.storage import (
    generate_session_id,
    save_session,
    load_session,
    delete_session,
    list_sessions,
)
from ai_interview_assistant.utils.config_handler import app_conf

# =========================
# FastAPI 实例
# =========================
app = FastAPI(title="AI 面试准备助手 API", version="1.0.0")

# 允许跨域（Streamlit 前端默认跑在不同端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 常量 & 本地缓存
# =========================
UPLOAD_DIR = Path(app_conf.get("temp_upload_dir", "tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 内存中缓存活跃的 Agent 实例，避免每次请求都重建
# key = session_id, value = ReactAgent
_agent_cache: dict[str, ReactAgent] = {}


def _get_or_create_agent(session_id: str) -> ReactAgent:
    """从缓存获取 Agent，不存在则新建并尝试从数据库恢复状态。"""
    if session_id not in _agent_cache:
        agent = ReactAgent()
        data = load_session(session_id)
        if data:
            _restore_agent_state(agent, data.get("agent_state", {}))
        _agent_cache[session_id] = agent
    return _agent_cache[session_id]


# =========================
# Agent 状态序列化 / 反序列化
# （与 app.py 中的逻辑一致，抽到后端独立维护）
# =========================
def _agent_state_to_dict(agent: ReactAgent) -> dict[str, Any]:
    """将 ReactAgent 的会话状态序列化为 dict。"""
    s = agent.session_state
    return {
        "has_resume": s.has_resume,
        "has_jd": s.has_jd,
        "current_mode": s.current_mode,
        "last_intent": s.last_intent,
        "resume_data": s.resume_data,
        "jd_data": s.jd_data,
        "last_questions": s.last_questions,
        "mock_interview_started": s.mock_interview_started,
        "current_question_index": s.current_question_index,
        "current_question": s.current_question,
        "awaiting_followup_answer": s.awaiting_followup_answer,
        "pending_followup_round": s.pending_followup_round,
        "current_question_followup_count": s.current_question_followup_count,
        "evaluation_history": s.evaluation_history,
        "total_followup_count": s.total_followup_count,
        "asked_question_count": s.asked_question_count,
        "answered_question_count": s.answered_question_count,
        "followup_question_indices": list(s.followup_question_indices),
        "final_summary_ready": s.final_summary_ready,
    }


def _restore_agent_state(agent: ReactAgent, state_dict: dict[str, Any]) -> None:
    """从 dict 恢复 ReactAgent 的会话状态。"""
    s = agent.session_state
    s.has_resume = state_dict.get("has_resume", False)
    s.has_jd = state_dict.get("has_jd", False)
    s.current_mode = state_dict.get("current_mode", "idle")
    s.last_intent = state_dict.get("last_intent", "")
    s.resume_data = state_dict.get("resume_data")
    s.jd_data = state_dict.get("jd_data")
    s.last_questions = state_dict.get("last_questions", [])
    s.mock_interview_started = state_dict.get("mock_interview_started", False)
    s.current_question_index = state_dict.get("current_question_index", -1)
    s.current_question = state_dict.get("current_question")
    s.awaiting_followup_answer = state_dict.get("awaiting_followup_answer", False)
    s.pending_followup_round = state_dict.get("pending_followup_round", 0)
    s.current_question_followup_count = state_dict.get("current_question_followup_count", 0)
    s.evaluation_history = state_dict.get("evaluation_history", [])
    s.total_followup_count = state_dict.get("total_followup_count", 0)
    s.asked_question_count = state_dict.get("asked_question_count", 0)
    s.answered_question_count = state_dict.get("answered_question_count", 0)
    s.followup_question_indices = set(state_dict.get("followup_question_indices", []))
    s.final_summary_ready = state_dict.get("final_summary_ready", False)


def _save_session_with_agent(session_id: str, chat_history: list[dict], agent: ReactAgent) -> None:
    """将 Agent 状态和聊天历史一起持久化到 SQLite。"""
    save_session(
        session_id=session_id,
        chat_history=chat_history,
        agent_state=_agent_state_to_dict(agent),
    )


# =========================
# 请求体模型
# =========================
class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str
    query: str
    runtime_context: dict[str, Any] | None = None
    chat_history: list[dict[str, Any]] = []


class SessionCreateRequest(BaseModel):
    """创建新会话请求（可选自定义 session_id）"""
    session_id: str | None = None


# =========================
# API 路由
# =========================

# ---------- 对话 ----------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    流式对话接口（SSE）。

    前端通过 EventSource 或 fetch + ReadableStream 接收逐字输出。
    后端调用 ReactAgent.execute_stream() 逐字符 yield。
    """
    agent = _get_or_create_agent(req.session_id)

    def _classify_error(e: Exception) -> str:
        """将异常转为用户友好的错误提示。"""
        err_msg = str(e).lower()
        err_type = type(e).__name__.lower()

        # LLM API 超时 / 网络错误
        if "timeout" in err_msg or "timed out" in err_msg or "connect" in err_msg:
            return "AI 服务响应超时，请稍后重试。"
        # API 限流
        if "429" in err_msg or "rate" in err_msg or "limit" in err_msg:
            return "AI 服务繁忙（限流），请稍等几秒后重试。"
        # API 密钥 / 认证
        if "401" in err_msg or "403" in err_msg or "auth" in err_msg or "api key" in err_msg:
            return "AI 服务认证失败，请检查 API Key 配置。"
        # JSON 解析失败（LLM 输出格式异常）
        if "json" in err_type or "json" in err_msg or "parse" in err_msg:
            return "AI 输出格式异常，请重新发送消息。"
        # 通用兜底
        return f"处理请求时发生错误：{e}"

    def event_generator():
        full_text = ""
        try:
            for ch in agent.execute_stream(req.query, runtime_context=req.runtime_context):
                full_text += ch
                yield f"data: {json.dumps({'token': ch}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'done': True, 'full_text': full_text}, ensure_ascii=False)}\n\n"

            chat_history = req.chat_history + [
                {"role": "user", "content": req.query},
                {"role": "assistant", "content": full_text},
            ]
            _save_session_with_agent(req.session_id, chat_history, agent)

        except Exception as e:
            friendly_msg = _classify_error(e)
            yield f"data: {json.dumps({'error': friendly_msg}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ---------- 会话管理 ----------

@app.get("/api/sessions")
async def api_list_sessions(limit: int = 20):
    """列出历史会话，按更新时间倒序。"""
    return {"sessions": list_sessions(limit=limit)}


@app.post("/api/sessions")
async def api_create_session(req: SessionCreateRequest | None = None):
    """创建新会话，返回 session_id。"""
    sid = (req.session_id if req and req.session_id else None) or generate_session_id()
    agent = ReactAgent()
    _agent_cache[sid] = agent
    _save_session_with_agent(sid, [], agent)
    return {"session_id": sid}


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str):
    """加载指定会话的聊天历史和 Agent 状态。"""
    data = load_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "session_id": session_id,
        "chat_history": data["chat_history"],
        "agent_state": data["agent_state"],
    }


@app.put("/api/sessions/{session_id}")
async def api_update_session(session_id: str, chat_history: list[dict[str, Any]]):
    """手动更新会话（如前端同步聊天历史）。"""
    agent = _get_or_create_agent(session_id)
    _save_session_with_agent(session_id, chat_history, agent)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """删除指定会话。"""
    _agent_cache.pop(session_id, None)
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="删除失败")
    return {"ok": True}


# ---------- 文件上传 ----------

@app.post("/api/sessions/{session_id}/upload")
async def api_upload_file(session_id: str, file: UploadFile = File(...)):
    """
    上传简历或 JD 文件，保存到本地并返回文件路径。
    前端拿到路径后放入 runtime_context 传给 /api/chat。
    """
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {"file_path": str(file_path), "filename": file.filename}


# ---------- 健康检查 ----------

@app.get("/api/health")
async def health():
    """健康检查接口。"""
    return {"status": "ok"}


# =========================
# 直接运行：python api_server.py
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=app_conf.get("backend_host", "0.0.0.0"),
        port=app_conf.get("backend_port", 8000),
        reload=True,
    )
