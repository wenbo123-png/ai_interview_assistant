"""
Streamlit 前端 —— 纯展示层

所有业务逻辑（Agent 调用、会话存储、文件解析）都通过 HTTP 调用
FastAPI 后端 (api_server.py) 完成，前端只负责：
- 渲染 UI
- 接收用户输入
- 调用后端 API
- 展示返回结果

启动方式（需要先后端再前端）：
    1. uvicorn api_server:app --host 0.0.0.0 --port 8000
    2. streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import streamlit as st

from ai_interview_assistant.utils.config_handler import app_conf

# =========================
# 后端 API 地址（从配置读取）
# =========================
_backend_host = app_conf.get("backend_host", "127.0.0.1")
_backend_port = app_conf.get("backend_port", 8000)
API_BASE = f"http://{_backend_host}:{_backend_port}"

# 超时设置：对话接口可能耗时较长（LLM 调用），设为 5 分钟
API_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


# =========================
# 页面基础配置
# =========================
st.set_page_config(
    page_title="AI 面试准备助手",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 AI 面试准备助手")
st.caption("支持简历/JD上传、面试题生成、模拟面试、追问评分与知识问答")
st.divider()

st.markdown(
    """
    <style>
    /* 强制显示侧边栏，不允许折叠 */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* 让左侧栏始终可见，保持固定宽度 */
    section[data-testid="stSidebar"] {
        min-width: 22rem;
        max-width: 22rem;
        width: 22rem;
    }

    /* 主内容区给左侧栏预留空间 */
    .main .block-container {
        margin-left: 18rem;
        padding-bottom: 6rem;
    }

    section.main {
        padding-bottom: 6rem;
    }

    /* 固定底部输入框，左边界直接按主内容区对齐 */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0.8rem;
        left: 24rem;
        right: 31rem;
        z-index: 9999;
        background: white;
        padding: 0.5rem 1rem;
        border-top: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 后端连接检测
# =========================
def wait_for_backend():
    """等待后端就绪，最多重试 15 秒。"""
    import time
    for i in range(30):
        try:
            resp = httpx.get(f"{API_BASE}/api/health", timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# 启动时先检测后端连接
if "backend_ready" not in st.session_state:
    with st.spinner("正在连接后端服务 ..."):
        if not wait_for_backend():
            st.error(
                "无法连接后端服务，请确认已启动：\n"
                "```\nuvicorn api_server:app --host 127.0.0.1 --port 8000\n```"
            )
            st.stop()
    st.session_state.backend_ready = True


# =========================
# 后端 API 封装
# =========================
def api_list_sessions(limit: int = 20) -> list[dict]:
    """调用后端获取历史会话列表。"""
    try:
        resp = httpx.get(f"{API_BASE}/api/sessions", params={"limit": limit}, timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("sessions", [])
    except Exception as e:
        st.error(f"获取会话列表失败：{e}")
        return []


def api_create_session() -> str:
    """调用后端创建新会话，返回 session_id。"""
    try:
        resp = httpx.post(f"{API_BASE}/api/sessions", json={}, timeout=10.0)
        resp.raise_for_status()
        return resp.json()["session_id"]
    except Exception as e:
        st.error(f"创建会话失败：{e}")
        return ""


def api_load_session(session_id: str) -> dict | None:
    """调用后端加载指定会话。"""
    try:
        resp = httpx.get(f"{API_BASE}/api/sessions/{session_id}", timeout=10.0)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"加载会话失败：{e}")
        return None


def api_save_session(session_id: str, chat_history: list[dict]) -> None:
    """调用后端保存会话（同步聊天历史）。"""
    try:
        httpx.put(
            f"{API_BASE}/api/sessions/{session_id}",
            json=chat_history,
            timeout=10.0,
        )
    except Exception:
        pass  # 保存失败不阻塞用户操作


def api_delete_session(session_id: str) -> bool:
    """调用后端删除会话。"""
    try:
        resp = httpx.delete(f"{API_BASE}/api/sessions/{session_id}", timeout=10.0)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"删除会话失败：{e}")
        return False


def api_upload_file(session_id: str, file_name: str, file_bytes: bytes) -> str:
    """调用后端上传文件，返回服务端文件路径。"""
    try:
        resp = httpx.post(
            f"{API_BASE}/api/sessions/{session_id}/upload",
            files={"file": (file_name, file_bytes)},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["file_path"]
    except Exception as e:
        st.error(f"上传文件失败：{e}")
        return ""


def api_chat_stream(session_id: str, query: str, runtime_context: dict, chat_history: list[dict]):
    """
    调用后端流式对话接口（SSE）。
    逐字符 yield，供前端实时渲染。
    """
    with httpx.stream(
        "POST",
        f"{API_BASE}/api/chat",
        json={
            "session_id": session_id,
            "query": query,
            "runtime_context": runtime_context,
            "chat_history": chat_history,
        },
        timeout=API_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])  # 去掉 "data: " 前缀
            if "error" in payload:
                raise RuntimeError(payload["error"])
            if payload.get("done"):
                return payload.get("full_text", "")
            if "token" in payload:
                yield payload["token"]


# =========================
# 工具函数
# =========================
def build_runtime_context(
    has_resume: bool,
    has_jd: bool,
    resume_input: str,
    jd_input: str,
) -> dict:
    """构造传给后端的 runtime_context。"""
    ctx = {
        "has_resume": has_resume,
        "has_jd": has_jd,
    }
    if has_resume and resume_input.strip():
        ctx["resume_input"] = resume_input.strip()
    if has_jd and jd_input.strip():
        ctx["jd_input"] = jd_input.strip()
    return ctx


def render_chat_history() -> None:
    """渲染历史对话。"""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# =========================
# 会话状态初始化
# =========================
def init_session_state() -> None:
    """
    初始化 Streamlit 会话状态。
    优先从 URL 参数恢复历史会话，否则创建新会话。
    所有数据通过后端 API 获取。
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "resume_input" not in st.session_state:
        st.session_state.resume_input = ""
    if "jd_input" not in st.session_state:
        st.session_state.jd_input = ""
    if "has_resume" not in st.session_state:
        st.session_state.has_resume = False
    if "has_jd" not in st.session_state:
        st.session_state.has_jd = False
    if "last_resume_name" not in st.session_state:
        st.session_state.last_resume_name = ""
    if "last_jd_name" not in st.session_state:
        st.session_state.last_jd_name = ""

    # 会话 ID：从 URL 参数读取，或取最近会话，或新建
    if "current_session_id" not in st.session_state:
        sid = st.query_params.get("sid", "")
        if sid:
            st.session_state.current_session_id = sid
        else:
            sessions = api_list_sessions(limit=1)
            if sessions:
                st.session_state.current_session_id = sessions[0]["session_id"]
            else:
                st.session_state.current_session_id = api_create_session()

    # 从后端加载会话数据
    if "session_loaded" not in st.session_state:
        sid = st.session_state.current_session_id
        data = api_load_session(sid)
        if data:
            st.session_state.chat_history = data.get("chat_history", [])
            # 恢复 has_resume / has_jd 状态
            agent_state = data.get("agent_state", {})
            st.session_state.has_resume = agent_state.get("has_resume", False)
            st.session_state.has_jd = agent_state.get("has_jd", False)
        st.session_state.session_loaded = True


def new_session() -> None:
    """创建新会话并跳转。"""
    # 保存当前会话
    api_save_session(st.session_state.current_session_id, st.session_state.chat_history)
    # 创建新会话
    new_id = api_create_session()
    if not new_id:
        return
    st.session_state.current_session_id = new_id
    st.session_state.chat_history = []
    st.session_state.resume_input = ""
    st.session_state.jd_input = ""
    st.session_state.has_resume = False
    st.session_state.has_jd = False
    st.session_state.last_resume_name = ""
    st.session_state.last_jd_name = ""
    st.rerun()


# =========================
# 初始化
# =========================
init_session_state()


# =========================
# 侧边栏：上下文输入区 + 会话管理
# =========================
with st.sidebar:
    st.header("📄 上下文输入")
    st.write("支持上传 `txt/pdf` 文件，也支持直接粘贴文本。")

    st.subheader("简历")
    resume_file = st.file_uploader(
        "上传简历文件",
        type=["txt", "pdf"],
        key="resume_uploader",
        help="支持 txt / pdf",
    )
    resume_text_input = st.text_area(
        "或者直接粘贴简历文本",
        value="" if st.session_state.last_resume_name else st.session_state.resume_input,
        height=180,
        key="resume_text_area",
        placeholder="请粘贴简历文本，或者上传 txt/pdf 简历文件",
    )

    st.subheader("岗位 JD")
    jd_file = st.file_uploader(
        "上传 JD 文件",
        type=["txt", "pdf"],
        key="jd_uploader",
        help="支持 txt / pdf",
    )
    jd_text_input = st.text_area(
        "或者直接粘贴 JD 文本",
        value="" if st.session_state.last_jd_name else st.session_state.jd_input,
        height=180,
        key="jd_text_area",
        placeholder="请粘贴 JD 文本，或者上传 txt/pdf JD 文件",
    )

    st.divider()

    # ---- 会话管理按钮 ----
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("+ 新对话", use_container_width=True, type="primary"):
            new_session()
    with btn_col2:
        if st.button("清空对话", use_container_width=True):
            st.session_state.chat_history = []
            api_save_session(st.session_state.current_session_id, [])
            st.rerun()

    # ---- 历史会话列表 ----
    st.divider()
    st.subheader("📋 历史会话")

    sessions = api_list_sessions(limit=20)
    current_sid = st.session_state.get("current_session_id", "")

    if not sessions:
        st.caption("暂无历史会话")
    else:
        for sess in sessions:
            sid = sess["session_id"]
            title = sess["title"] or "新会话"
            is_current = sid == current_sid

            ts = datetime.fromtimestamp(sess["updated_at"]).strftime("%m-%d %H:%M")
            label = f"{'▸ ' if is_current else ''}{ts} | {title}"

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    label,
                    key=f"sess_{sid}",
                    use_container_width=True,
                    disabled=is_current,
                    type="primary" if is_current else "secondary",
                ):
                    # 切换会话：先保存当前，再加载目标
                    api_save_session(st.session_state.current_session_id, st.session_state.chat_history)
                    target = api_load_session(sid)
                    if target:
                        st.session_state.current_session_id = sid
                        st.session_state.chat_history = target.get("chat_history", [])
                        agent_state = target.get("agent_state", {})
                        st.session_state.has_resume = agent_state.get("has_resume", False)
                        st.session_state.has_jd = agent_state.get("has_jd", False)
                        st.query_params["sid"] = sid
                    st.rerun()
            with col_del:
                if not is_current:
                    if st.button("🗑", key=f"del_{sid}"):
                        api_delete_session(sid)
                        st.rerun()

    # ---- 后端连接状态 ----
    st.divider()
    try:
        resp = httpx.get(f"{API_BASE}/api/health", timeout=2.0)
        if resp.status_code == 200:
            st.success(f"后端已连接：{API_BASE}")
        else:
            st.warning(f"后端异常（状态码 {resp.status_code}）")
    except Exception:
        st.error(f"后端未连接：{API_BASE}")


# =========================
# 同步输入：优先文件，其次文本
# =========================
# 简历
if resume_file is not None:
    file_path = api_upload_file(
        st.session_state.current_session_id,
        resume_file.name,
        bytes(resume_file.getbuffer()),
    )
    st.session_state.resume_input = file_path
    st.session_state.has_resume = bool(file_path)
    st.session_state.last_resume_name = resume_file.name
    if file_path:
        st.success(f"已上传简历文件：{resume_file.name}")
else:
    st.session_state.resume_input = resume_text_input.strip()
    st.session_state.has_resume = bool(resume_text_input.strip())
    if resume_text_input.strip():
        st.session_state.last_resume_name = ""

# JD
if jd_file is not None:
    file_path = api_upload_file(
        st.session_state.current_session_id,
        jd_file.name,
        bytes(jd_file.getbuffer()),
    )
    st.session_state.jd_input = file_path
    st.session_state.has_jd = bool(file_path)
    st.session_state.last_jd_name = jd_file.name
    if file_path:
        st.success(f"已上传 JD 文件：{jd_file.name}")
else:
    st.session_state.jd_input = jd_text_input.strip()
    st.session_state.has_jd = bool(jd_text_input.strip())
    if jd_text_input.strip():
        st.session_state.last_jd_name = ""


# =========================
# 主界面
# =========================
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("💬 对话区")
    render_chat_history()

    prompt = st.chat_input(
        "请输入你的问题，比如：帮我模拟面试 / 给我面试题 / 给我建议 / 什么是 RAG？"
    )

    if prompt:
        # 先显示用户消息
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 构建 runtime_context
        runtime_context = build_runtime_context(
            has_resume=st.session_state.has_resume,
            has_jd=st.session_state.has_jd,
            resume_input=st.session_state.resume_input,
            jd_input=st.session_state.jd_input,
        )

        with st.spinner("AI 正在分析你的需求并生成回复..."):
            try:
                # 调用后端 SSE 流式接口，逐字渲染
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    assistant_text = ""

                    for ch in api_chat_stream(
                        session_id=st.session_state.current_session_id,
                        query=prompt,
                        runtime_context=runtime_context,
                        chat_history=st.session_state.chat_history,
                    ):
                        assistant_text += ch
                        placeholder.markdown(assistant_text)

                # 统一保存历史
                assistant_text = assistant_text.strip()
                if not assistant_text:
                    assistant_text = "未生成有效回复，请检查输入内容。"

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_text}
                )

            except Exception as e:
                error_text = f"发生错误：{e}"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_text}
                )
                with st.chat_message("assistant"):
                    st.error(error_text)

with right_col:
    st.subheader("📌 当前上下文")
    st.write(f"**是否有简历：** {'是' if st.session_state.has_resume else '否'}")
    st.write(f"**是否有 JD：** {'是' if st.session_state.has_jd else '否'}")

    if st.session_state.last_resume_name:
        st.write(f"**简历文件：** `{st.session_state.last_resume_name}`")
    elif st.session_state.resume_input:
        st.write("**简历来源：** 手动输入")

    if st.session_state.last_jd_name:
        st.write(f"**JD 文件：** `{st.session_state.last_jd_name}`")
    elif st.session_state.jd_input:
        st.write("**JD 来源：** 手动输入")

    st.divider()
    st.subheader("🧭 使用提示")
    st.markdown(
        """
- 上传简历或 JD 后，可直接输入：
  - `帮我模拟面试`
  - `给我一些面试题`
  - `给我面试准备建议`
- 没有文件时，可直接问专业知识问题
- 遇到无关问题会自动拒答
        """
    )
