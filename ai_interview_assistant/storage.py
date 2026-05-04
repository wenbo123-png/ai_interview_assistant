"""
会话历史持久化存储模块

使用 SQLite 存储会话数据，支持：
- 保存/读取/删除会话
- 列出历史会话（按更新时间倒序）
- 自动清理过期会话
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from ai_interview_assistant.utils.logger_handler import logger
from ai_interview_assistant.utils.path_tool import get_abs_path

# 数据库路径
DB_PATH = get_abs_path("data/sessions.db")


def _get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接，自动创建目录和表。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            chat_history TEXT NOT NULL DEFAULT '[]',
            agent_state TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.commit()
    return conn


def generate_session_id() -> str:
    """生成唯一会话 ID：sess_日期_随机串。"""
    import random
    import string
    date_part = time.strftime("%Y%m%d")
    rand_part = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"sess_{date_part}_{rand_part}"


def save_session(
    session_id: str,
    chat_history: list[dict[str, Any]],
    agent_state: dict[str, Any],
    title: str = "",
) -> None:
    """保存或更新会话。"""
    conn = _get_conn()
    now = time.time()

    # 如果没给 title，用首条用户消息截断作为标题
    if not title:
        for msg in chat_history:
            if msg.get("role") == "user":
                title = msg["content"][:30]
                break

    try:
        conn.execute(
            """
            INSERT INTO sessions (session_id, title, created_at, updated_at, chat_history, agent_state)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at,
                chat_history = excluded.chat_history,
                agent_state = excluded.agent_state
            """,
            (
                session_id,
                title,
                now,
                now,
                json.dumps(chat_history, ensure_ascii=False),
                json.dumps(agent_state, ensure_ascii=False),
            ),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"[Storage] 保存会话失败: {e}")
    finally:
        conn.close()


def load_session(session_id: str) -> dict[str, Any] | None:
    """读取指定会话，返回 None 表示不存在。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT chat_history, agent_state FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "chat_history": json.loads(row[0]),
            "agent_state": json.loads(row[1]),
        }
    except Exception as e:
        logger.error(f"[Storage] 读取会话失败: {e}")
        return None
    finally:
        conn.close()


def delete_session(session_id: str) -> bool:
    """删除指定会话，返回是否成功。"""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"[Storage] 删除会话失败: {e}")
        return False
    finally:
        conn.close()


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """列出历史会话，按更新时间倒序。"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"[Storage] 列出会话失败: {e}")
        return []
    finally:
        conn.close()
