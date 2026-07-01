"""SQLite checkpoint factory — replaces MemorySaver for persistent graph state."""

import os
import sqlite3

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def get_checkpoint_db_path() -> str:
    """Get checkpoint DB path from env or default."""
    return os.getenv("CHECKPOINT_DB", "data/checkpoints.db")


def create_checkpointer(db_path: str | None = None) -> AsyncSqliteSaver:
    """Create an AsyncSqliteSaver for LangGraph checkpointing."""
    db_path = db_path or get_checkpoint_db_path()
    os.makedirs(os.path.dirname(db_path) or "data", exist_ok=True)
    conn = aiosqlite.connect(db_path)
    return AsyncSqliteSaver(conn)


def get_pm_checkpoint_db_path() -> str:
    """Get PM-specific checkpoint DB path."""
    return os.getenv("PM_CHECKPOINT_DB", "data/pm_checkpoints.db")


def get_chaos_checkpoint_db_path() -> str:
    """Get Chaos-specific checkpoint DB path."""
    return os.getenv("CHAOS_CHECKPOINT_DB", "data/chaos_checkpoints.db")


def list_thread_ids(limit: int = 50, db_path: str | None = None) -> list[str]:
    """List recent thread_ids from the checkpoint database."""
    db_path = db_path or get_checkpoint_db_path()
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
#  TestPilot history storage (not LangGraph-based)
# ═══════════════════════════════════════════════════════════

import json as _json
from datetime import datetime, timezone

TESTPILOT_DB = os.path.join("data", "testpilot.db")


def _get_testpilot_conn() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(TESTPILOT_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS testpilot_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            config TEXT,
            result TEXT,
            created_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def save_testpilot_record(thread_id: str, record_type: str, config: dict | None = None, result: dict | None = None) -> None:
    conn = _get_testpilot_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO testpilot_history (thread_id, type, config, result, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                record_type,
                _json.dumps(config, ensure_ascii=False) if config else None,
                _json.dumps(result, ensure_ascii=False) if result else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_testpilot_records(limit: int = 30) -> list[dict]:
    conn = _get_testpilot_conn()
    try:
        rows = conn.execute(
            "SELECT thread_id, type, config, result, created_at FROM testpilot_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        records = []
        for row in rows:
            config = _json.loads(row[2]) if row[2] else None
            records.append({
                "thread_id": row[0],
                "type": row[1],
                "title": _testpilot_title(row[1], config),
                "created_at": row[3],
            })
        return records
    finally:
        conn.close()


def get_testpilot_record(thread_id: str) -> dict | None:
    conn = _get_testpilot_conn()
    try:
        row = conn.execute(
            "SELECT thread_id, type, config, result, created_at FROM testpilot_history WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "thread_id": row[0],
            "type": row[1],
            "config": _json.loads(row[2]) if row[2] else None,
            "result": _json.loads(row[3]) if row[3] else None,
            "created_at": row[4],
        }
    finally:
        conn.close()


def _testpilot_title(record_type: str, config: dict | None) -> str:
    if record_type == "generate_cases":
        count = len((config or {}).get("cases", []))
        return f"用例生成 ({count} 条)"
    elif record_type == "generate_scripts":
        script_type = (config or {}).get("script_type", "pytest")
        return f"脚本生成 ({script_type})"
    elif record_type == "analyze_defect":
        return "缺陷分析"
    return record_type