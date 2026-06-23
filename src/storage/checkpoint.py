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