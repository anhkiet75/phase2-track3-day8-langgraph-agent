"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> object | None:
    """Return a LangGraph checkpointer.

    TODO(student): implement SQLite support for the persistence extension track.
    The starter provides MemorySaver only — SQLite/Postgres are extension tasks.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from exc
        connection = sqlite3.connect(
            database_url or "langgraph_agent_lab.sqlite",
            check_same_thread=False,
        )
        connection.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(connection)
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
