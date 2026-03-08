from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Interaction:
    interaction_id: str
    ts: float
    user_text: str
    assistant_text: str
    retrieved_context: str


class TrainingStore:
    """
    Stores incremental supervision data derived from usage:
    - user prompt
    - retrieved context (paper chunks)
    - assistant response

    This is not pretraining; it's a growing instruction dataset for periodic fine-tuning.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id      TEXT PRIMARY KEY,
                ts                  REAL NOT NULL,
                user_text           TEXT NOT NULL,
                assistant_text      TEXT NOT NULL,
                retrieved_context   TEXT NOT NULL DEFAULT '',
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);
            """
        )
        self._conn.commit()

    def log_interaction(
        self,
        *,
        interaction_id: str,
        ts: float,
        user_text: str,
        assistant_text: str,
        retrieved_context: str,
    ) -> None:
        self._conn.cursor().execute(
            """
            INSERT OR REPLACE INTO interactions
                (interaction_id, ts, user_text, assistant_text, retrieved_context)
            VALUES (?, ?, ?, ?, ?)
            """,
            (interaction_id, ts, user_text, assistant_text, retrieved_context),
        )
        self._conn.commit()

    def count_interactions(self) -> int:
        row = self._conn.cursor().execute("SELECT COUNT(*) AS n FROM interactions").fetchone()
        return int(row["n"]) if row else 0

    def iter_interactions(self, limit: int = 1000) -> list[Interaction]:
        rows = self._conn.cursor().execute(
            """
            SELECT interaction_id, ts, user_text, assistant_text, retrieved_context
            FROM interactions
            ORDER BY ts ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            Interaction(
                interaction_id=r["interaction_id"],
                ts=float(r["ts"]),
                user_text=r["user_text"],
                assistant_text=r["assistant_text"],
                retrieved_context=r["retrieved_context"],
            )
            for r in rows
        ]

