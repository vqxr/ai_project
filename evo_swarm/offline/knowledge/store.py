from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from evo_swarm.offline.training.store import TrainingStore


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    doc_id: str
    doc_path: str
    offset: int
    text: str
    score: float


class KnowledgeStore:
    """
    SQLite-backed paper store with FTS5 full-text search.

    Why FTS5:
    - fully offline
    - no Python deps
    - gives you BM25 ranking out of the box
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self.training = TrainingStore(self._conn)

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS documents (
                doc_id      TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                sha256      TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                offset      INTEGER NOT NULL,
                text        TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );

            -- Contentless FTS table; chunk_id is stored for lookup.
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                tokenize = 'porter'
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------------------
    # Ingestion
    # ---------------------------

    def upsert_document(self, path: str, sha256: str) -> str:
        doc_id = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO documents (doc_id, path, sha256) VALUES (?, ?, ?)",
            (doc_id, path, sha256),
        )
        self._conn.commit()
        return doc_id

    def replace_chunks(self, doc_id: str, chunks: Iterable[tuple[int, str]]) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        cur.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))

        for offset, text in chunks:
            chunk_id = hashlib.sha256(f"{doc_id}:{offset}".encode()).hexdigest()[:24]
            cur.execute(
                "INSERT INTO chunks (chunk_id, doc_id, offset, text) VALUES (?, ?, ?, ?)",
                (chunk_id, doc_id, offset, text),
            )
            cur.execute(
                "INSERT INTO chunks_fts (text, chunk_id, doc_id) VALUES (?, ?, ?)",
                (text, chunk_id, doc_id),
            )

        self._conn.commit()

    # ---------------------------
    # Retrieval
    # ---------------------------

    def search(self, query: str, limit: int = 8) -> list[ChunkHit]:
        """
        Returns top chunks by BM25 score (lower is better in SQLite bm25()).
        """
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT
                c.chunk_id AS chunk_id,
                c.doc_id AS doc_id,
                d.path AS doc_path,
                c.offset AS offset,
                c.text AS text,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts) ASC
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()

        return [
            ChunkHit(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                doc_path=r["doc_path"],
                offset=r["offset"],
                text=r["text"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    def get_document_path(self, doc_id: str) -> str | None:
        row = self._conn.cursor().execute(
            "SELECT path FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row["path"] if row else None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_probably_text_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in {".txt", ".md", ".rst", ".tex", ".pdf"}
