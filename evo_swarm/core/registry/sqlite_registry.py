import sqlite3
import json
import os
from typing import List, Optional
from evo_swarm.core.genomes import Candidate, Genome
from evo_swarm.core.registry.registry import Registry


class SqliteRegistry(Registry):
    """
    SQLite-backed implementation of the Registry.
    Stores candidates in a relational schema with a dedicated lineage table.
    """

    def __init__(self, db_path: str = "evo_swarm.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                id              TEXT PRIMARY KEY,
                generation      INTEGER NOT NULL,
                status          TEXT NOT NULL DEFAULT 'proposed',
                fitness_score   REAL,
                genome_json     TEXT NOT NULL,
                metrics_json    TEXT NOT NULL DEFAULT '{}',
                run_history_json TEXT NOT NULL DEFAULT '[]',
                artifacts_json  TEXT NOT NULL DEFAULT '{}',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS lineage (
                candidate_id TEXT NOT NULL,
                parent_id    TEXT NOT NULL,
                PRIMARY KEY (candidate_id, parent_id),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id),
                FOREIGN KEY (parent_id)    REFERENCES candidates(id)
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_generation
                ON candidates(generation);

            CREATE INDEX IF NOT EXISTS idx_candidates_fitness
                ON candidates(fitness_score DESC);

            CREATE INDEX IF NOT EXISTS idx_lineage_parent
                ON lineage(parent_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Registry interface
    # ------------------------------------------------------------------

    def save_candidate(self, candidate: Candidate):
        cursor = self._conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO candidates
                (id, generation, status, fitness_score,
                 genome_json, metrics_json, run_history_json, artifacts_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.id,
                candidate.generation,
                candidate.status,
                candidate.fitness_score,
                candidate.genome.model_dump_json(),
                json.dumps(candidate.metrics),
                json.dumps(candidate.run_history),
                json.dumps(candidate.artifacts),
            ),
        )

        # Upsert lineage rows
        for parent_id in candidate.parent_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO lineage (candidate_id, parent_id) VALUES (?, ?)",
                (candidate.id, parent_id),
            )

        self._conn.commit()

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        cursor = self._conn.cursor()
        row = cursor.execute(
            "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()

        if row is None:
            return None
        return self._row_to_candidate(row)

    def get_generation(self, generation_id: int) -> List[Candidate]:
        cursor = self._conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM candidates WHERE generation = ?", (generation_id,)
        ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def get_best_candidates(self, limit: int = 5) -> List[Candidate]:
        cursor = self._conn.cursor()
        rows = cursor.execute(
            """
            SELECT * FROM candidates
            WHERE status IN ('completed', 'evaluated')
              AND fitness_score IS NOT NULL
            ORDER BY fitness_score DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def get_lineage_tree(self, candidate_id: str) -> List[Candidate]:
        """
        Walk the lineage table upward and return the full ancestry chain
        (ordered from the given candidate back to the root).
        """
        chain: List[Candidate] = []
        visited = set()
        current_id = candidate_id

        while current_id and current_id not in visited:
            visited.add(current_id)
            candidate = self.get_candidate(current_id)
            if candidate is None:
                break
            chain.append(candidate)

            # Move to the first parent (primary lineage)
            row = self._conn.cursor().execute(
                "SELECT parent_id FROM lineage WHERE candidate_id = ? LIMIT 1",
                (current_id,),
            ).fetchone()
            current_id = row["parent_id"] if row else None

        return chain

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_candidate(self, row: sqlite3.Row) -> Candidate:
        parent_rows = self._conn.cursor().execute(
            "SELECT parent_id FROM lineage WHERE candidate_id = ?", (row["id"],)
        ).fetchall()

        return Candidate(
            id=row["id"],
            parent_ids=[r["parent_id"] for r in parent_rows],
            generation=row["generation"],
            genome=Genome(**json.loads(row["genome_json"])),
            fitness_score=row["fitness_score"],
            metrics=json.loads(row["metrics_json"]),
            run_history=json.loads(row["run_history_json"]),
            artifacts=json.loads(row["artifacts_json"]),
            status=row["status"],
        )

    def close(self):
        self._conn.close()
