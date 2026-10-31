from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from vector_cache import _cosine, _features


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "agent_output_vectors.sqlite3"
MAX_OUTPUTS_PER_USER = 5000


class AgentOutputVectorAgent:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.getenv("AGENT_OUTPUT_VECTOR_DB_PATH") or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_output_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS agent_output_vectors_user_idx "
                "ON agent_output_vectors (user_id, id DESC)"
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def store(
        self,
        user_id: int,
        model: str,
        run_id: str,
        query: str,
        role: str,
        action: str,
        output: str,
    ) -> None:
        text = output.strip()[:12000]
        vector = _features(f"{query}\n{role} {action}\n{text}")
        if not vector:
            return
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO agent_output_vectors (
                    user_id, run_id, model, query_text, role, action, output_text, vector_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    run_id,
                    model[:200],
                    query[:2000],
                    role[:80],
                    action[:80],
                    text,
                    json.dumps(vector, separators=(",", ":")),
                ),
            )
            connection.execute(
                """
                DELETE FROM agent_output_vectors
                WHERE user_id = ?
                  AND id NOT IN (
                      SELECT id FROM agent_output_vectors
                      WHERE user_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (user_id, user_id, MAX_OUTPUTS_PER_USER),
            )

    def retrieve(
        self,
        user_id: int,
        model: str,
        run_id: str,
        query: str,
        focus: dict,
        limit: int = 8,
    ) -> list[dict]:
        query_text = " ".join([
            query,
            focus.get("search_terms", ""),
            *focus.get("focus_points", []),
            *focus.get("comparative_points", []),
            *focus.get("constraints", []),
        ])
        query_vector = _features(query_text)
        if not query_vector:
            return []

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT run_id, model, query_text, role, action, output_text, vector_json, created_at
                FROM agent_output_vectors
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, MAX_OUTPUTS_PER_USER),
            ).fetchall()

        matches = []
        for row in rows:
            stored_vector = {
                int(index): float(value)
                for index, value in json.loads(row["vector_json"]).items()
            }
            similarity = _cosine(query_vector, stored_vector)
            if similarity < 0.035:
                continue
            matches.append({
                "run_id": row["run_id"],
                "model": row["model"],
                "query": row["query_text"],
                "role": row["role"],
                "action": row["action"],
                "output": row["output_text"],
                "created_at": row["created_at"],
                "similarity": round(similarity, 4),
                "current_run": row["run_id"] == run_id,
            })

        matches.sort(
            key=lambda item: (
                item["similarity"] + (0.12 if item["current_run"] else 0),
                item["created_at"],
            ),
            reverse=True,
        )
        return matches[:max(1, min(limit, 20))]

    def clear_user(self, user_id: int) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM agent_output_vectors WHERE user_id = ?",
                (user_id,),
            )

    @staticmethod
    def create_run_id() -> str:
        return uuid.uuid4().hex
