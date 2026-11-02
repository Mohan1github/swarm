from __future__ import annotations

import os
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from vector_cache import _cosine, _features

DEFAULT_DB_PATH = Path(__file__).with_name("query_activity.sqlite3")


class UserActivityStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.getenv("ACTIVITY_DB_PATH") or DEFAULT_DB_PATH)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    query_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    attachment_count INTEGER NOT NULL DEFAULT 0,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS query_activity_user_idx "
                "ON query_activity (user_id, id DESC)"
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(query_activity)").fetchall()
            }
            if "insights_json" not in columns:
                connection.execute(
                    "ALTER TABLE query_activity ADD COLUMN insights_json TEXT NOT NULL DEFAULT '{}'"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_activity_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    activity_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    output_excerpt TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (activity_id) REFERENCES query_activity(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS query_activity_steps_user_idx "
                "ON query_activity_steps (user_id, id DESC)"
            )

    def record(
        self,
        user_id: int,
        query: str,
        model: str,
        attachment_count: int = 0,
        cache_hit: bool = False,
    ) -> int | None:
        text = query.strip()
        if not text:
            return None
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO query_activity (user_id, query_text, model, attachment_count, cache_hit)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, text[:2000], model[:200], max(0, attachment_count), int(cache_hit)),
            )
            return int(cursor.lastrowid)

    def record_step(self, activity_id: int, user_id: int, role: str, output: str) -> None:
        excerpt = " ".join(output.split())[:320]
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO query_activity_steps (activity_id, user_id, role, output_excerpt)
                SELECT id, user_id, ?, ?
                FROM query_activity
                WHERE id = ? AND user_id = ?
                ON CONFLICT DO NOTHING
                """,
                (role[:80], excerpt, activity_id, user_id),
            )

    def record_insights(self, activity_id: int | None, user_id: int, workflow: dict) -> None:
        if activity_id is None:
            return
        steps = [
            {
                "role": str(step.get("role", ""))[:80],
                "action": str(step.get("action", ""))[:80],
                "status": str(step.get("status", "completed"))[:30],
                "output": " ".join(str(step.get("output", "")).split())[:220],
            }
            for step in workflow.get("steps", [])
        ]
        insights = {
            "steps": steps,
            "action_probabilities": workflow.get("action_probabilities", {}),
            "validation_score": workflow.get("validation_score"), 
            "validation_issues": workflow.get("validation_issues", []),
            "answer": str(workflow.get("answer", ""))[:6000],
        }
        with self._connection() as connection:
            connection.execute(
                "UPDATE query_activity SET insights_json = ? WHERE id = ? AND user_id = ?",
                (json.dumps(insights, ensure_ascii=True), activity_id, user_id),
            )

    def clear_user(self, user_id: int) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM query_activity_steps WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM query_activity WHERE user_id = ?", (user_id,))

    def graph(self, user_id: int, limit: int = 80, step_limit: int = 160) -> dict:
        with self._connection() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM query_activity WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            stage_total = connection.execute(
                "SELECT COUNT(*) FROM query_activity_steps WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT id, query_text, model, attachment_count, cache_hit, created_at, insights_json
                FROM query_activity
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, max(1, min(limit, 120))),
            ).fetchall()

        activities = list(reversed(rows))
        query_nodes = []
        for row in activities:
            try:
                insights = json.loads(row["insights_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                insights = {}
            query_nodes.append({
                "id": str(row["id"]),
                "kind": "query",
                "query": row["query_text"],
                "model": row["model"],
                "attachment_count": row["attachment_count"],
                "cache_hit": bool(row["cache_hit"]),
                "created_at": row["created_at"],
                "parent_id": None,
                "insights": insights,
            })

        step_rows = []
        activity_ids = [row["id"] for row in activities]
        if activity_ids:
            placeholders = ", ".join("?" for _ in activity_ids)
            with self._connection() as connection:
                step_rows = connection.execute(
                    f"""
                    SELECT id, activity_id, role, output_excerpt, created_at
                    FROM query_activity_steps
                    WHERE user_id = ? AND activity_id IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, *activity_ids, max(1, min(step_limit, 240))),
                ).fetchall()
            step_rows.reverse()

        query_by_id = {str(node["id"]): node for node in query_nodes}
        stage_nodes = [
            {
                "id": f"step-{row['id']}",
                "kind": "stage",
                "query": row["output_excerpt"] or f"{row['role']} stage completed",
                "output": row["output_excerpt"],
                "stage": row["role"],
                "created_at": row["created_at"],
                "parent_id": str(row["activity_id"]),
                "parent_query": query_by_id[str(row["activity_id"])]["query"],
                "attachment_count": 0,
                "cache_hit": False,
            }
            for row in step_rows
            if str(row["activity_id"]) in query_by_id
        ]
        nodes = query_nodes + stage_nodes

        vectors = [_features(node["query"]) for node in query_nodes]
        neighbors: dict[int, list[tuple[float, int]]] = {index: [] for index in range(len(query_nodes))}
        for left in range(len(query_nodes)):
            for right in range(left + 1, len(query_nodes)):
                similarity = _cosine(vectors[left], vectors[right])
                if similarity >= 0.2:
                    neighbors[left].append((similarity, right))
                    neighbors[right].append((similarity, left))

        selected_edges = set()
        for index, candidates in neighbors.items():
            for similarity, neighbor in sorted(candidates, reverse=True)[:3]:
                pair = tuple(sorted((index, neighbor)))
                selected_edges.add((pair[0], pair[1], similarity))

        related_edges = [
            {
                "id": f"related:{query_nodes[left]['id']}:{query_nodes[right]['id']}",
                "kind": "related",
                "source": query_nodes[left]["id"],
                "target": query_nodes[right]["id"],
                "similarity": round(similarity, 3),
                "context": f"Related prompts · {round(similarity * 100)}% similarity",
            }
            for left, right, similarity in sorted(selected_edges)
        ]
        stage_edges = [
            {
                "id": f"stage:{node['parent_id']}:{node['id']}",
                "kind": "stage",
                "source": node["parent_id"],
                "target": node["id"],
                "similarity": 1.0,
                "context": f"{node['stage']} output for this prompt",
            }
            for node in stage_nodes
        ]
        edges = related_edges + stage_edges

        return {
            "nodes": nodes,
            "edges": edges,
            "total": total,
            "shown": len(query_nodes),
            "stage_total": stage_total,
            "stage_shown": len(stage_nodes),
            "connections": len(edges),
        }