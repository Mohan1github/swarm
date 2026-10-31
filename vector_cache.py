from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

VECTOR_DIMENSIONS = 768
VECTOR_CACHE_VERSION = "lexical-hash-v1"
DEFAULT_DB_PATH = Path(__file__).with_name("vector_cache.sqlite3")
STOP_WORDS = {
    "a", "an", "and", "are", "about", "at", "be", "can", "could", "do", "does",
    "for", "from", "in", "is", "it", "me", "my", "of", "on", "or", "our", "please",
    "the", "to", "us", "we", "what", "when", "where", "which", "who", "why", "with",
    "would", "you", "your",
}


@dataclass(frozen=True)
class VectorCacheHit:
    query: str
    result: dict
    similarity: float


def _features(text: str) -> dict[int, float]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = [
        token for token in re.findall(r"[\w']+", normalized, flags=re.UNICODE)
        if token not in STOP_WORDS
    ]
    weights: dict[int, float] = {}

    def add(feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % VECTOR_DIMENSIONS
        sign = 1.0 if value & (1 << 63) else -1.0
        weights[index] = weights.get(index, 0.0) + sign * weight

    for token in tokens:
        add(f"word:{token}", 1.0)
        padded = f"^{token}$"
        for size in (3, 4):
            for start in range(max(0, len(padded) - size + 1)):
                add(f"char{size}:{padded[start:start + size]}", 0.12)

    for left, right in zip(tokens, tokens[1:]):
        add(f"pair:{left}:{right}", 0.7)

    magnitude = math.sqrt(sum(value * value for value in weights.values()))
    if magnitude == 0:
        return {}
    return {index: value / magnitude for index, value in weights.items() if value}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return max(0.0, sum(value * right.get(index, 0.0) for index, value in left.items()))


def _query_key(query: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", query).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class VectorAnswerCache:
    def __init__(
        self,
        db_path: str | Path | None = None,
        similarity_threshold: float | None = None,
        max_entries: int = 5000,
    ) -> None:
        self.db_path = Path(db_path or os.getenv("VECTOR_CACHE_DB_PATH") or DEFAULT_DB_PATH)
        threshold = similarity_threshold
        if threshold is None:
            threshold = float(os.getenv("VECTOR_CACHE_SIMILARITY_THRESHOLD", "0.93"))
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Vector cache similarity threshold must be in (0, 1].")
        self.similarity_threshold = threshold
        self.max_entries = max(1, max_entries)
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
                CREATE TABLE IF NOT EXISTS vector_answers (
                    user_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    cache_version TEXT NOT NULL,
                    query_key TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, model, cache_version, query_key)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS vector_answers_scope_idx "
                "ON vector_answers (user_id, model, cache_version)"
            )

    def lookup(self, user_id: int, model: str, query: str) -> VectorCacheHit | None:
        query_vector = _features(query)
        if not query_vector:
            return None

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT query_text, vector_json, result_json
                FROM vector_answers
                WHERE user_id = ? AND model = ? AND cache_version = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, model, VECTOR_CACHE_VERSION, self.max_entries),
            ).fetchall()

        best_hit = None
        best_similarity = 0.0
        for row in rows:
            stored_vector = {int(index): value for index, value in json.loads(row["vector_json"]).items()}
            similarity = _cosine(query_vector, stored_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_hit = row

        if best_hit is None or best_similarity < self.similarity_threshold:
            return None

        return VectorCacheHit(
            query=best_hit["query_text"],
            result=json.loads(best_hit["result_json"]),
            similarity=best_similarity,
        )

    def store(self, user_id: int, model: str, query: str, result: dict) -> None:
        vector = _features(query)
        if not vector:
            return

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO vector_answers (
                    user_id, model, cache_version, query_key, query_text, vector_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, model, cache_version, query_key) DO UPDATE SET
                    query_text = excluded.query_text,
                    vector_json = excluded.vector_json,
                    result_json = excluded.result_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    model,
                    VECTOR_CACHE_VERSION,
                    _query_key(query),
                    query,
                    json.dumps(vector, separators=(",", ":")),
                    json.dumps(result, separators=(",", ":")),
                ),
            )
            connection.execute(
                """
                DELETE FROM vector_answers
                WHERE rowid IN (
                    SELECT rowid FROM vector_answers
                    WHERE user_id = ? AND model = ? AND cache_version = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (user_id, model, VECTOR_CACHE_VERSION, self.max_entries),
            )

    def clear_user(self, user_id: int) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM vector_answers WHERE user_id = ?", (user_id,))
