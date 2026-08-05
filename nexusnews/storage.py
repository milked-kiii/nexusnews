from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Item


class SQLiteItemStore:
    def __init__(self, path: str | Path):
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT,
                content TEXT,
                published_at TEXT,
                dedupe_key TEXT NOT NULL UNIQUE,
                stored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                digest_id TEXT NOT NULL, scope TEXT NOT NULL, target_id TEXT NOT NULL,
                user_id TEXT NOT NULL, vote TEXT NOT NULL CHECK(vote IN ('up', 'down')),
                source_id TEXT, event_key TEXT, feedback_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (digest_id, scope, target_id, user_id)
            )
        """)

    def close(self) -> None:
        self._connection.close()

    def put(self, item: Item) -> bool:
        with self._connection:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO items (id, source, title, url, content, published_at, dedupe_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item.id, item.source, item.title, item.url, item.content, item.published_at, item.dedupe_key),
            )
        return cursor.rowcount == 1

    def put_many(self, items: Iterable[Item]) -> int:
        return sum(self.put(item) for item in items)

    def list(self, *, limit: int = 100) -> list[Item]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._connection.execute(
            "SELECT id, source, title, url, content, published_at, dedupe_key FROM items ORDER BY stored_at, id LIMIT ?",
            (limit,),
        ).fetchall()
        return [Item(**dict(row)) for row in rows]

    def recent(self, *, since: str, limit: int = 500) -> list[Item]:
        """Return published items in newest-first order."""
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._connection.execute(
            """SELECT id, source, title, url, content, published_at, dedupe_key
               FROM items WHERE published_at IS NOT NULL AND published_at >= ?
               ORDER BY published_at DESC, id LIMIT ?""",
            (since, limit),
        ).fetchall()
        return [Item(**dict(row)) for row in rows]

    def record_feedback(self, *, digest_id: str, scope: str, vote: str, user_id: str,
                        item_id: str | None = None, source_id: str | None = None, event_key: str | None = None) -> None:
        if scope not in {"item", "digest"} or vote not in {"up", "down"}:
            raise ValueError("feedback scope/vote must be item|digest and up|down")
        target = item_id if scope == "item" else digest_id
        if not target:
            raise ValueError("item feedback requires item_id")
        with self._connection:
            self._connection.execute("""INSERT INTO feedback
                (digest_id, scope, target_id, user_id, vote, source_id, event_key)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(digest_id, scope, target_id, user_id) DO UPDATE SET
                vote=excluded.vote, source_id=excluded.source_id, event_key=excluded.event_key,
                feedback_at=CURRENT_TIMESTAMP""",
                (digest_id, scope, target, user_id, vote, source_id, event_key))

    def feedback_rate(self, digest_id: str, *, scope: str = "item") -> dict[str, float | int | None]:
        row = self._connection.execute("""SELECT COUNT(*) total,
            SUM(CASE WHEN vote='up' THEN 1 ELSE 0 END) up FROM feedback
            WHERE digest_id=? AND scope=?""", (digest_id, scope)).fetchone()
        total, up = int(row["total"]), int(row["up"] or 0)
        return {"up": up, "down": total - up, "total": total, "rate": up / total if total else None}

    def __enter__(self) -> "SQLiteItemStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
