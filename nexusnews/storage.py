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

    def __enter__(self) -> "SQLiteItemStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
