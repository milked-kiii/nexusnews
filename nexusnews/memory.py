# nexusnews/memory.py
"""Cross-run repo memory: first-seen dates, push history, star snapshots.

Why this exists (2026-09-07): the GitHub Actions runner starts from a fresh
checkout every day, so the main items DB (gitignored) is empty on each run —
``delivered`` flags reset daily and the same repos (stablyai/orca appeared 8
times in 14 days) got pushed day after day. This store is a tiny SQLite DB
persisted across runs via actions/cache; it answers three questions the main
DB cannot:

  1. When did we FIRST see this repo?  (first_seen — the "emerging" gate)
  2. Have we ever pushed this repo?    (last_pushed — one push per repo)
  3. How fast are its stars growing?   (daily snapshots → 7d delta)

Design notes:
- Repo keys are lowercase ``owner/repo``.
- Repo rows are never pruned: a repo observed once but never pushed must NOT
  become "first seen" again after a prune (that would reset its emerging
  window). Only star snapshots older than 30 days are pruned (deltas only
  need 7 days).
- Seeding (``load_seed``) bootstraps the store from a committed JSON mapping
  of repos pushed before the memory existed, extracted from past Actions
  logs. INSERT OR IGNORE semantics mean seeding never clobbers live data.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def _today(now: datetime | None = None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()


class RepoMemory:
    """Persistent repo memory backed by SQLite (see module docstring)."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS repos (
                repo TEXT PRIMARY KEY,
                first_seen TEXT NOT NULL,
                last_pushed TEXT,
                last_stars INTEGER
            )
        """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS stars (
                repo TEXT NOT NULL,
                date TEXT NOT NULL,
                stars INTEGER NOT NULL,
                PRIMARY KEY (repo, date)
            )
        """)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "RepoMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── observation ────────────────────────────────────────────────

    def ensure_repo(self, repo: str, *, stars: int | None = None,
                    now: datetime | None = None) -> None:
        """Record that we saw ``repo`` today. First observation fixes
        first_seen forever; repeated observations only refresh star data."""
        repo = repo.lower()
        today = _today(now).isoformat()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO repos (repo, first_seen, last_pushed, last_stars) "
                "VALUES (?, ?, NULL, ?)",
                (repo, today, stars),
            )
            if stars is not None:
                self._connection.execute(
                    "UPDATE repos SET last_stars = ? WHERE repo = ?", (stars, repo))
                # upsert today's snapshot (same-day refresh wins)
                self._connection.execute(
                    "INSERT OR REPLACE INTO stars (repo, date, stars) VALUES (?, ?, ?)",
                    (repo, today, stars))

    # ── queries ────────────────────────────────────────────────────

    def is_pushed(self, repo: str) -> bool:
        row = self._connection.execute(
            "SELECT last_pushed FROM repos WHERE repo = ?", (repo.lower(),)).fetchone()
        return bool(row and row["last_pushed"])

    def is_emerging(self, repo: str, *, days: int = 14,
                    now: datetime | None = None) -> bool:
        """True if we first saw this repo within the last ``days`` days."""
        row = self._connection.execute(
            "SELECT first_seen FROM repos WHERE repo = ?", (repo.lower(),)).fetchone()
        if not row:
            return True  # never observed → treat as new (ensure_repo runs first anyway)
        cutoff = (_today(now) - timedelta(days=days)).isoformat()
        return row["first_seen"] >= cutoff

    def stars_delta(self, repo: str, *, days: int = 7,
                    now: datetime | None = None) -> int | None:
        """Star growth over the last ``days`` days: latest snapshot minus the
        most recent snapshot that is at least ``days`` old. None when there is
        no old-enough baseline yet (cold start)."""
        repo = repo.lower()
        latest = self._connection.execute(
            "SELECT stars FROM stars WHERE repo = ? ORDER BY date DESC LIMIT 1",
            (repo,)).fetchone()
        if not latest:
            return None
        baseline_date = (_today(now) - timedelta(days=days)).isoformat()
        baseline = self._connection.execute(
            "SELECT stars FROM stars WHERE repo = ? AND date <= ? ORDER BY date DESC LIMIT 1",
            (repo, baseline_date)).fetchone()
        if not baseline:
            return None
        return int(latest["stars"]) - int(baseline["stars"])

    # ── delivery ───────────────────────────────────────────────────

    def mark_pushed(self, repos: list[str], *, now: datetime | None = None) -> None:
        today = _today(now).isoformat()
        with self._connection:
            self._connection.executemany(
                "UPDATE repos SET last_pushed = ? WHERE repo = ?",
                [(today, repo.lower()) for repo in repos if repo])

    # ── bootstrap & maintenance ────────────────────────────────────

    def load_seed(self, seed: dict[str, str], *, now: datetime | None = None) -> int:
        """Seed repos pushed before the memory existed: {repo: push_date}.

        Rows are only inserted when absent, so seeding is idempotent and never
        clobbers live observation data. Returns the number of new rows."""
        today = _today(now).isoformat()
        inserted = 0
        with self._connection:
            for repo, pushed_at in seed.items():
                repo = repo.strip().lower()
                if not repo or "/" not in repo:
                    continue
                seen = pushed_at if pushed_at <= today else today
                cursor = self._connection.execute(
                    "INSERT OR IGNORE INTO repos (repo, first_seen, last_pushed, last_stars) "
                    "VALUES (?, ?, ?, NULL)",
                    (repo, seen, pushed_at),
                )
                inserted += cursor.rowcount
        return inserted

    def prune_snapshots(self, *, keep_days: int = 30, now: datetime | None = None) -> None:
        """Drop star snapshots older than ``keep_days``. Repo rows are kept
        forever (see module docstring for why)."""
        cutoff = (_today(now) - timedelta(days=keep_days)).isoformat()
        with self._connection:
            self._connection.execute("DELETE FROM stars WHERE date < ?", (cutoff,))


def load_seed_file(path: str | Path) -> dict[str, str]:
    """Read {repo: push_date} seed JSON; missing file → empty dict (warn)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logging.warning("memory seed file not found, skipping", extra={"path": str(path)})
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("memory seed file unreadable, skipping",
                        extra={"path": str(path), "error": str(exc)})
        return {}
    if not isinstance(data, dict):
        logging.warning("memory seed must be a JSON object, skipping", extra={"path": str(path)})
        return {}
    return {str(k): str(v) for k, v in data.items()}


class ReviewMemory:
    """Cross-run memory for the 🧪 竞品实测 (competitor review) line.

    GitHub repos have a stable identity (owner/name); competitor reviews do
    NOT — the same product event (e.g. "Claude Code 3.8 release") spawns
    dozens of reviews across Reddit/YouTube. The user's dedup rule (2026-09-08
    Q10 answer: "那这种不算") is:

      - Same competitor + same product version → push only the best review,
        ever. The second review of that version is NOT counted.
      - Version upgrade → new event, counting restarts.
      - URL level: a pushed URL never repeats.

    The LLM review scorer extracts the version (e.g. "3.8"); when extraction
    fails the version key is empty and the row degrades to URL-level dedup
    only (7-day review window still bounds repetition).

    Like RepoMemory this lives in its own SQLite file persisted across
    Actions runs via actions/cache (the main items DB resets daily).
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                competitor TEXT NOT NULL,
                version TEXT NOT NULL,
                url TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_pushed TEXT,
                PRIMARY KEY (competitor, version, url)
            )
        """)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ReviewMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── queries ──────────────────────────────────────────────────

    def is_version_pushed(self, competitor: str, version: str) -> bool:
        """True if this (competitor, version) event was already pushed."""
        row = self._connection.execute(
            "SELECT last_pushed FROM reviews WHERE competitor = ? AND version = ? LIMIT 1",
            (competitor, version)).fetchone()
        return bool(row and row["last_pushed"])

    def is_url_pushed(self, url: str) -> bool:
        """True if this review URL was already pushed for any competitor."""
        row = self._connection.execute(
            "SELECT last_pushed FROM reviews WHERE url = ? LIMIT 1",
            (url,)).fetchone()
        return bool(row and row["last_pushed"])

    def best_url_for_event(self, competitor: str, version: str) -> str | None:
        """Return the URL already pushed for this (competitor, version), if any."""
        row = self._connection.execute(
            "SELECT url FROM reviews WHERE competitor = ? AND version = ? AND last_pushed IS NOT NULL LIMIT 1",
            (competitor, version)).fetchone()
        return row["url"] if row else None

    # ── observation & delivery ───────────────────────────────────

    def observe(self, competitor: str, version: str, url: str, *,
                now: datetime | None = None) -> None:
        """Record that we saw this review today (first_seen fix, idempotent)."""
        today = _today(now).isoformat()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO reviews (competitor, version, url, first_seen, last_pushed) "
                "VALUES (?, ?, ?, ?, NULL)",
                (competitor, version, url, today))

    def mark_pushed(self, competitor: str, version: str, url: str, *,
                    now: datetime | None = None) -> None:
        today = _today(now).isoformat()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO reviews (competitor, version, url, first_seen, last_pushed) "
                "VALUES (?, ?, ?, ?, ?)",
                (competitor, version, url, today, today))
            self._connection.execute(
                "UPDATE reviews SET last_pushed = ? WHERE competitor = ? AND version = ? AND url = ?",
                (today, competitor, version, url))
