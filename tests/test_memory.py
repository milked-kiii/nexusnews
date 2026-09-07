"""Tests for the cross-run repo memory (2026-09-07 anti-repeat redesign).

Scenario coverage mirrors the real-world failure: stablyai/orca was pushed 8
times in 14 days because the Actions runner starts from an empty DB daily.
"""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nexusnews.memory import RepoMemory


NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


class RepoMemoryTests(unittest.TestCase):
    def test_first_seen_is_immutable_across_days(self):
        with tempfile.TemporaryDirectory() as d:
            with RepoMemory(Path(d) / "m.db") as m:
                m.ensure_repo("acme/new", stars=100, now=NOW)
                m.ensure_repo("acme/new", stars=200, now=NOW + timedelta(days=20))
                # first_seen stays day 1 even after 20 more days of observation
                self.assertTrue(m.is_emerging("acme/new", now=NOW))
                self.assertFalse(m.is_emerging("acme/new", now=NOW + timedelta(days=20)))

    def test_pushed_repo_is_never_pushed_again(self):
        with tempfile.TemporaryDirectory() as d:
            with RepoMemory(Path(d) / "m.db") as m:
                m.ensure_repo("stablyai/orca", stars=63000, now=NOW)
                self.assertFalse(m.is_pushed("stablyai/orca"))
                m.mark_pushed(["stablyai/orca"], now=NOW)
                self.assertTrue(m.is_pushed("stablyai/orca"))
                # still pushed months later — one push per repo, ever
                self.assertTrue(m.is_pushed("stablyai/orca", ))

    def test_star_delta_needs_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            with RepoMemory(Path(d) / "m.db") as m:
                m.ensure_repo("acme/surge", stars=1000, now=NOW)
                self.assertIsNone(m.stars_delta("acme/surge", now=NOW))  # cold start: no baseline
                m.ensure_repo("acme/surge", stars=1300, now=NOW + timedelta(days=7))
                self.assertEqual(m.stars_delta("acme/surge", now=NOW + timedelta(days=7)), 300)

    def test_seed_is_idempotent_and_never_clobbers(self):
        with tempfile.TemporaryDirectory() as d:
            with RepoMemory(Path(d) / "m.db") as m:
                seed = {"stablyai/orca": "2026-08-27", "HKUDS/nanobot": "2026-08-27"}
                self.assertEqual(m.load_seed(seed, now=NOW), 2)
                # re-seed does nothing
                self.assertEqual(m.load_seed(seed, now=NOW), 0)
                # the one-push-per-repo gate blocks seeded repos immediately —
                # even while they are still inside the 14d first-seen window
                self.assertTrue(m.is_pushed("stablyai/orca"))
                self.assertFalse(m.is_emerging("HKUDS/nanobot", now=NOW + timedelta(days=20)))

    def test_prune_keeps_repo_rows_forever(self):
        with tempfile.TemporaryDirectory() as d:
            with RepoMemory(Path(d) / "m.db") as m:
                m.ensure_repo("acme/keep", stars=5, now=NOW - timedelta(days=60))
                m.prune_snapshots(keep_days=30, now=NOW)
                # star snapshot pruned, repo row (first_seen) survives
                self.assertIsNone(m.stars_delta("acme/keep", now=NOW))
                self.assertFalse(m.is_emerging("acme/keep", now=NOW))  # still remembers first_seen


if __name__ == "__main__":
    unittest.main()
