"""Replay validation (Q5-B, 2026-09-07): prove the redesigned filters block
the repos that were actually pushed 8/25–9/7 (data from Actions logs).

Simulates the pipeline stage order exactly: fetch (live API, today's world)
→ seed memory → observe repos → apply pushed/emerging gates → report.
"""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexusnews.config import Source
from nexusnews.fetchers import PlatformFetcher, UrlLibTransport
from nexusnews.memory import RepoMemory

NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
SEED = json.loads(Path(__file__).parent.parent.joinpath("var/repo-memory-seed.json").read_text(encoding="utf-8"))

# 1. live fetch with the new dual-radar github_search
fetcher = PlatformFetcher(UrlLibTransport())
source = Source(name="GitHub Agent Search", kind="github_search",
                query="agent OR coding OR copilot OR MCP OR tool-use", limit=15)
raw_items = fetcher.fetch(source)
print(f"fetched {len(raw_items)} repos from dual radar (newborn + surge)")
for r in raw_items:
    print(f"  - {r.title.split(':')[0]}")

# 2. fresh memory with seed (simulating Actions cache restore + bootstrap)
with tempfile.TemporaryDirectory() as d:
    mem_path = Path(d) / "repo-memory.db"
    with RepoMemory(mem_path) as mem:
        inserted = mem.load_seed(SEED, now=NOW)
        print(f"\nseeded {inserted} historical pushes into fresh memory")

        blocked_pushed, blocked_old, kept = [], [], []
        for r in raw_items:
            repo = r.title.split(":", 1)[0].lower()
            stars = None
            if r.content:
                import re as _re
                m = _re.search(r"stars:(\d+)", r.content)
                stars = int(m.group(1)) if m else None
            mem.ensure_repo(repo, stars=stars, now=NOW)
            if mem.is_pushed(repo):
                blocked_pushed.append(repo)
            elif not mem.is_emerging(repo, now=NOW):
                blocked_old.append(repo)
            else:
                kept.append(repo)

        print(f"\n=== VERDICT ===")
        print(f"blocked by one-push gate (was pushed 8/25-9/7): {sorted(blocked_pushed)}")
        print(f"blocked by emerging gate (first_seen >14d):     {sorted(blocked_old)}")
        print(f"KEPT (fresh repos for today's digest):           {sorted(kept)}")

        # hard assertions — the historical repeat offenders must all be gone
        must_block = {r.lower() for r in SEED}
        leaked = must_block & set(kept)
        print(f"\nseeded repos leaking through gates: {sorted(leaked) or 'NONE ✓'}")
        assert not leaked, f"REPLAY FAILED: {leaked} would be re-pushed!"
        assert "stablyai/orca" in {r.lower() for r in blocked_pushed} or "stablyai/orca" in {r.lower() for r in blocked_old} or "stablyai/orca" not in [r.title.split(':')[0].lower() for r in raw_items], "orca unaccounted"
        print("REPLAY PASSED ✓")
