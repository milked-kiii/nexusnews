from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Callable

from .config import Config
from .delivery import send_feishu, send_feishu_card, send_feishu_card_to_chats, send_feishu_chat, send_feishu_dm
from .digest import (DigestEntry, cutoff, filter_entries, local_summarize, render_card, render_digest,
                     render_empty_digest, select_items)
from .feishu_doc import sync_digest_to_doc, sync_entries_to_doc
from .fetchers import PlatformFetcher, Transport
from .memory import RepoMemory, load_seed_file
from .models import Item, RawItem, normalize_item
from .llm import OpenAICompatibleSummarizer, with_fallback
from .storage import SQLiteItemStore

# ── Agent relevance keyword filter ──────────────────────────────
# Hard filter applied BEFORE LLM scoring to reduce noise and API calls.
# Items must hit at least one positive keyword and must NOT hit any negative
# keyword to proceed to summarization.

_AGENT_POSITIVE = {
    "agent", "coding", "code", "copilot", "cursor", "claude code", "mcp",
    "tool use", "function calling", "multi-agent", "ide", "autocomplete",
    "llm", "大模型", "代码生成", "编程助手", "智能体", "自动化", "aider",
    "devin", "windsurf", "trae", "qoder", "codex", "openai codex",
    "github copilot", "claude", "gpt", "gemini", "deepseek", "qwen",
    "kimi", "minimax", "zhipu", "huggingface", "ollama", "vllm",
    "langchain", "llamaindex", "crewai", "autogen", "swarm",
}

_AGENT_NEGATIVE = {
    "image generation", "video generation", "diffusion", "绘画", "写真",
    "语音", "tts", "text-to-speech", "自动驾驶", "autonomous driving",
    "医疗诊断", "medical diagnosis", "机器人", "robotics", "无人机",
    "drone", "算命", "占卜", "娱乐", "game", " gaming",
}


def _is_agent_relevant(item: Item) -> bool:
    """Return True if the item passes the hard keyword filter."""
    text = f"{item.title} {item.content or ''}".lower()
    # Negative filter: immediate reject
    for kw in _AGENT_NEGATIVE:
        if kw in text:
            return False
    # Positive filter: need at least one hit
    for kw in _AGENT_POSITIVE:
        if kw in text:
            return True
    return False


# ── source priority helpers ─────────────────────────────────────

# Reddit subreddits in priority order (most relevant first)
_REDDIT_PRIORITY = (
    "LocalLLaMA", "MachineLearning", "ClaudeAI", "OpenAI", "programming", "artificial",
)


def _source_sort_key(entry: DigestEntry) -> tuple[int, int, str]:
    """Sort key for digest entries: Reddit (by subreddit priority) -> GitHub -> domestic.

    Returns (tier, sub_priority, source_name) so entries within the same tier
    are ordered by relevance, then subreddit priority, then alphabetically.
    """
    source = entry.source.lower()
    if source.startswith("reddit"):
        # Extract subreddit name: "Reddit LocalLLaMA" -> "LocalLLaMA"
        sub = entry.source.replace("Reddit ", "", 1)
        try:
            sub_idx = _REDDIT_PRIORITY.index(sub)
        except ValueError:
            sub_idx = len(_REDDIT_PRIORITY)
        return (0, sub_idx, entry.source)
    if source.startswith("github"):
        return (1, 0, entry.source)
    return (2, 0, entry.source)


def _is_primary_source(source_name: str) -> bool:
    """Reddit and GitHub sources are primary; domestic sources are secondary."""
    s = source_name.lower()
    return s.startswith("reddit") or s.startswith("github")


def _apply_primary_quota(
    entries: list[DigestEntry],
    minimum: int,
    maximum: int,
    quota: float,
) -> list[DigestEntry]:
    """Ensure at least `quota` fraction of final entries come from primary sources,
    and at least `min_per_source` entries each from Reddit and GitHub when available.

    If not enough primary entries exist, backfill with secondary sources but
    never exceed `maximum`. If the day is thin on primary sources, we allow
    a smaller digest rather than padding with low-quality domestic content.

    Entries are sorted by source tier (Reddit -> GitHub -> domestic), then by
    relevance score within each tier.
    """
    # Sort by score first, then by source tier within same score
    # This ensures high-score GitHub items aren't buried under low-score Reddit
    sorted_entries = sorted(entries, key=lambda e: (-e.relevance_score, _source_sort_key(e)))

    primary = [e for e in sorted_entries if _is_primary_source(e.source)]
    secondary = [e for e in sorted_entries if not _is_primary_source(e.source)]

    # Split primary by source type
    reddit = [e for e in primary if e.source.lower().startswith("reddit")]
    github = [e for e in primary if e.source.lower().startswith("github")]

    # Guarantee at least min_per_source from each primary type when available
    min_per_source = 2
    selected: list[DigestEntry] = []

    # Take top min_per_source from each if available
    take_reddit = min(min_per_source, len(reddit), maximum)
    take_github = min(min_per_source, len(github), maximum - take_reddit)
    selected.extend(reddit[:take_reddit])
    selected.extend(github[:take_github])

    # Fill remaining slots from either primary type, alternating by score
    remaining_primary = [e for e in sorted_entries if e in primary and e not in selected]
    for e in remaining_primary:
        if len(selected) >= maximum:
            break
        selected.append(e)

    # Fill remaining slots with secondary, respecting the quota
    remaining_slots = maximum - len(selected)
    if remaining_slots > 0 and secondary:
        # How many secondary can we take while keeping primary fraction >= quota?
        max_secondary = int(len(selected) * (1 - quota) / quota) if quota > 0 else remaining_slots
        take_secondary = min(remaining_slots, max_secondary, len(secondary))
        selected.extend(secondary[:take_secondary])

    # If we still don't have minimum, and we have secondary left, relax quota
    if len(selected) < minimum and len(selected) < maximum:
        remaining = maximum - len(selected)
        extra = [e for e in secondary if e not in selected]
        selected.extend(extra[:remaining])

    return selected[:maximum]


Summarizer = Callable[[Item], DigestEntry]


def _deliver_text(config: Config, text: str) -> None:
    if config.delivery_mode == "chat":
        chat_id = config.feishu_chat_id
        if not chat_id:
            raise RuntimeError("chat delivery requires feishu_chat_id in config")
        send_feishu_chat(text, chat_id, app_id_env=config.feishu_app_id_env,
                         app_secret_env=config.feishu_app_secret_env)
    elif config.delivery_mode == "dm":
        open_id = config.feishu_open_id or os.environ.get(config.feishu_open_id_env, "")
        if not open_id:
            raise RuntimeError(f"DM delivery requires feishu_open_id in config or {config.feishu_open_id_env} env var")
        send_feishu_dm(text, open_id, app_id_env=config.feishu_app_id_env,
                       app_secret_env=config.feishu_app_secret_env)
    else:
        webhook = os.environ.get(config.webhook_env)
        if not webhook:
            raise RuntimeError(f"missing required environment variable: {config.webhook_env}")
        send_feishu(webhook, text)


def _deliver_card(config: Config, card_json: str) -> None:
    if config.delivery_mode == "card_chat":
        # Resolve chat IDs: prefer feishu_chat_ids, fall back to single feishu_chat_id
        if config.feishu_chat_ids:
            chat_ids = [c for c in config.feishu_chat_ids if c]
        elif config.feishu_chat_id:
            chat_ids = [config.feishu_chat_id]
        else:
            raise RuntimeError("card_chat delivery requires feishu_chat_ids or feishu_chat_id in config")
        if not chat_ids:
            raise RuntimeError("card_chat delivery requires at least one non-empty chat_id")
        send_feishu_card_to_chats(card_json, chat_ids,
                                  app_id_env=config.feishu_app_id_env,
                                  app_secret_env=config.feishu_app_secret_env)
    elif config.delivery_mode == "card_dm":
        open_id = config.feishu_open_id or os.environ.get(config.feishu_open_id_env, "")
        if not open_id:
            raise RuntimeError(f"card_dm delivery requires feishu_open_id in config or {config.feishu_open_id_env} env var")
        send_feishu_card(card_json, open_id, "open_id",
                         app_id_env=config.feishu_app_id_env,
                         app_secret_env=config.feishu_app_secret_env)


# ── repo memory helpers ──────────────────────────────────────────
# GitHub repos get cross-run state (first_seen / pushed / star history) from
# RepoMemory. Items carry the repo key in the content prefix added by
# parse_github_search: "repo:owner/name created:... stars:...".

_REPO_PREFIX = re.compile(r"^repo:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def _repo_key(item: Item) -> str | None:
    """Extract owner/repo from a GitHub item's content prefix, if any."""
    if not item.content or not item.source.lower().startswith("github"):
        return None
    m = _REPO_PREFIX.match(item.content)
    return m.group(1).lower() if m else None


def _repo_stars_of(item: Item) -> int | None:
    if not item.content:
        return None
    m = re.match(r"repo:\S+\s+created:\S+\s+stars:(\d+)", item.content)
    return int(m.group(1)) if m else None


def run(config: Config, transport: Transport, *, dry_run: bool, now: datetime | None = None, summarizer: Summarizer | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    database = Path(config.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    fetched = []
    failed_sources = 0
    fetcher = PlatformFetcher(transport)
    for source in config.sources:
        try:
            fetched.extend(fetcher.fetch(source))
        except Exception:
            failed_sources += 1
            logging.exception("source fetch failed", extra={"source": source.name})
    if not fetched:
        raise RuntimeError("all configured sources failed or returned no items")
    with SQLiteItemStore(database) as store:
        inserted = store.put_many(normalize_item(raw) for raw in fetched)
        # Recency window from config (default 96h for weekly-publish sources;
        # tighten to 48h or less when hot-list sources like GitHub/Reddit
        # hot are configured and freshness matters more).
        recent = store.recent(since=cutoff(hours=config.window_hours, now=now))

    # ── GitHub repo memory (cross-run, persisted via actions/cache) ──
    # Sequence matters: observe first (fixes first_seen for every repo seen
    # today, pushed or not), then drop repos that were already pushed
    # (one push per repo, ever) or that stopped being "emerging" (first
    # seen more than 14 days ago). Star deltas are computed BEFORE the
    # pushed-filter so a re-pushed repo could still show growth (it won't:
    # pushed repos are dropped) — the delta is injected into LLM context.
    memory_path = Path(config.memory_db)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with RepoMemory(memory_path) as memory:
        # Bootstrap from the seed file (idempotent; no-op once live data exists)
        if config.memory_seed:
            seed = load_seed_file(config.memory_seed)
            if seed:
                memory.load_seed(seed, now=now)

        star_deltas: dict[str, int | None] = {}
        repos_observed: list[str] = []
        github_items: list[Item] = []
        for item in recent:
            key = _repo_key(item)
            if not key:
                continue
            memory.ensure_repo(key, stars=_repo_stars_of(item), now=now)
            repos_observed.append(key)
            star_deltas[key] = memory.stars_delta(key, now=now)
            github_items.append(item)

        pushed_or_old = {k for k in repos_observed
                         if memory.is_pushed(k) or not memory.is_emerging(k, now=now)}
        if pushed_or_old:
            dropped = [item for item in github_items if _repo_key(item) in pushed_or_old]
            logging.info("repo memory dropped already-pushed/stale repos",
                         extra={"repos": sorted(pushed_or_old), "dropped_items": len(dropped)})
        recent = [item for item in recent if _repo_key(item) not in pushed_or_old]

    # Hard keyword filter: drop non-Agent items before LLM scoring
    agent_relevant = [item for item in recent if _is_agent_relevant(item)]

    # Select generous candidate pool (6x maximum) for scoring before filtering.
    # Larger pool ensures slow-moving sources (Qoder, Cursor, Trae, 公众号) survive
    # the sort by recency that would otherwise push them below chatty feeds.
    candidate_pool_size = config.maximum * 6
    candidates = select_items(agent_relevant, minimum=config.minimum, maximum=candidate_pool_size)

    card_mode = config.delivery_mode in ("card_dm", "card_chat")
    if summarizer is None:
        if config.llm_endpoint and config.llm_model:
            summarizer = with_fallback(
                OpenAICompatibleSummarizer(
                    config.llm_endpoint, config.llm_model, config.llm_api_key_env,
                    vc_watchlist=config.vc_watchlist,
                ),
                vc_watchlist=config.vc_watchlist,
            )
        else:
            summarizer = lambda item: local_summarize(item, vc_watchlist=config.vc_watchlist)  # noqa: E731

    # Summarize all candidates to get relevance scores. For GitHub repos we
    # wrap the configured summarizer to inject the star-growth signal (from
    # repo memory) into the item content the LLM sees — the LLM stays
    # item-independent (Q4 decision: no push-history injection), it just
    # gets richer facts about the repo itself.
    def _with_growth(item: Item) -> DigestEntry:
        key = _repo_key(item)
        if key and item.content:
            delta = star_deltas.get(key)
            if delta is not None:
                growth_note = f"近7天新增⭐{delta}" if delta >= 0 else f"近7天⭐{delta}"
                enriched = Item(item.id, item.source, item.title, item.url,
                                f"{item.content} · {growth_note}",
                                item.published_at, item.dedupe_key)
                return summarizer(enriched)
        return summarizer(item)

    all_entries = [_with_growth(item) for item in candidates]

    # Filter by business relevance (≥6) and re-rank by score
    # NOTE: don't truncate to maximum yet — _apply_primary_quota needs the full
    # pool to ensure Reddit/GitHub representation. It handles final truncation.
    entries = filter_entries(all_entries, maximum=len(all_entries), min_relevance=6)

    # Apply primary-source quota: Reddit/GitHub must dominate the digest
    entries = _apply_primary_quota(
        entries, config.minimum, config.maximum, config.primary_source_quota,
    )

    selected = [item for item in candidates if any(e.item_id == item.id for e in entries)]
    if len(entries) < config.minimum:
        text = render_empty_digest(generated_at=now, failed_sources=failed_sources, minimum=config.minimum)
    elif card_mode:
        text = render_card(entries, generated_at=now, failed_sources=failed_sources,
                           window_hours=config.window_hours)
    else:
        text = render_digest(entries, generated_at=now, failed_sources=failed_sources,
                             window_hours=config.window_hours)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    logging.info("digest generated", extra={"fetched": len(fetched), "inserted": inserted, "candidates": len(candidates), "scored": len(all_entries), "selected": len(entries)})
    if not dry_run:
        if card_mode:
            # Optionally sync the digest to a Feishu cloud doc and append the
            # doc link to the card before delivery. Only when there is actual
            # content (empty digest has nothing worth archiving).
            doc_url = None
            if config.doc_sync and len(entries) >= config.minimum:
                open_id = config.feishu_open_id or os.environ.get(config.feishu_open_id_env, "")
                if open_id:
                    title = f"🤖 AI 日报 {now.astimezone(timezone.utc).strftime('%Y-%m-%d')}"
                    doc = sync_entries_to_doc(title, entries, open_id,
                                              app_id_env=config.feishu_app_id_env,
                                              app_secret_env=config.feishu_app_secret_env)
                    doc_url = doc.get("url")
                    logging.info("digest synced to doc", extra={"doc_url": doc_url})
            if doc_url:
                text = render_card(entries, generated_at=now, failed_sources=failed_sources,
                                   doc_url=doc_url, window_hours=config.window_hours)
            _deliver_card(config, text)
        else:
            _deliver_text(config, text)
        with SQLiteItemStore(database) as store:
            store.mark_delivered(item.id for item in selected)
        # Record pushed repos in the cross-run memory (one push per repo).
        # Runs even on empty digests so observation-only days still persist.
        pushed_repos = [k for k in (_repo_key(item) for item in selected) if k]
        with RepoMemory(memory_path) as memory:
            if pushed_repos:
                memory.mark_pushed(pushed_repos, now=now)
            memory.prune_snapshots(now=now)
    return text
