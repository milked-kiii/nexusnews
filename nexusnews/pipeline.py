from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Callable

from .config import Config, Competitor
from .delivery import send_feishu, send_feishu_card, send_feishu_card_to_chats, send_feishu_chat, send_feishu_dm
from .digest import (DigestEntry, cutoff, filter_entries, local_summarize, render_card, render_digest,
                     render_empty_digest, select_items)
from .feishu_doc import sync_digest_to_doc, sync_entries_to_doc
from .fetchers import (PlatformFetcher, Transport, fetch_reddit_search, fetch_youtube_search)
from .memory import RepoMemory, ReviewMemory, load_seed_file
from .models import Item, RawItem, normalize_item
from .llm import OpenAICompatibleSummarizer, ReviewScorer, with_fallback
from .storage import SQLiteItemStore

# 🧪 竞品实测 source tag prefix. Review RawItems carry
# source = "竞品实测|{competitor}|{platform}" so the pipeline can route them
# to the review pool and the renderer can show a distinct tier.
REVIEW_SOURCE_PREFIX = "竞品实测"

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


# ── 🧪 竞品实测 (competitor review) helpers ─────────────────────
# The review line collects hands-on reviews of watched competitors from
# Reddit search + YouTube search, scores them with a DEDICATED LLM pass
# (is_hands_on_review gate + quality × tier weight), and dedupes at the
# version level (same competitor + same version → push once; version upgrade
# restarts the counter). Reviews share the radar slot cap with GitHub repos.

_REVIEW_POSITIVE = (
    "review", "tried", "using", "experience", "hands-on", "thoughts",
    "impressions", "switched", "moved to", "week with", "days with",
    "评测", "实测", "体验", "使用心得", "用了", "对比",
)
# Only high-confidence pure-announcement phrases are rejected here — the
# coarse filter must NOT out-recall the LLM's is_hands_on_review gate.
# Words like "release"/"launch" appear inside genuine reviews ("after the
# 3.8 release, I tried…") so they are deliberately NOT in this list.
_REVIEW_NEGATIVE = (
    "announce", "introducing", "introduces", "now available", "official launch",
    "官宣", "正式发布", "正式推出", "webinar", "register now", "sign up",
)


def _is_review_source(source_name: str) -> bool:
    return source_name.startswith(REVIEW_SOURCE_PREFIX)


def _review_competitor_of(source_name: str) -> str:
    """Extract competitor name from "竞品实测|{competitor}|{platform}"."""
    parts = source_name.split("|")
    return parts[1] if len(parts) >= 2 else source_name


def _is_review_candidate(item: Item) -> bool:
    """Cheap keyword pre-filter BEFORE the LLM review pass, so we don't send
    every Reddit/YouTube search hit to the LLM. Announcements / release notes
    are rejected here (user Q2-B: 纯公告/转发不算)."""
    text = f"{item.title} {item.content or ''}".lower()
    if any(neg in text for neg in _REVIEW_NEGATIVE):
        return False
    return any(pos in text for pos in _REVIEW_POSITIVE)


def _fetch_review_raws(transport: Transport, watchlist: tuple[Competitor, ...],
                       timeout: float = 15.0) -> tuple[list[RawItem], int]:
    """Fetch competitor reviews from Reddit search + YouTube search.

    Returns (raw_items, failed_sources). Each RawItem's source is
    "竞品实测|{competitor}|{platform}" so it routes to the review pool.
    A failed search for one competitor is logged, not fatal — the rest of
    the line (and the news line) still runs.

    Query syntax differs per platform: Reddit's search supports boolean
    OR ("Claude Code" OR "claude-code"); YouTube does NOT — yt-dlp passes
    the string verbatim, so OR/quote syntax just degrades the match. Use
    the plain competitor name for YouTube, and give it a much longer
    timeout (full metadata extraction for N results takes ~3-4s per video,
    vs 15s politeness-throttle for a single Reddit RSS fetch).
    """
    raws: list[RawItem] = []
    failed = 0
    for competitor in watchlist:
        queries = [competitor.name]
        queries.extend(a for a in competitor.aliases if a)
        reddit_query = " OR ".join(f'"{q}"' for q in queries[:4])
        youtube_query = competitor.name  # plain phrase; YouTube has no boolean search
        for platform, fetch in (
            ("Reddit", lambda q=reddit_query: fetch_reddit_search(transport, q, source="", limit=12, timeout=timeout)),
            ("YouTube", lambda q=youtube_query: fetch_youtube_search(q, source="", limit=6, timeout=90.0)),
        ):
            source = f"{REVIEW_SOURCE_PREFIX}|{competitor.name}|{platform}"
            try:
                items = fetch()
                for raw in items:
                    raws.append(RawItem(
                        source=source,
                        title=raw.title,
                        url=raw.url,
                        content=raw.content,
                        published_at=raw.published_at,
                        external_id=raw.external_id,
                    ))
            except Exception:
                failed += 1
                logging.exception("competitor review fetch failed",
                                  extra={"competitor": competitor.name, "platform": platform})
    return raws, failed


def _score_review_pool(review_items: list[Item], config: Config,
                       scorer: ReviewScorer, memory: ReviewMemory, *,
                       now: datetime) -> list[DigestEntry]:
    """Run the dedicated review scoring pass and version-level dedup.

    - Local keyword pre-filter (cheap, drops announcements)
    - LLM: is_hands_on_review gate → None drops the item entirely
    - final relevance_score = quality × tier_weight (user Q8-A)
    - ReviewMemory: same competitor + same version already pushed → drop;
      URL already pushed → drop. Version extraction happens in the LLM.
    - In-batch version dedup: same (competitor, version) candidates in THIS
      run keep only the highest-scoring one (the user's rule "同版本第二篇
      不算" applies within a single digest too — the memory's mark_pushed
      only happens after delivery, so it can't dedup the current batch).
    Returns entries with relevance_score >= 6, sorted desc.
    """
    competitor_weight = {c.name: c.weight for c in config.competitor_watchlist}
    scored: list[DigestEntry] = []
    for item in review_items:
        competitor = _review_competitor_of(item.source)
        weight = competitor_weight.get(competitor, 0.7)
        if not _is_review_candidate(item):
            continue
        try:
            entry = scorer(item, competitor=competitor, weight=weight)
        except Exception:
            logging.exception("review scoring failed; review dropped",
                              extra={"competitor": competitor, "title": item.title})
            continue
        if entry is None or entry.relevance_score < 6:
            continue
        memory.observe(competitor, entry.review_version or "", entry.url, now=now)
        if memory.is_url_pushed(entry.url):
            continue
        # Cross-run version gate: this (competitor, version) event was pushed
        # on a previous day — the new review of the same version doesn't count
        # (user Q10). In-batch dedup below handles same-day duplicates.
        version = entry.review_version or ""
        if version and memory.is_version_pushed(competitor, version):
            continue
        scored.append(entry)
    # In-batch version dedup: keep the best entry per (competitor, version).
    best_by_event: dict[tuple[str, str], DigestEntry] = {}
    for entry in scored:
        key = (entry.competitor or "", entry.review_version or "")
        current = best_by_event.get(key)
        if current is None or entry.relevance_score > current.relevance_score:
            best_by_event[key] = entry
    result = list(best_by_event.values())
    result.sort(key=lambda e: e.relevance_score, reverse=True)
    return result


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

    # 🧪 竞品实测 review line: dynamic per-competitor searches (Reddit search
    # RSS + YouTube). Failures count toward failed_sources but never abort.
    review_raws, review_failures = [], 0
    if config.competitor_watchlist:
        review_raws, review_failures = _fetch_review_raws(transport, config.competitor_watchlist)
        failed_sources += review_failures
        fetched.extend(review_raws)

    if not fetched:
        raise RuntimeError("all configured sources failed or returned no items")
    with SQLiteItemStore(database) as store:
        inserted = store.put_many(normalize_item(raw) for raw in fetched)
        # Main news pool: 24h window. Review pool: its own window (default 7d).
        # The two pools share the items DB but never cross-contaminate.
        main_recent = [i for i in store.recent(since=cutoff(hours=config.window_hours, now=now))
                       if not _is_review_source(i.source)]
        review_recent = store.recent(since=cutoff(hours=config.review_window_hours, now=now),
                                     source_prefix=REVIEW_SOURCE_PREFIX)

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
        for item in main_recent:
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
        main_recent = [item for item in main_recent if _repo_key(item) not in pushed_or_old]

    # Hard keyword filter: drop non-Agent items before LLM scoring
    agent_relevant = [item for item in main_recent if _is_agent_relevant(item)]

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
    # NOTE: don't truncate to maximum yet — the quota step needs the full
    # pool. filter_entries here only filters; truncation happens per pool.
    entries = filter_entries(all_entries, maximum=len(all_entries), min_relevance=6,
                             ensure_top_fire=0)

    # ── 双池拆分（user 2026-09-08 拍板）──────────────────────────
    # 主池 = Reddit + 国内（GitHub 移出；GitHub 保底≥2 撤掉，见确认①）
    # 雷达池 = GitHub 新兴项目 + 🧪 竞品实测 合并，cap=radar_cap（≤4，无下限）
    github_entries = [e for e in entries if e.source.lower().startswith("github")]
    main_entries = [e for e in entries if not e.source.lower().startswith("github")]

    # 主池：保持原 quota 语义（Reddit≥2 保底 + 国内填充，3-5 条）
    main_selected = _apply_primary_quota(
        main_entries, config.minimum, config.maximum, config.primary_source_quota,
    )

    # 雷达池：竞品评测（ReviewMemory 版本级去重）+ GitHub 按分混合竞争
    review_entries: list[DigestEntry] = []
    review_memory_path = Path(config.review_memory_db)
    review_memory_path.parent.mkdir(parents=True, exist_ok=True)
    if config.competitor_watchlist and review_recent:
        # Per-competitor cap: at most `review_cap` candidates per competitor
        # (both platforms) so the LLM review pass stays bounded even when a
        # big competitor has many fresh hits.
        review_cap = max(2, config.radar_cap)
        by_competitor: dict[str, list[Item]] = {}
        for item in review_recent:
            by_competitor.setdefault(_review_competitor_of(item.source), []).append(item)
        review_items: list[Item] = []
        for competitor, items in by_competitor.items():
            ranked = sorted(items, key=lambda i: (i.published_at or ""), reverse=True)
            review_items.extend(ranked[:review_cap])
        review_items = select_items(review_items, minimum=1,
                                    maximum=len(review_items),
                                    per_source_cap=review_cap)
        if config.llm_endpoint and config.llm_model:
            review_scorer = ReviewScorer(config.llm_endpoint, config.llm_model,
                                         config.llm_api_key_env)
            with ReviewMemory(review_memory_path) as review_memory:
                review_entries = _score_review_pool(review_items, config, review_scorer,
                                                    review_memory, now=now)
        else:
            logging.warning("no LLM configured; competitor review line skipped")

    radar_pool = sorted(github_entries + review_entries, key=lambda e: e.relevance_score, reverse=True)
    # 评测保底：GitHub 天天有高分条目，纯按分混排会让评测永远进不了日报
    # （orca 霸榜问题的翻版）。若当天有 ≥6 分评测，雷达池保证 1 席给评测，
    # 剩余席位 GitHub + 其余评测按分竞争（评测内部仍是 quality×tier 排序）。
    radar_selected: list[DigestEntry] = []
    if review_entries:
        radar_selected.append(max(review_entries, key=lambda e: e.relevance_score))
    for entry in radar_pool:
        if len(radar_selected) >= config.radar_cap:
            break
        if entry in radar_selected:
            continue
        radar_selected.append(entry)

    # 合并双池；空日报判定按总条数（雷达区可补主池缺口）
    entries = main_selected + radar_selected
    if entries and all(e.relevance_score < 9 for e in entries):
        # promote top-1 to 🔥 so the card always has a headline tier
        from dataclasses import replace
        top_idx = max(range(len(entries)), key=lambda i: entries[i].relevance_score)
        entries = [replace(e, relevance_score=9) if i == top_idx else e
                   for i, e in enumerate(entries)]

    selected = [item for item in candidates if any(e.item_id == item.id for e in entries)]
    # review items aren't in `candidates` — track them separately for delivery
    review_selected_ids = {e.item_id for e in radar_selected if e.category == "review"}
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
    logging.info("digest generated", extra={"fetched": len(fetched), "inserted": inserted, "candidates": len(candidates), "scored": len(all_entries), "selected": len(entries), "reviews": len(review_entries)})
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
            store.mark_delivered(review_selected_ids)
        # Record pushed repos in the cross-run memory (one push per repo).
        # Runs even on empty digests so observation-only days still persist.
        pushed_repos = [k for k in (_repo_key(item) for item in selected) if k]
        with RepoMemory(memory_path) as memory:
            if pushed_repos:
                memory.mark_pushed(pushed_repos, now=now)
            memory.prune_snapshots(now=now)
        # Record pushed reviews in the cross-run review memory (version-level:
        # same competitor + same version pushes once ever; URL once ever).
        with ReviewMemory(review_memory_path) as review_memory:
            for entry in radar_selected:
                if entry.category != "review":
                    continue
                review_memory.mark_pushed(entry.competitor or "",
                                          entry.review_version or "",
                                          entry.url, now=now)
    return text
