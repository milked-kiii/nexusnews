from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    url: str | None = None
    kind: str = "rss"
    query: str | None = None
    channel_id: str | None = None
    token_env: str | None = None
    limit: int = 20
    # webpage kind options
    link_pattern: str | None = None
    title_group: str | None = None
    exclude_pattern: str | None = None
    # reddit kind options
    subreddit: str | None = None
    sort: str = "hot"
    min_score: int = 0


@dataclass(frozen=True)
class Competitor:
    """One watched competitor product for the 🧪 竞品实测 review line.

    ``name`` is the canonical display name (e.g. "Claude Code"). ``tier``
    maps to a strategic weight (1→1.0, 2→0.85, 3→0.7) that multiplies the
    LLM review-quality score so final ranking reflects both how important
    the competitor is to us and how good the review is.
    """

    name: str
    tier: int = 3
    aliases: tuple[str, ...] = ()

    @property
    def weight(self) -> float:
        return {1: 1.0, 2: 0.85, 3: 0.7}.get(self.tier, 0.7)


@dataclass(frozen=True)
class Config:
    sources: tuple[Source, ...]
    database: str = "var/nexusnews.db"
    output: str = "var/latest-digest.txt"
    minimum: int = 5
    maximum: int = 10
    webhook_env: str = "FEISHU_WEBHOOK_URL"
    feishu_app_id_env: str = "FEISHU_APP_ID"
    feishu_app_secret_env: str = "FEISHU_APP_SECRET"
    feishu_open_id: str | None = None
    feishu_chat_id: str | None = None
    feishu_chat_ids: tuple[str, ...] | None = None
    feishu_open_id_env: str = "NEXUSNEWS_FEISHU_OPEN_ID"
    delivery_mode: str = "webhook"
    doc_sync: bool = False
    llm_endpoint: str | None = None
    llm_model: str | None = None
    llm_api_key_env: str = "NEXUSNEWS_LLM_API_KEY"
    vc_watchlist: tuple[str, ...] = ()
    # Recency window for digest candidates. Slow blog/changelog sources need a
    # wide window; hot-list sources (GitHub, Reddit) stay fresh either way.
    window_hours: int = 96
    # Minimum fraction of final digest slots that must come from primary sources
    # (Reddit / GitHub). 0.5 means at least half the digest is Reddit/GitHub.
    primary_source_quota: float = 0.5
    # Cross-run GitHub repo memory (first_seen / pushed / star history).
    # Persisted across Actions runs via actions/cache on this path.
    memory_db: str = "var/repo-memory.db"
    # Optional seed JSON {repo: push_date} for repos pushed before the memory
    # existed (extracted from past Actions logs). Idempotent bootstrap.
    memory_seed: str | None = None
    # ── 竞品实测 (competitor review) line ─────────────────────────
    # Watchlist of competitors whose hands-on reviews /评测 are collected
    # from Reddit search + YouTube search. Each entry's tier maps to a
    # strategic weight (see Competitor.weight) applied on top of the LLM's
    # review-quality score.
    competitor_watchlist: tuple[Competitor, ...] = ()
    # Shared slot cap for the radar zone: GitHub emerging repos + competitor
    # reviews combined never exceed this (user: "github的东西和竞品加起来4条").
    radar_cap: int = 4
    # Review line uses its own recency window (reviews stay valuable for days,
    # unlike 24h news). 168h = 7 days, matching the user's "评测放宽到7天".
    review_window_hours: int = 168
    # Cross-run competitor-review memory (pushed URLs + pushed versions),
    # persisted across Actions runs via actions/cache like repo memory.
    review_memory_db: str = "var/review-memory.db"


def load_config(path: str | Path) -> Config:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources = tuple(Source(**row) for row in data["sources"])
        top_level = {k: v for k, v in data.items() if k != "sources"}
        if "vc_watchlist" in top_level:
            top_level["vc_watchlist"] = tuple(top_level["vc_watchlist"])
        if "feishu_chat_ids" in top_level:
            top_level["feishu_chat_ids"] = tuple(top_level["feishu_chat_ids"])
        if "competitor_watchlist" in top_level:
            raw = top_level.pop("competitor_watchlist")
            competitors = []
            for item in raw:
                if isinstance(item, dict):
                    aliases = item.get("aliases") or []
                    if isinstance(aliases, str):
                        aliases = [aliases]
                    competitors.append(Competitor(
                        name=item.get("name", ""),
                        tier=int(item.get("tier", 3)),
                        aliases=tuple(str(a) for a in aliases),
                    ))
                else:
                    competitors.append(Competitor(name=str(item)))
            top_level["competitor_watchlist"] = tuple(competitors)
        config = Config(sources=sources, **top_level)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid config {path}: {exc}") from exc
    if not sources:
        raise ValueError("config must contain at least one source")
    supported_kinds = {"rss", "medium", "reddit", "github", "github_trending", "github_search", "x", "discord", "webpage",
                       "reddit_search", "youtube_search"}
    for source in sources:
        if source.kind not in supported_kinds:
            raise ValueError(f"unsupported source kind: {source.kind}")
        if not 1 <= source.limit <= 100:
            raise ValueError(f"source {source.name!r} limit must be between 1 and 100")
        if source.kind in {"rss", "medium"} and not source.url:
            raise ValueError(f"source {source.name!r} requires url")
        if source.kind == "reddit":
            if not source.subreddit and not source.url:
                raise ValueError(f"reddit source {source.name!r} requires subreddit or url")
        if source.kind == "webpage":
            if not source.url:
                raise ValueError(f"webpage source {source.name!r} requires url")
            if not source.link_pattern:
                raise ValueError(f"webpage source {source.name!r} requires link_pattern")
        if source.kind == "x" and (not source.query or not source.token_env):
            raise ValueError(f"X source {source.name!r} requires query and token_env")
        if source.kind == "discord" and (not source.channel_id or not source.token_env):
            raise ValueError(f"Discord source {source.name!r} requires channel_id and token_env")
        if source.kind == "reddit_search" and not source.query:
            raise ValueError(f"reddit_search source {source.name!r} requires query")
        if source.kind == "youtube_search" and not source.query:
            raise ValueError(f"youtube_search source {source.name!r} requires query")
    for competitor in config.competitor_watchlist:
        if not competitor.name:
            raise ValueError("competitor_watchlist entries must have a name")
        if competitor.tier not in (1, 2, 3):
            raise ValueError(f"competitor {competitor.name!r} tier must be 1, 2 or 3")
    if not 0 <= config.radar_cap <= 10:
        raise ValueError("radar_cap must be between 0 and 10")
    if not 1 <= config.review_window_hours <= 24 * 14:
        raise ValueError("review_window_hours must be between 1 and 336 (14 days)")
    if not 1 <= config.minimum <= config.maximum <= 10:
        raise ValueError("config selection must satisfy 1 <= minimum <= maximum <= 10")
    if not 1 <= config.window_hours <= 24 * 14:
        raise ValueError("window_hours must be between 1 and 336 (14 days)")
    if config.delivery_mode not in ("webhook", "dm", "chat", "card_dm", "card_chat"):
        raise ValueError("delivery_mode must be one of: webhook, dm, chat, card_dm, card_chat")
    if config.delivery_mode in ("dm", "card_dm") and not config.feishu_open_id:
        raise ValueError("feishu_open_id is required for DM delivery modes")
    if config.delivery_mode in ("chat", "card_chat"):
        if not config.feishu_chat_id and not config.feishu_chat_ids:
            raise ValueError("feishu_chat_id or feishu_chat_ids is required for chat delivery modes")
    if bool(config.llm_endpoint) != bool(config.llm_model):
        raise ValueError("llm_endpoint and llm_model must be configured together")
    if config.doc_sync and not config.feishu_open_id:
        raise ValueError("feishu_open_id is required when doc_sync is enabled (doc permission grant)")
    return config
