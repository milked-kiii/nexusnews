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
    feishu_open_id_env: str = "NEXUSNEWS_FEISHU_OPEN_ID"
    delivery_mode: str = "webhook"
    llm_endpoint: str | None = None
    llm_model: str | None = None
    llm_api_key_env: str = "NEXUSNEWS_LLM_API_KEY"


def load_config(path: str | Path) -> Config:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources = tuple(Source(**row) for row in data["sources"])
        config = Config(sources=sources, **{k: v for k, v in data.items() if k != "sources"})
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid config {path}: {exc}") from exc
    if not sources:
        raise ValueError("config must contain at least one source")
    supported_kinds = {"rss", "medium", "reddit", "x", "discord"}
    for source in sources:
        if source.kind not in supported_kinds:
            raise ValueError(f"unsupported source kind: {source.kind}")
        if not 1 <= source.limit <= 100:
            raise ValueError(f"source {source.name!r} limit must be between 1 and 100")
        if source.kind in {"rss", "medium", "reddit"} and not source.url:
            raise ValueError(f"source {source.name!r} requires url")
        if source.kind == "x" and (not source.query or not source.token_env):
            raise ValueError(f"X source {source.name!r} requires query and token_env")
        if source.kind == "discord" and (not source.channel_id or not source.token_env):
            raise ValueError(f"Discord source {source.name!r} requires channel_id and token_env")
    if not 1 <= config.minimum <= config.maximum <= 10:
        raise ValueError("config selection must satisfy 1 <= minimum <= maximum <= 10")
    if config.delivery_mode not in ("webhook", "dm", "chat", "card_dm", "card_chat"):
        raise ValueError("delivery_mode must be one of: webhook, dm, chat, card_dm, card_chat")
    if config.delivery_mode in ("dm", "card_dm") and not config.feishu_open_id:
        raise ValueError("feishu_open_id is required for DM delivery modes")
    if config.delivery_mode in ("chat", "card_chat") and not config.feishu_chat_id:
        raise ValueError("feishu_chat_id is required for chat delivery modes")
    if bool(config.llm_endpoint) != bool(config.llm_model):
        raise ValueError("llm_endpoint and llm_model must be configured together")
    return config
