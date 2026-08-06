from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Callable

from .config import Config
from .delivery import send_feishu, send_feishu_card, send_feishu_chat, send_feishu_dm
from .digest import (DigestEntry, cutoff, filter_entries, local_summarize, render_card, render_digest,
                     render_empty_digest, select_items)
from .fetchers import PlatformFetcher, Transport
from .models import Item, normalize_item
from .llm import OpenAICompatibleSummarizer, with_fallback
from .storage import SQLiteItemStore


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
        chat_id = config.feishu_chat_id
        if not chat_id:
            raise RuntimeError("card_chat delivery requires feishu_chat_id in config")
        send_feishu_card(card_json, chat_id, "chat_id",
                         app_id_env=config.feishu_app_id_env,
                         app_secret_env=config.feishu_app_secret_env)
    elif config.delivery_mode == "card_dm":
        open_id = config.feishu_open_id or os.environ.get(config.feishu_open_id_env, "")
        if not open_id:
            raise RuntimeError(f"card_dm delivery requires feishu_open_id in config or {config.feishu_open_id_env} env var")
        send_feishu_card(card_json, open_id, "open_id",
                         app_id_env=config.feishu_app_id_env,
                         app_secret_env=config.feishu_app_secret_env)


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
        recent = store.recent(since=cutoff(hours=48, now=now))
    
    # Select generous candidate pool (3x maximum) for scoring before filtering
    candidate_pool_size = config.maximum * 3
    candidates = select_items(recent, minimum=config.minimum, maximum=candidate_pool_size)
    
    card_mode = config.delivery_mode in ("card_dm", "card_chat")
    if summarizer is None:
        summarizer = (with_fallback(OpenAICompatibleSummarizer(config.llm_endpoint, config.llm_model, config.llm_api_key_env))
                      if config.llm_endpoint and config.llm_model else local_summarize)
    
    # Summarize all candidates to get relevance scores
    all_entries = [summarizer(item) for item in candidates]
    
    # Filter by business relevance (≥6) and re-rank by score
    entries = filter_entries(all_entries, maximum=config.maximum, min_relevance=6)
    selected = [item for item in candidates if any(e.item_id == item.id for e in entries)]
    if len(entries) < config.minimum:
        text = render_empty_digest(generated_at=now, failed_sources=failed_sources)
    elif card_mode:
        text = render_card(entries, generated_at=now, failed_sources=failed_sources)
    else:
        text = render_digest(entries, generated_at=now, failed_sources=failed_sources)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    logging.info("digest generated", extra={"fetched": len(fetched), "inserted": inserted, "candidates": len(candidates), "scored": len(all_entries), "selected": len(entries)})
    if not dry_run:
        if card_mode:
            _deliver_card(config, text)
        else:
            _deliver_text(config, text)
        with SQLiteItemStore(database) as store:
            store.mark_delivered(item.id for item in selected)
    return text
