from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Callable

from .config import Config
from .delivery import send_feishu
from .digest import DigestEntry, cutoff, local_summarize, render_digest, render_empty_digest, select_items
from .fetchers import RSSFetcher, Transport
from .models import Item, normalize_item
from .llm import OpenAICompatibleSummarizer, with_fallback
from .storage import SQLiteItemStore


Summarizer = Callable[[Item], DigestEntry]


def run(config: Config, transport: Transport, *, dry_run: bool, now: datetime | None = None, summarizer: Summarizer | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    database = Path(config.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    fetched = []
    failed_sources = 0
    fetcher = RSSFetcher(transport)
    for source in config.sources:
        try:
            fetched.extend(fetcher.fetch(source.url, source=source.name))
        except Exception:
            failed_sources += 1
            logging.exception("source fetch failed", extra={"source": source.name})
    if not fetched:
        raise RuntimeError("all configured sources failed or returned no items")
    with SQLiteItemStore(database) as store:
        inserted = store.put_many(normalize_item(raw) for raw in fetched)
        recent = store.recent(since=cutoff(now=now))
    selected = select_items(recent, minimum=config.minimum, maximum=config.maximum)
    if len(selected) < config.minimum:
        text = render_empty_digest(generated_at=now, failed_sources=failed_sources)
    elif summarizer is None:
        summarizer = (with_fallback(OpenAICompatibleSummarizer(config.llm_endpoint, config.llm_model, config.llm_api_key_env))
                      if config.llm_endpoint and config.llm_model else local_summarize)
        text = render_digest([summarizer(item) for item in selected], generated_at=now, failed_sources=failed_sources)
    else:
        text = render_digest([summarizer(item) for item in selected], generated_at=now, failed_sources=failed_sources)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    logging.info("digest generated", extra={"fetched": len(fetched), "inserted": inserted, "selected": len(selected)})
    if not dry_run:
        webhook = os.environ.get(config.webhook_env)
        if not webhook:
            raise RuntimeError(f"missing required environment variable: {config.webhook_env}")
        send_feishu(webhook, text)
    return text
