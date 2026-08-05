from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SPACE = re.compile(r"\s+")
_TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid"}


@dataclass(frozen=True)
class RawItem:
    source: str
    title: str
    url: str | None = None
    content: str | None = None
    published_at: str | datetime | None = None
    external_id: str | None = None


@dataclass(frozen=True)
class Item:
    id: str
    source: str
    title: str
    url: str | None
    content: str | None
    published_at: str | None
    dedupe_key: str


def _text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _SPACE.sub(" ", value).strip()
    return cleaned or None


def _url(value: str | None) -> str | None:
    value = _text(value)
    if not value:
        return None
    parts = urlsplit(value)
    query = urlencode(
        sorted(
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING
        )
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _date(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid published_at: {value!r}") from exc
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_item(raw: RawItem) -> Item:
    source = _text(raw.source)
    title = _text(raw.title)
    if not source:
        raise ValueError("source is required")
    if not title:
        raise ValueError("title is required")
    url = _url(raw.url)
    content = _text(raw.content)
    published_at = _date(raw.published_at)
    external_id = _text(raw.external_id)

    if external_id:
        identity = ["external_id", source, external_id]
    elif url:
        identity = ["url", url]
    else:
        identity = ["content", title.casefold(), (content or "").casefold(), published_at]
    canonical = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    dedupe_key = hashlib.sha256(canonical.encode()).hexdigest()
    return Item(dedupe_key, source, title, url, content, published_at, dedupe_key)
