from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from urllib.parse import urlsplit

from .models import Item


_WORDS = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STOP = {"the", "a", "an", "and", "of", "to", "in", "for", "on", "with", "ai", "发布", "推出", "宣布"}


def _tokens(title: str) -> set[str]:
    return {word.casefold() for word in _WORDS.findall(title) if word.casefold() not in _STOP and len(word) > 1}


def _similar(left: Item, right: Item, threshold: float) -> bool:
    if left.url and right.url and left.url == right.url:
        return True
    a, b = _tokens(left.title), _tokens(right.title)
    return bool(a and b) and len(a & b) / len(a | b) >= threshold


def select_items(items: list[Item], *, minimum: int = 5, maximum: int = 10, title_threshold: float = 0.6) -> list[Item]:
    if not 1 <= minimum <= maximum:
        raise ValueError("selection requires 1 <= minimum <= maximum")
    ranked = sorted(items, key=lambda item: (item.published_at or "", bool(item.content), item.source, item.id), reverse=True)
    selected: list[Item] = []
    for item in ranked:
        if not any(_similar(item, prior, title_threshold) for prior in selected):
            selected.append(item)
        if len(selected) == maximum:
            break
    # A short digest is preferable to duplicating an event merely to reach minimum.
    return selected


@dataclass(frozen=True)
class DigestEntry:
    title: str
    source: str
    url: str
    summary: str
    why_important: str


def local_summarize(item: Item) -> DigestEntry:
    """Deterministic offline fallback; replace through Pipeline's summarizer interface."""
    text = (item.content or item.title).strip()
    excerpt = text[:133] + "…" if len(text) > 136 else text
    summary = f"该报道介绍：{excerpt}"
    host = urlsplit(item.url or "").netloc
    why = f"这项进展可能影响 AI 技术或产品的后续采用，建议结合 {item.source} 原文评估。"
    return DigestEntry(item.title, item.source, item.url or f"https://{host}" if host else "（无链接）", summary, why)


def render_digest(entries: list[DigestEntry], *, generated_at: datetime | None = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    lines = [f"AI 新闻日报 · {generated_at:%Y-%m-%d}", ""]
    for number, entry in enumerate(entries, 1):
        lines.extend([f"{number}. {entry.title}", f"来源：{entry.source}", f"链接：{entry.url}", f"摘要：{entry.summary}", f"为什么重要：{entry.why_important}", ""])
    return "\n".join(lines).rstrip()


def cutoff(hours: int = 24, *, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now.astimezone(timezone.utc) - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
