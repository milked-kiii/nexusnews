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
    item_id: str = ""
    source_id: str = ""
    event_key: str = ""


def local_summarize(item: Item) -> DigestEntry:
    """Deterministic offline fallback; replace through Pipeline's summarizer interface."""
    text = (item.content or item.title).strip()
    summary = f"据{item.source}原文，该动态介绍了{text[:62]}。目前可确认的信息以原文披露为准，具体能力边界与适用条件仍需结合完整材料评估。"
    summary = summary[:110]
    if len(summary) < 60:
        summary += "建议查阅一手链接核对发布时间、适用范围和相关限制。"[:60 - len(summary)]
    host = urlsplit(item.url or "").netloc
    why = "这项进展可能影响 AI 技术或产品的能力、成本与可用性，值得结合一手材料持续观察后续采用情况。"
    return DigestEntry(item.title, item.source, item.url or f"https://{host}" if host else "（无链接）", summary, why,
                       item.id, item.source, item.dedupe_key)


def render_digest(entries: list[DigestEntry], *, generated_at: datetime | None = None, digest_id: str | None = None,
                  failed_sources: int = 0) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    digest_id = digest_id or generated_at.strftime("%Y-%m-%d")
    lines = [f"🤖 AI 日报｜{generated_at:%Y-%m-%d}（共 {len(entries)} 条）", "过去 24 小时的低噪音精选；回复反馈可校准明日选题。", ""]
    for number, entry in enumerate(entries, 1):
        lines.extend([f"{number}. {entry.title}", f"来源：{entry.source}", f"原文：[阅读原文]({entry.url})", f"摘要：{entry.summary}", f"为什么重要：{entry.why_important}", f"反馈：回复 {number}好 / {number}差", ""])
    lines.append("本期反馈：回复 本期好 / 本期差")
    if failed_sources:
        lines.append(f"注：今日有 {failed_sources} 个信源暂时不可用，已基于其余来源完成筛选；不会用低质量内容补位。")
    lines.append(f"反馈标识：{digest_id}")
    return "\n".join(lines).rstrip()


def render_empty_digest(*, generated_at: datetime, failed_sources: int = 0) -> str:
    text = (f"🤖 AI 日报｜{generated_at:%Y-%m-%d}\n"
            "今天没有凑够 5 条达到质量阈值的 AI 动态，因此不发送常规精选。明天会继续为你筛选。")
    if failed_sources:
        text += f"\n注：今日有 {failed_sources} 个信源暂时不可用，已基于其余来源完成筛选；不会用低质量内容补位。"
    return text


def cutoff(hours: int = 24, *, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now.astimezone(timezone.utc) - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
