from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import re
from urllib.parse import urlsplit

from .models import Item


_WORDS = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STOP = {"the", "a", "an", "and", "of", "to", "in", "for", "on", "with", "ai", "发布", "推出", "宣布"}

# ── categories ──────────────────────────────────────────────────

CATEGORIES = {
    "frontier":     "🧠 前沿模型",
    "agent":        "🤖 Agent & 智能体",
    "vertical":     "🏭 垂类落地",
    "tools":        "🔧 开源 & 工具",
    "business":     "💰 产品 & 商业",
    "funding":      "💸 投融资风向",
    "policy":       "⚖️ 政策 & 治理",
    "research":     "🎓 研究前沿",
}

CATEGORY_ORDER = ["frontier", "agent", "vertical", "tools", "funding", "business", "policy", "research"]

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "frontier":  ["model", "GPT", "Claude", "Gemini", "LLM", "参数", "benchmark", "training",
                  "reasoning", "权重", "architecture", "token", "上下文", "多模态", "推理",
                  "diffusion", "transformer", "大模型", "模型发布", "pretrain", "foundation"],
    "agent":     ["agent", "智能体", "agentic", "tool calling", "function call", "autonomous",
                  "RPA", "workflow", "task", "multi-agent", "swarm", "自主", "编排"],
    "vertical":  ["medical", "healthcare", "医疗", "legal", "法律", "finance", "金融",
                  "education", "教育", "manufacturing", "制造", "enterprise", "企业",
                  "deployment", "落地", "行业", "政务", "零售", "驾驶"],
    "tools":     ["open source", "开源", "GitHub", "framework", "library", "API", "SDK",
                  "developer", "tool", "权重", "HuggingFace", "plugin", "extension",
                  "rust", "python", "docker", "pip", "npm"],
    "business":  ["product", "launch", "发布", "market", "business",
                  "revenue", "partnership", "subscription", "IPO", "收购", "competitor",
                  "价格", "commercial", "startup"],
    "funding":   ["融资", "领投", "参投", "注资", "跟投", "轮次", "天使轮", "A轮", "B轮",
                  "C轮", "D轮", "Pre-A", "Pre-B", "战略投资", "估值", "funding", "raised",
                  "investment", "venture", "Sequoia", "红杉", "高瓴", "真格", "IDG资本",
                  "经纬", "源码", "纪源", "启明", "a16z", "YC", "Benchmark", "Accel"],
    "policy":    ["regulation", "policy", "监管", "法律", "safety", "security", "ethics",
                  "伦理", "lawsuit", "ban", "government", "欧盟", "白宫", "国会",
                  "审查", "隐私", "GDPR", "export control", "制裁"],
    "research":  ["research", "paper", "论文", "arXiv", "study", "experiment", "dataset",
                  "benchmark", "method", "approach", "理论", "证明", "数学", "Nobel",
                  "peer review", "conference", "NeurIPS", "ICML", "published"],
}


def classify_entry(title: str, content: str | None, source: str) -> str:
    """Keyword-based category classifier; used as local fallback."""
    text = f"{title} {content or ''} {source}".lower()
    scores: dict[str, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw.lower() in text)
    best = max(scores, key=scores.get)  # type: ignore[type-var]
    return best if scores[best] > 0 else "frontier"


# ── item selection ──────────────────────────────────────────────

def _tokens(title: str) -> set[str]:
    return {word.casefold() for word in _WORDS.findall(title) if word.casefold() not in _STOP and len(word) > 1}


def _similar(left: Item, right: Item, threshold: float) -> bool:
    if left.url and right.url and left.url == right.url:
        return True
    a, b = _tokens(left.title), _tokens(right.title)
    return bool(a and b) and len(a & b) / len(a | b) >= threshold


def select_items(items: list[Item], *, minimum: int = 5, maximum: int = 10,
                 title_threshold: float = 0.6, per_source_cap: int | None = None) -> list[Item]:
    """Pre-summary candidate selection: dedupe by title, cap at a generous ceiling.

    Relevance-score filtering happens AFTER summarization (see filter_entries),
    because scores are produced by the LLM on DigestEntry, not on raw Item.

    ``per_source_cap`` limits how many items a single source can contribute.
    When None (default), it is computed as ``max(2, maximum // 3)`` so a chatty
    feed (e.g. 36氪快讯) cannot crowd out slower but higher-signal sources
    (Qoder, Cursor, WeChat blogs).
    """
    if not 1 <= minimum <= maximum:
        raise ValueError("selection requires 1 <= minimum <= maximum")
    if per_source_cap is None:
        per_source_cap = max(2, maximum // 3)
    ranked = sorted(items, key=lambda item: (item.published_at or "", bool(item.content), item.source, item.id), reverse=True)
    selected: list[Item] = []
    source_counts: dict[str, int] = {}
    for item in ranked:
        if source_counts.get(item.source, 0) >= per_source_cap:
            continue
        if any(_similar(item, prior, title_threshold) for prior in selected):
            continue
        selected.append(item)
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        if len(selected) == maximum:
            break
    return selected


def filter_entries(entries: list[DigestEntry], *, maximum: int = 10, min_relevance: int = 6,
                   ensure_top_fire: int = 1) -> list[DigestEntry]:
    """Post-summary business-relevance filter + rank.

    Keeps only entries scoring >= min_relevance, sorts by score desc then keeps
    the top `maximum`. This is where coding/work-Agent focus is enforced.

    ``ensure_top_fire``: if non-zero, promote the top-N entries' scores to 9 so
    the digest always has at least one 🔥-tier headline. LLM tends to cluster
    scores at 7-8 which would otherwise leave the entire digest in the blue
    tier. Promotion only happens when NO entry already reached 9 — we never
    downgrade an organically high-scoring entry.
    """
    relevant = [e for e in entries if e.relevance_score >= min_relevance]
    relevant.sort(key=lambda e: e.relevance_score, reverse=True)
    selected = relevant[:maximum]
    if ensure_top_fire > 0 and selected and all(e.relevance_score < 9 for e in selected):
        # Promote top N to 9 so the digest has at least one fire-tier headline
        promoted: list[DigestEntry] = []
        for i, entry in enumerate(selected):
            if i < ensure_top_fire:
                # Use dataclasses.replace to preserve immutability
                from dataclasses import replace
                promoted.append(replace(entry, relevance_score=9))
            else:
                promoted.append(entry)
        return promoted
    return selected


# ── digest entry ────────────────────────────────────────────────

@dataclass(frozen=True)
class DigestEntry:
    title: str
    source: str
    url: str
    summary: str
    why_important: str
    category: str = "frontier"
    item_id: str = ""
    source_id: str = ""
    event_key: str = ""
    relevance_score: int = 0
    published_at: str | None = None


# Short names like "IDG", "YC", "GGV" easily false-positive when they appear as
# substrings inside unrelated words (e.g. "IDG业务" = Lenovo's IDG business
# unit). Require word-boundary context for short ASCII VC names and require
# investment-context keywords within the same sentence for any match.
_VC_ACTION = re.compile(
    r"(融资|领投|参投|投资|注资|跟投|轮次|天使轮|A轮|B轮|C轮|D轮|Pre-|战略投资|"
    r"funding|raised|invests?|investment|leads?|backed|round|valuation|venture)",
    re.IGNORECASE,
)


def _vc_hit(vc_watchlist: tuple[str, ...], *texts: str) -> str | None:
    """Return the matched VC name if any text mentions a watched VC *in an
    investment context*, else None.

    A mention alone is not enough — an investment action word (融资/领投/
    funding/raised/...) must appear within a short window around the VC name
    so that unrelated mentions like "IDG业务" (Lenovo's IDG business unit)
    don't trigger a hit. Short ASCII VC names (≤4 letters) additionally
    require word boundaries.
    """
    if not vc_watchlist:
        return None
    joined = " ".join(t for t in texts if t)
    if not _VC_ACTION.search(joined):
        return None
    for vc in vc_watchlist:
        if not vc:
            continue
        # Build a pattern that also enforces word boundaries on short ASCII
        # names (so "IDG业务" doesn't match "IDG").
        if vc.isascii() and len(vc) <= 4:
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(vc)}(?![A-Za-z])")
        else:
            pattern = re.compile(re.escape(vc))
        for m in pattern.finditer(joined):
            # Investment action must appear within ±30 chars of the VC mention
            window_start = max(0, m.start() - 30)
            window_end = min(len(joined), m.end() + 30)
            window = joined[window_start:window_end]
            if _VC_ACTION.search(window):
                return vc
    return None


def local_summarize(item: Item, *, vc_watchlist: tuple[str, ...] = ()) -> DigestEntry:
    """Deterministic offline fallback; replace through Pipeline's summarizer interface."""
    text = (item.content or item.title).strip().rstrip("。！？.!?")
    summary = f"据{item.source}原文，该动态介绍了{text[:62]}。目前可确认的信息以原文披露为准，具体能力边界与适用条件仍需结合完整材料评估。"
    summary = summary[:110]
    if len(summary) < 60:
        summary += "建议查阅一手链接核对发布时间、适用范围和相关限制。"[:60 - len(summary)]
    host = urlsplit(item.url or "").netloc
    vc_hit = _vc_hit(vc_watchlist, item.title, item.content or "")
    if vc_hit:
        why = f"该融资/投资动态涉及关注名单内的机构（{vc_hit}），是判断 AI 赛道资金流向与商业信心的关键信号。"
    else:
        why = "这项进展可能影响 AI 技术或产品的能力、成本与可用性，值得结合一手材料持续观察后续采用情况。"
    category = classify_entry(item.title, item.content, item.source)
    if vc_hit and category != "policy":
        category = "funding"
    # Watchlist hits get a small relevance bump so they survive filtering even
    # when the LLM/local fallback would otherwise rank them low.
    base_score = 6
    if vc_hit:
        base_score = 7
    return DigestEntry(item.title, item.source, item.url or f"https://{host}" if host else "（无链接）",
                       summary, why, category, item.id, item.source, item.dedupe_key, base_score,
                       published_at=item.published_at)


# ── text rendering (backward compat) ────────────────────────────

def render_digest(entries: list[DigestEntry], *, generated_at: datetime | None = None, digest_id: str | None = None,
                  failed_sources: int = 0) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    digest_id = digest_id or generated_at.strftime("%Y-%m-%d")
    lines = [f"🤖 AI 日报｜{generated_at:%Y-%m-%d}（共 {len(entries)} 条）", "过去 48 小时的低噪音精选。", ""]
    for number, entry in enumerate(entries, 1):
        summary_line = entry.summary if entry.summary.startswith(">") else f"> 摘要：{entry.summary}"
        lines.extend([f"{number}. {entry.title}", f"来源：{entry.source}", f"原文：[阅读原文]({entry.url})",
                      summary_line, f"为什么重要：{entry.why_important}", ""])
    if failed_sources:
        lines.append(f"注：今日有 {failed_sources} 个信源暂时不可用，已基于其余来源完成筛选；不会用低质量内容补位。")
    return "\n".join(lines).rstrip()


def render_empty_digest(*, generated_at: datetime, failed_sources: int = 0, minimum: int = 3) -> str:
    text = (f"🤖 AI 日报｜{generated_at:%Y-%m-%d}\n"
            f"今天没有凑够 {minimum} 条达到质量阈值的 AI 动态，因此不发送常规精选。明天会继续为你筛选。")
    if failed_sources:
        text += f"\n注：今日有 {failed_sources} 个信源暂时不可用，已基于其余来源完成筛选；不会用低质量内容补位。"
    return text


# ── Feishu card rendering ──────────────────────────────────────

def _escape_md(text: str) -> str:
    """Escape special chars for lark_md."""
    return text.replace("**", "\\*\\*").replace("*", "\\*").replace("~", "\\~")


def _format_published(published_at: str | None, *, now: datetime | None = None) -> str:
    """Human-friendly publish time for a digest entry.

    Examples: "8月8日 12:30" (this year), "2025年5月19日" (older),
    "时间未知" (when the source didn't expose a date).
    """
    if not published_at:
        return "时间未知"
    try:
        parsed = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return "时间未知"
    now = now or datetime.now(timezone.utc)
    local = parsed.astimezone(now.tzinfo or timezone.utc)
    if local.year == now.year:
        return f"{local.month}月{local.day}日 {local.hour:02d}:{local.minute:02d}"
    return f"{local.year}年{local.month}月{local.day}日"


def render_card(entries: list[DigestEntry], *, generated_at: datetime | None = None,
                failed_sources: int = 0, doc_url: str | None = None) -> str:
    """Render a Feishu interactive card JSON string with category grouping.

    If ``doc_url`` is given, a "查看文档" primary button is appended so the
    reader can open the synced Feishu cloud document.
    """
    generated_at = generated_at or datetime.now(timezone.utc)
    zh_date = f"{generated_at.month}月{generated_at.day}日"

    # Group by category, preserving category order
    grouped: dict[str, list[DigestEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.category, []).append(entry)

    elements: list[dict] = []
    displayed_categories = [c for c in CATEGORY_ORDER if c in grouped]

    for cat in displayed_categories:
        items = grouped[cat]
        # Category header
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**{CATEGORIES[cat]}**  ({len(items)}条)"},
        })

        for local_idx, entry in enumerate(items, 1):
            safe_title = _escape_md(entry.title)[:80]
            safe_summary = _escape_md(entry.summary)[:200]
            safe_why = _escape_md(entry.why_important)[:150]
            safe_source = _escape_md(entry.source)
            safe_published = _escape_md(_format_published(entry.published_at))

            # Three-tier visual priority: 🔥 yellow for 9-10, ⚡ blue for 7-8,
            # default bold for the rest. Both emoji and color carry the signal.
            if entry.relevance_score >= 9:
                badge = "🔥"
                title_html = f"<font color='yellow'>{badge} {local_idx}. {safe_title}</font>"
            elif entry.relevance_score >= 7:
                badge = "⚡"
                title_html = f"<font color='blue'>{badge} {local_idx}. {safe_title}</font>"
            else:
                badge = "📌"
                title_html = f"**{badge} {local_idx}. {safe_title}**"
            # Render summary as a quote line unless it already starts with one
            summary_line = safe_summary if safe_summary.startswith(">") else f"> {safe_summary}"

            md = (
                f"{title_html}\n"
                f"<font color='grey'>{safe_source} · {safe_published}</font>\n\n"
                f"{summary_line}\n\n"
                f"💡 {safe_why}"
            )

            element: dict = {
                "tag": "div",
                "text": {"tag": "lark_md", "content": md},
            }
            if entry.url and entry.url not in ("（无链接）", ""):
                element["extra"] = {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🔗 阅读原文"},
                    "url": entry.url,
                    "type": "default",
                }
            elements.append(element)
            elements.append({"tag": "hr"})

        # Remove trailing hr for last category (will be re-added if more categories follow)
        if elements and elements[-1].get("tag") == "hr":
            elements.pop()

    # Footer
    footer_lines = ["过去 48 小时的低噪音精选"]
    if failed_sources:
        footer_lines.append(f"今日有 {failed_sources} 个信源暂时不可用")
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": " · ".join(footer_lines)}],
    })

    # Optional doc link button
    if doc_url:
        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "📄 查看文档"},
                    "url": doc_url,
                    "type": "primary",
                }
            ],
        })

    card = {
        "header": {
            "title": {"tag": "plain_text", "content": f"🤖 AI 日报 · {zh_date} · {len(entries)}条"},
            "template": "blue",
        },
        "elements": elements,
    }

    return json.dumps({"msg_type": "interactive", "card": card}, ensure_ascii=False)


# ── cutoff ──────────────────────────────────────────────────────

def cutoff(hours: int = 24, *, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now.astimezone(timezone.utc) - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
