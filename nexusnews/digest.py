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
    "policy":       "⚖️ 政策 & 治理",
    "research":     "🎓 研究前沿",
}

CATEGORY_ORDER = ["frontier", "agent", "vertical", "tools", "business", "policy", "research"]

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
    "business":  ["product", "launch", "funding", "融资", "发布", "market", "business",
                  "revenue", "partnership", "subscription", "IPO", "收购", "competitor",
                  "价格", "commercial", "startup"],
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


def select_items(items: list[Item], *, minimum: int = 5, maximum: int = 10, title_threshold: float = 0.6) -> list[Item]:
    """Pre-summary candidate selection: dedupe by title, cap at a generous ceiling.

    Relevance-score filtering happens AFTER summarization (see filter_entries),
    because scores are produced by the LLM on DigestEntry, not on raw Item.
    """
    if not 1 <= minimum <= maximum:
        raise ValueError("selection requires 1 <= minimum <= maximum")
    ranked = sorted(items, key=lambda item: (item.published_at or "", bool(item.content), item.source, item.id), reverse=True)
    selected: list[Item] = []
    for item in ranked:
        if not any(_similar(item, prior, title_threshold) for prior in selected):
            selected.append(item)
        if len(selected) == maximum:
            break
    return selected


def filter_entries(entries: list[DigestEntry], *, maximum: int = 10, min_relevance: int = 6) -> list[DigestEntry]:
    """Post-summary business-relevance filter + rank.

    Keeps only entries scoring >= min_relevance, sorts by score desc then keeps
    the top `maximum`. This is where coding/work-Agent focus is enforced.
    """
    relevant = [e for e in entries if e.relevance_score >= min_relevance]
    relevant.sort(key=lambda e: e.relevance_score, reverse=True)
    return relevant[:maximum]


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


def local_summarize(item: Item) -> DigestEntry:
    """Deterministic offline fallback; replace through Pipeline's summarizer interface."""
    text = (item.content or item.title).strip().rstrip("。！？.!?")
    summary = f"据{item.source}原文，该动态介绍了{text[:62]}。目前可确认的信息以原文披露为准，具体能力边界与适用条件仍需结合完整材料评估。"
    summary = summary[:110]
    if len(summary) < 60:
        summary += "建议查阅一手链接核对发布时间、适用范围和相关限制。"[:60 - len(summary)]
    host = urlsplit(item.url or "").netloc
    why = "这项进展可能影响 AI 技术或产品的能力、成本与可用性，值得结合一手材料持续观察后续采用情况。"
    category = classify_entry(item.title, item.content, item.source)
    return DigestEntry(item.title, item.source, item.url or f"https://{host}" if host else "（无链接）",
                       summary, why, category, item.id, item.source, item.dedupe_key, 6)


# ── text rendering (backward compat) ────────────────────────────

def render_digest(entries: list[DigestEntry], *, generated_at: datetime | None = None, digest_id: str | None = None,
                  failed_sources: int = 0) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    digest_id = digest_id or generated_at.strftime("%Y-%m-%d")
    lines = [f"🤖 AI 日报｜{generated_at:%Y-%m-%d}（共 {len(entries)} 条）", "过去 48 小时的低噪音精选。", ""]
    for number, entry in enumerate(entries, 1):
        lines.extend([f"{number}. {entry.title}", f"来源：{entry.source}", f"原文：[阅读原文]({entry.url})",
                      f"摘要：{entry.summary}", f"为什么重要：{entry.why_important}", ""])
    if failed_sources:
        lines.append(f"注：今日有 {failed_sources} 个信源暂时不可用，已基于其余来源完成筛选；不会用低质量内容补位。")
    return "\n".join(lines).rstrip()


def render_empty_digest(*, generated_at: datetime, failed_sources: int = 0) -> str:
    text = (f"🤖 AI 日报｜{generated_at:%Y-%m-%d}\n"
            "今天没有凑够 5 条达到质量阈值的 AI 动态，因此不发送常规精选。明天会继续为你筛选。")
    if failed_sources:
        text += f"\n注：今日有 {failed_sources} 个信源暂时不可用，已基于其余来源完成筛选；不会用低质量内容补位。"
    return text


# ── Feishu card rendering ──────────────────────────────────────

def _escape_md(text: str) -> str:
    """Escape special chars for lark_md."""
    return text.replace("**", "\\*\\*").replace("*", "\\*").replace("~", "\\~")


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
            
            # High-value indicator for coding/work Agent focus
            priority_badge = "🔥 " if entry.relevance_score >= 9 else ""
            
            md = (
                f"**{local_idx}. {priority_badge}{safe_title}**\n"
                f"来源：{safe_source}\n\n"
                f"{safe_summary}\n\n"
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
