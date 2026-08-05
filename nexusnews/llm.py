from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from .digest import DigestEntry, local_summarize
from .models import Item


class LLMSummaryError(RuntimeError):
    pass


class OpenAICompatibleSummarizer:
    """Small OpenAI-compatible client with an explicit deterministic fallback."""

    def __init__(self, endpoint: str, model: str, api_key_env: str, *, transport=urlopen, timeout: float = 20):
        self.endpoint, self.model, self.api_key_env = endpoint, model, api_key_env
        self.transport, self.timeout = transport, timeout

    def __call__(self, item: Item) -> DigestEntry:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMSummaryError(f"missing required environment variable: {self.api_key_env}")
        prompt = ("请仅返回 JSON，字段为 title、summary、why_important。title 是不超过28字的中文标题；"
                  "summary 为60至110个中文字符，陈述已证实事实；why_important 为35至75个中文字符，说明具体影响。\n"
                  f"来源：{item.source}\n标题：{item.title}\n正文：{item.content or item.title}\n链接：{item.url or ''}")
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "response_format": {"type": "json_object"}}, ensure_ascii=False).encode()
        request = Request(self.endpoint, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with self.transport(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
            result = json.loads(data["choices"][0]["message"]["content"])
            summary, why = result["summary"].strip(), result["why_important"].strip()
            if not 60 <= len(summary) <= 110 or not 35 <= len(why) <= 75:
                raise ValueError("generated text violates editorial length limits")
            return DigestEntry(result["title"].strip()[:28], item.source, item.url or "（无链接）", summary, why,
                               item.id, item.source, item.dedupe_key)
        except LLMSummaryError:
            raise
        except Exception as exc:
            raise LLMSummaryError(f"LLM summary failed: {exc}") from exc


def with_fallback(primary):
    def summarize(item: Item) -> DigestEntry:
        try:
            return primary(item)
        except Exception:
            return local_summarize(item)
    return summarize
