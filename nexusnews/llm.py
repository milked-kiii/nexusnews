from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from .digest import DigestEntry, classify_entry, local_summarize
from .models import Item


class LLMSummaryError(RuntimeError):
    pass


class OpenAICompatibleSummarizer:
    """Small OpenAI-compatible client with business-relevance scoring and deterministic fallback."""

    def __init__(self, endpoint: str, model: str, api_key_env: str, *, transport=urlopen, timeout: float = 20):
        self.endpoint, self.model, self.api_key_env = endpoint, model, api_key_env
        self.transport, self.timeout = transport, timeout

    def __call__(self, item: Item) -> DigestEntry:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMSummaryError(f"missing required environment variable: {self.api_key_env}")
        
        # Business context for coding/work Agent product team
        system_context = (
            "你是 AI 新闻筛选助手，服务于 coding 和 work Agent 产品团队。\n\n"
            "**核心关注**：\n"
            "- Coding Agent（代码生成、IDE 集成、编程助手）\n"
            "- Work Agent（任务自动化、RPA、流程编排）\n\n"
            "**打分标准（0-10）**：\n"
            "- 9-10：Coding/Work Agent 核心能力、工具调用、多 Agent 协作、上下文/推理突破\n"
            "- 6-8：开源模型/工具、API 更新、代码数据集/评测、Agent 框架\n"
            "- 3-5：纯学术论文（无实现）、图像/视频生成、政策法规\n"
            "- 0-2：纯商业/融资、娱乐应用、与 Agent 无关"
        )
        
        prompt = (
            f"{system_context}\n\n"
            f"来源：{item.source}\n"
            f"标题：{item.title}\n"
            f"正文：{item.content or item.title}\n"
            f"链接：{item.url or ''}\n\n"
            "请返回 JSON，字段为 relevance_score (整数0-10), title (28字符以内), "
            "summary (60到110个汉字), why_important (35到75个汉字), "
            "category (从以下7选1: frontier(前沿模型) / agent(Agent与智能体) / vertical(垂类落地) / "
            "tools(开源&工具) / business(产品&商业) / policy(政策&治理) / research(研究前沿))。"
        )
        
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }, ensure_ascii=False).encode()
        
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
        )
        
        try:
            with self.transport(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
            result = json.loads(data["choices"][0]["message"]["content"])
            
            # Extract and validate
            relevance_score = int(result.get("relevance_score", 0))
            summary = result["summary"].strip()
            why = result["why_important"].strip()
            category = result.get("category", "frontier").strip()
            
            if category not in {"frontier", "agent", "vertical", "tools", "business", "policy", "research"}:
                category = classify_entry(item.title, item.content, item.source)
            
            if not 60 <= len(summary) <= 110 or not 35 <= len(why) <= 75:
                raise ValueError("generated text violates editorial length limits")
            
            return DigestEntry(
                result["title"].strip()[:28],
                item.source,
                item.url or "（无链接）",
                summary,
                why,
                category,
                item.id,
                item.source,
                item.dedupe_key,
                relevance_score
            )
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
