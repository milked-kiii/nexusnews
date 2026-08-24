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

    def __init__(self, endpoint: str, model: str, api_key_env: str, *,
                 transport=urlopen, timeout: float = 20,
                 vc_watchlist: tuple[str, ...] = ()):
        self.endpoint, self.model, self.api_key_env = endpoint, model, api_key_env
        self.transport, self.timeout = transport, timeout
        self.vc_watchlist = vc_watchlist

    def __call__(self, item: Item) -> DigestEntry:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMSummaryError(f"missing required environment variable: {self.api_key_env}")

        # VC watchlist hint
        vc_hint = ""
        if self.vc_watchlist:
            watchlist_str = "、".join(self.vc_watchlist[:15])
            vc_hint = (
                f"\n- 若新闻涉及这些 VC/投资机构领投或参投（{watchlist_str} 等），"
                "且对象是 AI/Agent/编程工具赛道，relevance_score 至少给到 7"
            )

        # Business context for coding/work Agent product team
        system_context = (
            "你是 AI 新闻筛选助手，服务于 coding 和 work Agent 产品团队。\n\n"
            "**核心关注**：\n"
            "- Coding Agent（代码生成、IDE 集成、编程助手）\n"
            "- Work Agent（任务自动化、RPA、流程编排）\n"
            "- AI 赛道 VC 投资风向（融资、领投、估值）\n\n"
            "**打分标准（0-10）**：\n"
            "- 9-10：Coding/Work Agent 核心能力、工具调用、多 Agent 协作、上下文/推理突破、模型/Agent 架构重大创新\n"
            "- 7-8：开源模型/工具、API 更新、代码数据集/评测、Agent 框架、AI 赛道大额融资/VC 投资动态、有实战参考价值的工程实践\n"
            "- 5-6：行业动态、产品更新、有一定参考价值但不直接影响 Agent 产品方向\n"
            "- 3-4：纯学术论文（无实现）、图像/视频生成、政策法规、外围新闻\n"
            "- 0-2：纯娱乐应用、与 Agent 完全无关\n\n"
            "**模型新闻降权规则**：\n"
            "- 纯模型 benchmark 刷分、小版本更新（如 GPT-4o → GPT-4o-mini）、非架构性改进：最高 6 分\n"
            "- 只有以下级别的模型发布才给 7+：\n"
            "  · 新一代旗舰（GPT-5、Claude 4、Gemini 2.0、Kimi K3、DeepSeek V4 等）\n"
            "  · 架构创新（全新注意力机制、推理范式、训练方法）\n"
            "  · 对 Agent 能力有实质提升（工具调用、多步推理、代码生成显著增强）\n"
            "- 模型新闻必须说明对 Agent 产品的具体影响，否则降 1-2 分\n\n"
            "**时效性要求**：\n"
            "- 若标题或正文提到具体旧日期（如\"5月19日\"、\"今年3月\"），且事件不是近期发生的，relevance_score 降 2-3 分\n"
            "- 若产品/模型版本号是几个月前发布的（如 GPT-4o 已发布一年），按当下时点判断是否仍有时效价值\n"
            "- Reddit/GitHub 内容通常是英文，直接用英文理解即可，不需要翻译后再判断\n"
            f"{vc_hint}"
        )

        prompt = (
            f"{system_context}\n\n"
            f"来源：{item.source}\n"
            f"标题：{item.title}\n"
            f"正文：{item.content or item.title}\n"
            f"链接：{item.url or ''}\n\n"
            "请返回 JSON，字段为 relevance_score (整数0-10), title (28字符以内), "
            "summary (60到110个汉字), why_important (35到75个汉字), "
            "category (从以下8选1: frontier(前沿模型) / agent(Agent与智能体) / vertical(垂类落地) / "
            "tools(开源&工具) / business(产品&商业) / funding(投融资风向，含VC领投/融资公告) / "
            "policy(政策&治理) / research(研究前沿))。"
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

            if category not in {"frontier", "agent", "vertical", "tools", "business", "funding", "policy", "research"}:
                category = classify_entry(item.title, item.content, item.source)

            # Soft length enforcement: pad short text, truncate long text.
            # The LLM sometimes produces terse output for low-relevance items
            # (which is fine — we just need it not to crash the pipeline).
            if summary and len(summary) < 60:
                summary = (summary + " 详情参见原文。")[:110]
            summary = (summary or "（无摘要）")[:110]
            if why and len(why) < 35:
                why = (why + " 值得持续关注。")[:75]
            why = (why or "（暂无解读）")[:75]
            
            # Parse GitHub repo metadata from content prefix
            repo_created = None
            repo_stars = None
            if item.content and item.source.lower().startswith("github"):
                import re as _re
                m = _re.match(r"created:(\S+)\s+stars:(\d+)", item.content)
                if m:
                    repo_created = m.group(1)
                    repo_stars = int(m.group(2))

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
                relevance_score,
                published_at=item.published_at,
                repo_created=repo_created,
                repo_stars=repo_stars,
            )
        except LLMSummaryError:
            raise
        except Exception as exc:
            raise LLMSummaryError(f"LLM summary failed: {exc}") from exc


def with_fallback(primary, *, vc_watchlist: tuple[str, ...] = ()):
    def summarize(item: Item) -> DigestEntry:
        try:
            return primary(item)
        except Exception:
            return local_summarize(item, vc_watchlist=vc_watchlist)
    return summarize
