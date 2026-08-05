from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    kind: str = "rss"


@dataclass(frozen=True)
class Config:
    sources: tuple[Source, ...]
    database: str = "var/nexusnews.db"
    output: str = "var/latest-digest.txt"
    minimum: int = 5
    maximum: int = 10
    webhook_env: str = "FEISHU_WEBHOOK_URL"
    llm_endpoint: str | None = None
    llm_model: str | None = None
    llm_api_key_env: str = "NEXUSNEWS_LLM_API_KEY"


def load_config(path: str | Path) -> Config:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources = tuple(Source(**row) for row in data["sources"])
        config = Config(sources=sources, **{k: v for k, v in data.items() if k != "sources"})
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid config {path}: {exc}") from exc
    if not sources:
        raise ValueError("config must contain at least one source")
    if any(source.kind != "rss" for source in sources):
        raise ValueError("configured source kind must currently be 'rss'")
    if not 1 <= config.minimum <= config.maximum <= 10:
        raise ValueError("config selection must satisfy 1 <= minimum <= maximum <= 10")
    if bool(config.llm_endpoint) != bool(config.llm_model):
        raise ValueError("llm_endpoint and llm_model must be configured together")
    return config
