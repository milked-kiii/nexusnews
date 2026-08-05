from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from .models import RawItem


class FetchError(RuntimeError):
    """A source could not be fetched or decoded."""


class Transport(Protocol):
    def get(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes: ...


class UrlLibTransport:
    def get(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes:
        try:
            with urlopen(Request(url, headers=dict(headers)), timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            raise FetchError(f"HTTP {exc.code} fetching {url}") from exc
        except (URLError, TimeoutError) as exc:
            raise FetchError(f"network error fetching {url}: {exc}") from exc


@dataclass(frozen=True)
class LocalOrUrlTransport:
    """Read explicit local paths for demos and delegate URLs to HTTP transport."""

    remote: Transport

    def get(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes:
        if "://" in url:
            return self.remote.get(url, timeout=timeout, headers=headers)
        try:
            return Path(url).read_bytes()
        except OSError as exc:
            raise FetchError(f"cannot read local source {url}: {exc}") from exc


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, *names: str) -> str | None:
    for child in element:
        if _local_name(child.tag) in names and child.text:
            return child.text
    return None


@dataclass
class RSSFetcher:
    transport: Transport
    timeout: float = 10.0

    def fetch(self, url: str, *, source: str) -> list[RawItem]:
        try:
            payload = self.transport.get(url, timeout=self.timeout, headers={"Accept": "application/rss+xml, application/atom+xml, application/xml"})
            root = ET.fromstring(payload)
        except FetchError:
            raise
        except ET.ParseError as exc:
            raise FetchError(f"invalid RSS/XML from {url}: {exc}") from exc

        entries = [node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}]
        result: list[RawItem] = []
        for entry in entries:
            link = _child_text(entry, "link")
            if not link:
                link_node = next((c for c in entry if _local_name(c.tag) == "link" and c.get("href")), None)
                link = link_node.get("href") if link_node is not None else None
            result.append(RawItem(
                source=source,
                title=_child_text(entry, "title") or "",
                url=link,
                content=_child_text(entry, "description", "summary", "content"),
                published_at=_child_text(entry, "pubDate", "published", "updated"),
                external_id=_child_text(entry, "guid", "id"),
            ))
        return result


ApiParser = Callable[[object], Sequence[RawItem]]


@dataclass
class APIFetcher:
    transport: Transport
    parser: ApiParser
    timeout: float = 10.0
    headers: Mapping[str, str] | None = None

    def fetch(self, url: str) -> list[RawItem]:
        headers = {"Accept": "application/json", **(self.headers or {})}
        try:
            payload = self.transport.get(url, timeout=self.timeout, headers=headers)
            decoded = json.loads(payload)
        except FetchError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FetchError(f"invalid JSON from {url}: {exc}") from exc
        try:
            return list(self.parser(decoded))
        except Exception as exc:
            raise FetchError(f"API response from {url} did not match expected schema: {exc}") from exc
