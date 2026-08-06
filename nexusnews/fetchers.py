from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlencode
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
        headers = {
            "Accept": "application/rss+xml, application/atom+xml, application/xml",
            "User-Agent": "Mozilla/5.0 (compatible; Nexusnews/1.0)",
        }
        try:
            payload = self.transport.get(url, timeout=self.timeout, headers=headers)
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


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def parse_reddit(data: object, *, source: str) -> Sequence[RawItem]:
    children = data["data"]["children"]  # type: ignore[index]
    result = []
    for child in children:
        post = child["data"]
        permalink = _string(post.get("permalink"))
        result.append(RawItem(
            source=source,
            title=_string(post.get("title")) or "Reddit post",
            url=f"https://www.reddit.com{permalink}" if permalink else _string(post.get("url")),
            content=_string(post.get("selftext")),
            published_at=(datetime.fromtimestamp(post["created_utc"], tz=timezone.utc).isoformat()
                          if isinstance(post.get("created_utc"), (int, float)) else post.get("created_utc")),
            external_id=_string(post.get("name")) or _string(post.get("id")),
        ))
    return result


def parse_x(data: object, *, source: str) -> Sequence[RawItem]:
    users = {user.get("id"): user for user in data.get("includes", {}).get("users", [])}  # type: ignore[union-attr]
    result = []
    for post in data.get("data", []):  # type: ignore[union-attr]
        text = _string(post.get("text")) or "X post"
        author = users.get(post.get("author_id"), {})
        username = _string(author.get("username"))
        result.append(RawItem(
            source=source,
            title=(f"@{username}: {text}" if username else text)[:280],
            url=f"https://x.com/{username}/status/{post['id']}" if username else f"https://x.com/i/web/status/{post['id']}",
            content=text,
            published_at=post.get("created_at"),
            external_id=str(post["id"]),
        ))
    return result


def parse_discord(data: object, *, source: str, channel_id: str) -> Sequence[RawItem]:
    result = []
    for message in data:  # type: ignore[union-attr]
        content = _string(message.get("content"))
        if not content:
            continue
        author = message.get("author") or {}
        author_name = _string(author.get("global_name")) or _string(author.get("username"))
        title = (f"{author_name}: {content}" if author_name else content).splitlines()[0][:280]
        result.append(RawItem(
            source=source,
            title=title,
            url=f"https://discord.com/channels/{message.get('guild_id', '@me')}/{channel_id}/{message['id']}",
            content=content,
            published_at=message.get("timestamp"),
            external_id=str(message["id"]),
        ))
    return result


@dataclass
class PlatformFetcher:
    """Fetch official platform APIs, keeping credentials in environment variables."""

    transport: Transport
    timeout: float = 10.0

    def fetch(self, source: object) -> list[RawItem]:
        kind = source.kind
        if kind in {"rss", "medium"}:
            return RSSFetcher(self.transport, timeout=self.timeout).fetch(source.url, source=source.name)
        if kind == "reddit":
            return APIFetcher(
                self.transport,
                lambda data: parse_reddit(data, source=source.name),
                timeout=self.timeout,
                headers={"User-Agent": "Nexusnews/1.0 (read-only news digest)"},
            ).fetch(source.url)

        token = os.environ.get(source.token_env or "")
        if not token:
            raise FetchError(f"missing required environment variable: {source.token_env}")
        if kind == "x":
            params = urlencode({
                "query": source.query,
                "max_results": max(10, source.limit),
                "tweet.fields": "created_at,author_id",
                "expansions": "author_id",
                "user.fields": "username,name",
            })
            return APIFetcher(
                self.transport, lambda data: parse_x(data, source=source.name), timeout=self.timeout,
                headers={"Authorization": f"Bearer {token}"},
            ).fetch(f"https://api.x.com/2/tweets/search/recent?{params}")
        if kind == "discord":
            params = urlencode({"limit": source.limit})
            url = f"https://discord.com/api/v10/channels/{source.channel_id}/messages?{params}"
            return APIFetcher(
                self.transport,
                lambda data: parse_discord(data, source=source.name, channel_id=source.channel_id),
                timeout=self.timeout,
                headers={"Authorization": f"Bot {token}"},
            ).fetch(url)
        raise FetchError(f"unsupported source kind: {kind}")
