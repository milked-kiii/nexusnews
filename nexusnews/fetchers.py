from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlencode, urljoin
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import xml.etree.ElementTree as ET

from .models import RawItem


class _RedirectWith308(HTTPRedirectHandler):
    """urllib's redirect_request only handles 301/302/303/307 — 308 (Permanent
    Redirect, RFC 7538) is rejected even though it is semantically a GET/HEAD
    redirect. Override redirect_request to allow 308 for GET/HEAD."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code == 308 and req.get_method() in ("GET", "HEAD"):
            # Re-implement the GET/HEAD branch without the strict code check.
            from urllib.request import Request
            newurl = newurl.replace(" ", "%20")
            newheaders = {k: v for k, v in req.headers.items()
                          if k.lower() not in ("content-length", "content-type")}
            return Request(newurl, headers=newheaders,
                           origin_req_host=req.origin_req_host,
                           unverifiable=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_307(self, req, fp, code, msg, headers):
        return self.http_error_302(req, fp, code, msg, headers)

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_302(req, fp, code, msg, headers)


_OPENER = build_opener(_RedirectWith308)


class FetchError(RuntimeError):
    """A source could not be fetched or decoded."""


class Transport(Protocol):
    def get(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes: ...


class UrlLibTransport:
    def get(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes:
        try:
            with _OPENER.open(Request(url, headers=dict(headers)), timeout=timeout) as response:
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


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_html(fragment: str) -> str:
    """Collapse an HTML fragment to plain text."""
    return _WS.sub(" ", unescape(_TAG.sub(" ", fragment))).strip()


# Match common English date prefixes like "Aug 6, 2026" / "Jul 30, 2026"
_DATE_PREFIX = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s+(\d{4})\b"
)
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _extract_date(text: str) -> str | None:
    """Pull a 'Mon DD, YYYY' date out of card text and return ISO format."""
    m = _DATE_PREFIX.search(text)
    if not m:
        return None
    month = _MONTHS.get(m.group(1)[:3].lower())
    if not month:
        return None
    try:
        dt = datetime(int(m.group(3)), month, int(m.group(2)), tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


@dataclass
class WebpageFetcher:
    """Scrape a blog/list page by extracting <a> cards pointing at article URLs.

    Config shape (kind="webpage"):
        url            — the listing page to GET
        link_pattern   — regex matched against the href (e.g. r"^/blog/[\\w-]+$")
        title_group    — optional regex with one capture group applied to the
                         anchor's inner text to extract the title (default:
                         whole inner text, whitespace collapsed)
        exclude_pattern — optional regex; matching hrefs are skipped (e.g.
                          r"^/blog/topic/" to skip category pages on Cursor)
    """

    transport: Transport
    timeout: float = 10.0

    _ANCHOR = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)

    def fetch(self, url: str, *, source: str, link_pattern: str,
              title_group: str | None = None, exclude_pattern: str | None = None,
              limit: int = 30) -> list[RawItem]:
        headers = {
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        }
        payload = self.transport.get(url, timeout=self.timeout, headers=headers)
        html = payload.decode("utf-8", errors="ignore")

        link_re = re.compile(link_pattern)
        exclude_re = re.compile(exclude_pattern) if exclude_pattern else None
        title_re = re.compile(title_group, re.DOTALL) if title_group else None

        seen: set[str] = set()
        result: list[RawItem] = []
        for m in self._ANCHOR.finditer(html):
            href, inner = m.group(1), m.group(2)
            if not link_re.search(href):
                continue
            if exclude_re and exclude_re.search(href):
                continue
            absolute = urljoin(url, href)
            if absolute in seen:
                continue
            seen.add(absolute)

            text = _strip_html(inner)
            if title_re:
                tm = title_re.search(text)
                if not tm:
                    continue
                title = tm.group(1).strip()
            else:
                title = text
            if not title or len(title) < 6:
                continue
            # Inner text often holds extra metadata (date, author, blurb); keep
            # a trimmed version as content so the LLM has signal for scoring.
            content = text[:600] if len(text) > len(title) + 10 else None
            result.append(RawItem(
                source=source,
                title=title[:300],
                url=absolute,
                content=content,
                published_at=_extract_date(text),
                external_id=absolute,
            ))
            if len(result) >= limit:
                break
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
        if kind == "webpage":
            return WebpageFetcher(self.transport, timeout=self.timeout).fetch(
                source.url, source=source.name,
                link_pattern=source.link_pattern,
                title_group=source.title_group,
                exclude_pattern=source.exclude_pattern,
                limit=source.limit,
            )
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
