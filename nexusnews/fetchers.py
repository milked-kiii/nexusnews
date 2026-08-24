from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
import json
import os
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import xml.etree.ElementTree as ET

from .config import Source
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

# Browser-style UA: several hosts (notably reddit.com) throttle or reject
# obvious bot agents with 403/429 even for public feeds.
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class FetchError(RuntimeError):
    """A source could not be fetched or decoded."""


class Transport(Protocol):
    def get(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes: ...


class UrlLibTransport:
    """urllib-based transport with per-host politeness throttling.

    Hosts in ``throttled_hosts`` (e.g. reddit.com, which answers bursts with
    HTTP 429) get a minimum interval between requests; a 429 from any host is
    retried with escalating backoff — reddit's limiter typically refills its
    bucket within a minute.
    """

    throttled_hosts = frozenset({"reddit.com", "www.reddit.com", "old.reddit.com"})

    def __init__(self, host_min_interval: float = 15.0,
                 retry_backoffs: tuple[float, ...] = (20.0, 45.0, 90.0)):
        self.host_min_interval = host_min_interval
        self.retry_backoffs = retry_backoffs
        self._last_request: dict[str, float] = {}

    def get(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes:
        host = urlsplit(url).netloc.lower()
        if host in self.throttled_hosts:
            elapsed = time.monotonic() - self._last_request.get(host, float("-inf"))
            if elapsed < self.host_min_interval:
                time.sleep(self.host_min_interval - elapsed)
            self._last_request[host] = time.monotonic()
        try:
            return self._open(url, timeout=timeout, headers=headers)
        except FetchError as exc:
            if "HTTP 429" not in str(exc):
                raise
        for backoff in self.retry_backoffs:
            time.sleep(backoff)
            self._last_request[host] = time.monotonic()
            try:
                return self._open(url, timeout=timeout, headers=headers)
            except FetchError as exc:
                if "HTTP 429" not in str(exc):
                    raise
        raise FetchError(f"HTTP 429 fetching {url} (retries exhausted)")

    def _open(self, url: str, *, timeout: float, headers: Mapping[str, str]) -> bytes:
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
            "User-Agent": _BROWSER_UA,
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
            content = _child_text(entry, "description", "summary", "content")
            result.append(RawItem(
                source=source,
                title=_child_text(entry, "title") or "",
                url=link,
                # Feeds like reddit's embed HTML tables in <content>; collapse
                # to plain text so summaries and LLM prompts stay readable.
                content=_strip_html(content) if content else None,
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
            "User-Agent": _BROWSER_UA,
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


def parse_reddit(data: object, *, source: str, min_score: int = 0) -> Sequence[RawItem]:
    children = data["data"]["children"]  # type: ignore[index]
    result = []
    for child in children:
        post = child["data"]
        # Hot/top listings pin megathreads (weekly discussions, rules) that are
        # not news; skip them so the digest stays timely.
        if post.get("stickied"):
            continue
        # Upvote threshold: filter low-engagement posts when configured.
        score = post.get("score")
        if isinstance(score, int) and score < min_score:
            continue
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


def parse_github_trending(data: object, *, source: str) -> Sequence[RawItem]:
    """Convert GitHub trending page HTML into RawItems.

    GitHub trending has no official API. The HTML contains <article> tags
    with repo links like <a href="/owner/repo">. We extract those, skipping
    non-repo paths like /sponsors/, /trending/developers, etc.
    """
    text = data if isinstance(data, str) else ""
    result = []
    seen: set[str] = set()

    # Match /owner/repo links, excluding known non-repo prefixes
    _skip = {"sponsors", "trending", "settings", "login", "signup", "features",
             "about", "pricing", "explore", "topics", "collections", "events",
             "marketplace", "apps", "new", "organizations", "users"}
    link_pattern = re.compile(r'href="/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)"')

    for m in link_pattern.finditer(text):
        full_name = m.group(1)
        owner = full_name.split("/")[0]
        if owner.lower() in _skip:
            continue
        if full_name in seen:
            continue
        seen.add(full_name)

        # Try to extract description from nearby context
        start = max(0, m.start() - 3000)
        context = text[start:m.end() + 3000]
        # Look for a <p> tag after the link
        desc_m = re.search(
            r'<p[^>]*>(.*?)</p>',
            context[m.end() - start:],
            re.DOTALL | re.IGNORECASE,
        )
        description = _strip_html(desc_m.group(1)).strip() if desc_m else None
        # Clean up description: remove extra whitespace, truncate
        if description:
            description = re.sub(r'\s+', ' ', description)[:200]

        # Use date-stamped external_id so each day's trending list is treated as
        # fresh items, while same-repo within one day dedupes correctly.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result.append(RawItem(
            source=source,
            title=full_name,
            url=f"https://github.com/{full_name}",
            content=description,
            # Trending page is daily-refreshed; use fetch time as published_at
            # so items aren't buried by recency-sorted selection.
            published_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            external_id=f"{full_name}#{today}",
        ))
        if len(result) >= 30:
            break
    return result


def parse_github_search(data: object, *, source: str) -> Sequence[RawItem]:
    """Convert GitHub repository search results into RawItems."""
    result = []
    for repo in data.get("items", []):  # type: ignore[union-attr]
        full_name = _string(repo.get("full_name"))
        if not full_name:
            continue
        description = _string(repo.get("description"))
        # Pack repo metadata into content prefix for card display:
        # "created:YYYY-MM-DD stars:N ⭐ · language · topics · description"
        created_at = _string(repo.get("created_at"))
        created_date = created_at[:10] if created_at else "未知"
        star_count = repo.get("stargazers_count")
        star_str = str(star_count) if isinstance(star_count, int) else "0"
        meta = [f"created:{created_date} stars:{star_str} ⭐"]
        language = _string(repo.get("language"))
        if language:
            meta.append(language)
        topics = repo.get("topics")
        if isinstance(topics, list) and topics:
            meta.append(" ".join(f"#{topic}" for topic in topics[:5] if isinstance(topic, str)))
        if description:
            meta.append(description)
        result.append(RawItem(
            source=source,
            title=f"{full_name}: {description}" if description else full_name,
            url=_string(repo.get("html_url")),
            content=" · ".join(meta) or None,
            # Use pushed_at (last commit) as the freshness signal, not created_at.
            # A repo created weeks ago but pushed today is still news.
            published_at=_string(repo.get("pushed_at")) or _string(repo.get("updated_at")) or _string(repo.get("created_at")),
            external_id=_string(repo.get("node_id")) or (str(repo["id"]) if "id" in repo else None),
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

    def fetch(self, source: Source) -> list[RawItem]:
        kind = source.kind
        if kind in {"rss", "medium"}:
            return RSSFetcher(self.transport, timeout=self.timeout).fetch(source.url, source=source.name)  # type: ignore[arg-type]
        if kind == "webpage":
            return WebpageFetcher(self.transport, timeout=self.timeout).fetch(
                source.url, source=source.name,  # type: ignore[arg-type]
                link_pattern=source.link_pattern,  # type: ignore[arg-type]
                title_group=source.title_group,
                exclude_pattern=source.exclude_pattern,
                limit=source.limit,
            )
        if kind == "reddit":
            # Reddit JSON API is Cloudflare-blocked; fall back to .rss feed.
            # Note: RSS lacks upvote score, so min_score filtering is skipped.
            if source.subreddit:
                url = f"https://www.reddit.com/r/{source.subreddit}/{source.sort}/.rss?limit={source.limit}"
            else:
                url = source.url or ""
            return RSSFetcher(self.transport, timeout=self.timeout).fetch(url, source=source.name)
        if kind == "github_trending":
            # Scrape GitHub trending page; no official API exists.
            # Optional language filter via query param, e.g. ?language=python
            params = {}
            if source.query:
                params["language"] = source.query
            qs = f"?{urlencode(params)}" if params else ""
            # Use WebpageFetcher-style HTML scraping, not APIFetcher (which sends Accept: application/json)
            html = self.transport.get(
                f"https://github.com/trending{qs}",
                timeout=self.timeout,
                headers={"Accept": "text/html,application/xhtml+xml", "User-Agent": _BROWSER_UA},
            ).decode("utf-8", errors="ignore")
            return list(parse_github_trending(html, source=source.name))
        if kind == "github_search":
            # GitHub Search API: find EMERGING repos — recently created (last 30 days)
            # AND actively pushed (last 24h) with meaningful star traction.
            # This filters out established projects like Dify/Codex that push daily.
            created_since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
            pushed_since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d")
            base_query = source.query or "agent OR coding OR copilot OR MCP"
            params = urlencode({
                "q": f"{base_query} created:>{created_since} pushed:>{pushed_since} stars:>20",
                "sort": "stars",
                "order": "desc",
                "per_page": source.limit,
            })
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "Nexusnews/1.0 (read-only news digest)",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            github_token = os.environ.get(source.token_env or "")
            if github_token:
                headers["Authorization"] = f"Bearer {github_token}"
            return APIFetcher(
                self.transport,
                lambda data: parse_github_search(data, source=source.name),
                timeout=self.timeout,
                headers=headers,
            ).fetch(f"https://api.github.com/search/repositories?{params}")
        if kind == "github":
            # Legacy GitHub search (kept for backward compat); hot-new-repos proxy:
            # repos created in the last 7 days, sorted by stars.
            since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            base_query = source.query or "stars:>50"
            params = urlencode({
                "q": f"{base_query} created:>{since}",
                "sort": "stars",
                "order": "desc",
                "per_page": source.limit,
            })
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "Nexusnews/1.0 (read-only news digest)",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            github_token = os.environ.get(source.token_env or "")
            if github_token:
                headers["Authorization"] = f"Bearer {github_token}"
            return APIFetcher(
                self.transport,
                lambda data: parse_github_search(data, source=source.name),
                timeout=self.timeout,
                headers=headers,
            ).fetch(f"https://api.github.com/search/repositories?{params}")

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
                lambda data: parse_discord(data, source=source.name, channel_id=source.channel_id),  # type: ignore[arg-type]
                timeout=self.timeout,
                headers={"Authorization": f"Bot {token}"},
            ).fetch(url)
        raise FetchError(f"unsupported source kind: {kind}")
