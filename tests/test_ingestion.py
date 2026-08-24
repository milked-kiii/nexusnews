import tempfile
import unittest
from pathlib import Path

from nexusnews.fetchers import APIFetcher, FetchError, RSSFetcher, parse_discord, parse_github, parse_reddit, parse_x
from nexusnews.models import RawItem, normalize_item
from nexusnews.storage import SQLiteItemStore


class FakeTransport:
    def __init__(self, payload=b"", error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def get(self, url, *, timeout, headers):
        self.calls.append((url, timeout, headers))
        if self.error:
            raise self.error
        return self.payload


class FetcherTests(unittest.TestCase):
    def test_rss_and_atom_are_parsed(self):
        payload = b'''<rss><channel><item><title>  Hello  world </title><link>https://example.com/a</link><description>Body</description><guid>42</guid><pubDate>Tue, 05 Aug 2025 10:00:00 GMT</pubDate></item></channel></rss>'''
        transport = FakeTransport(payload)
        items = RSSFetcher(transport, timeout=2).fetch("https://feed", source="Example")
        self.assertEqual(items[0].external_id, "42")
        self.assertEqual(transport.calls[0][1], 2)

    def test_rss_content_html_is_stripped(self):
        payload = b'''<rss><channel><item><title>T</title><link>https://example.com/a</link><description>&lt;table&gt;&lt;tr&gt;&lt;td&gt;hello&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt; world</description></item></channel></rss>'''
        items = RSSFetcher(FakeTransport(payload)).fetch("https://feed", source="Example")
        self.assertEqual(items[0].content, "hello world")

    def test_invalid_rss_has_actionable_error(self):
        with self.assertRaisesRegex(FetchError, "invalid RSS/XML"):
            RSSFetcher(FakeTransport(b"<broken")).fetch("https://feed", source="x")

    def test_api_parser_is_injected(self):
        fetcher = APIFetcher(
            FakeTransport(b'{"stories":[{"headline":"News","id":7}]}'),
            lambda data: [RawItem(source="api", title=row["headline"], external_id=str(row["id"])) for row in data["stories"]],
        )
        self.assertEqual(fetcher.fetch("https://api")[0].title, "News")

    def test_api_schema_and_transport_errors_are_wrapped(self):
        with self.assertRaisesRegex(FetchError, "expected schema"):
            APIFetcher(FakeTransport(b"{}"), lambda data: data["missing"]).fetch("https://api")
        with self.assertRaisesRegex(FetchError, "timeout"):
            APIFetcher(FakeTransport(error=FetchError("timeout fetching https://api")), list).fetch("https://api")

    def test_platform_payloads_are_converted_to_raw_items(self):
        reddit = parse_reddit({"data": {"children": [{"data": {"name": "t3_1", "title": "Post", "permalink": "/r/test/1", "selftext": "Body", "created_utc": 0}}]}}, source="Reddit")
        self.assertEqual(reddit[0].url, "https://www.reddit.com/r/test/1")
        self.assertEqual(reddit[0].published_at, "1970-01-01T00:00:00+00:00")

    def test_reddit_skips_stickied_megathreads(self):
        data = {"data": {"children": [
            {"data": {"name": "t3_pinned", "title": "Weekly megathread", "stickied": True, "created_utc": 0}},
            {"data": {"name": "t3_news", "title": "Fresh post", "permalink": "/r/test/2", "created_utc": 1}},
        ]}}
        items = parse_reddit(data, source="Reddit")
        self.assertEqual([item.external_id for item in items], ["t3_news"])

    def test_github_search_payload_is_converted(self):
        data = {"items": [{
            "id": 7, "node_id": "R_7", "full_name": "acme/hot-repo",
            "description": "Fast LLM serving", "html_url": "https://github.com/acme/hot-repo",
            "stargazers_count": 1234, "language": "Python", "topics": ["llm", "inference"],
            "created_at": "2026-08-20T01:02:03Z",
        }]}
        items = parse_github(data, source="GitHub 热榜")
        self.assertEqual(items[0].title, "acme/hot-repo: Fast LLM serving")
        self.assertEqual(items[0].url, "https://github.com/acme/hot-repo")
        self.assertEqual(items[0].published_at, "2026-08-20T01:02:03Z")
        self.assertEqual(items[0].external_id, "R_7")
        self.assertIn("⭐ 1234", items[0].content)
        x = parse_x({"data": [{"id": "2", "text": "Hello", "author_id": "3", "created_at": "2025-08-05T00:00:00Z"}], "includes": {"users": [{"id": "3", "username": "alice"}]}}, source="X")
        self.assertEqual(x[0].url, "https://x.com/alice/status/2")
        discord = parse_discord([{ "id": "4", "content": "News", "timestamp": "2025-08-05T00:00:00Z", "guild_id": "5", "author": {"username": "bob"}}], source="Discord", channel_id="6")
        self.assertEqual(discord[0].url, "https://discord.com/channels/5/6/4")


class NormalizeAndStorageTests(unittest.TestCase):
    def test_normalization_and_dedupe_are_deterministic(self):
        first = normalize_item(RawItem(source=" Feed ", title=" Hello   World ", url="HTTPS://EXAMPLE.COM/a/?utm_source=x&b=2&a=1", content=" x ", published_at="2025-08-05T10:00:00+08:00"))
        second = normalize_item(RawItem(source="Feed", title="Hello World", url="https://example.com/a?a=1&b=2", content="x", published_at="2025-08-05T02:00:00Z"))
        self.assertEqual(first.dedupe_key, second.dedupe_key)
        self.assertEqual(first.published_at, "2025-08-05T02:00:00Z")

    def test_required_fields_and_bad_date_fail(self):
        with self.assertRaisesRegex(ValueError, "title is required"):
            normalize_item(RawItem(source="x", title=" "))
        with self.assertRaisesRegex(ValueError, "invalid published_at"):
            normalize_item(RawItem(source="x", title="y", published_at="not-a-date"))

    def test_sqlite_is_durable_and_ignores_duplicates(self):
        item = normalize_item(RawItem(source="x", title="Story", external_id="1"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.db"
            with SQLiteItemStore(path) as store:
                self.assertTrue(store.put(item))
                self.assertFalse(store.put(item))
            with SQLiteItemStore(path) as reopened:
                self.assertEqual(reopened.list(), [item])


if __name__ == "__main__":
    unittest.main()
