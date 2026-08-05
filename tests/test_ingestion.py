import tempfile
import unittest
from pathlib import Path

from nexusnews.fetchers import APIFetcher, FetchError, RSSFetcher
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
