import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nexusnews.config import Config, Source, load_config
from nexusnews.digest import DigestEntry, render_digest, select_items
from nexusnews.models import RawItem, normalize_item
from nexusnews.pipeline import run


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, *, timeout, headers):
        return self.payload


def item(title, url, published="2026-08-05T01:00:00Z", source="Source"):
    return normalize_item(RawItem(source=source, title=title, url=url, content="正文", published_at=published))


class DigestTests(unittest.TestCase):
    def test_title_and_url_duplicates_are_collapsed_deterministically(self):
        values = [
            item("OpenAI launches GPT 6 model", "https://a.example/1", source="A"),
            item("OpenAI launches new GPT 6 model", "https://b.example/2", source="B"),
            item("Different robotics research", "https://c.example/3"),
            item("Another title", "https://c.example/3"),
        ]
        selected = select_items(values, minimum=1, maximum=10, title_threshold=.5)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].source, "Source")

    def test_output_contains_all_required_fields(self):
        text = render_digest([DigestEntry("标题", "来源", "https://example.com", "中文摘要", "影响解读")], generated_at=datetime(2026, 8, 5, tzinfo=timezone.utc))
        for expected in ("🤖 AI 日报｜2026-08-05（共 1 条）", "标题", "来源：来源", "原文：[阅读原文](https://example.com)", "摘要：中文摘要", "为什么重要：影响解读", "1好 / 1差", "本期好 / 本期差"):
            self.assertIn(expected, text)

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"sources": [{"name": "x", "url": "https://x"}], "minimum": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection"):
                load_config(path)

    def test_pipeline_dry_run_is_offline_testable_and_persistent(self):
        topics = ("robotics", "medicine", "coding", "hardware", "research", "productivity")
        rows = "".join(
            f"<item><title>News about {topic}</title><link>https://example.com/{topic}</link><description>{topic} 新闻内容</description><pubDate>Wed, 05 Aug 2026 01:00:00 GMT</pubDate></item>"
            for topic in topics
        )
        payload = f"<rss><channel>{rows}</channel></rss>".encode()
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                sources=(Source("Test", "https://feed.invalid"),),
                database=str(Path(directory) / "items.db"),
                output=str(Path(directory) / "digest.txt"),
                minimum=5,
                maximum=5,
            )
            text = run(config, FakeTransport(payload), dry_run=True, now=datetime(2026, 8, 5, 2, tzinfo=timezone.utc))
            self.assertEqual(text.count("为什么重要："), 5)
            self.assertEqual(Path(config.output).read_text(encoding="utf-8"), text + "\n")


if __name__ == "__main__":
    unittest.main()
