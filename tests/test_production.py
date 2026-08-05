import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nexusnews.llm import OpenAICompatibleSummarizer, with_fallback
from nexusnews.models import RawItem, normalize_item
from nexusnews.storage import SQLiteItemStore


class Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def read(self): return json.dumps(self.payload, ensure_ascii=False).encode()


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.item = normalize_item(RawItem(source="OpenAI", title="Model update", url="https://example.com/x", content="facts"))

    def test_llm_success_enforces_and_returns_editorial_fields(self):
        summary = "这" * 60
        why = "重" * 35
        transport = lambda request, timeout: Response({"choices": [{"message": {"content": json.dumps({"title": "中文标题", "summary": summary, "why_important": why})}}]})
        with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}):
            entry = OpenAICompatibleSummarizer("https://llm.invalid", "model", "TEST_LLM_KEY", transport=transport)(self.item)
        self.assertEqual((len(entry.summary), len(entry.why_important)), (60, 35))
        self.assertEqual(entry.item_id, self.item.id)

    def test_llm_failure_uses_explicit_local_fallback(self):
        primary = OpenAICompatibleSummarizer("https://llm.invalid", "model", "ABSENT_TEST_KEY")
        with patch.dict(os.environ, {}, clear=True):
            entry = with_fallback(primary)(self.item)
        self.assertIn("原文", entry.summary)
        self.assertTrue(60 <= len(entry.summary) <= 110)
        self.assertTrue(35 <= len(entry.why_important) <= 75)

    def test_feedback_latest_vote_wins_and_rate_is_per_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteItemStore(Path(directory) / "db.sqlite") as store:
                args = {"digest_id": "d1", "scope": "item", "item_id": "i1", "user_id": "u1"}
                store.record_feedback(vote="down", **args)
                store.record_feedback(vote="up", **args)
                store.record_feedback(digest_id="d1", scope="item", item_id="i2", user_id="u2", vote="down")
                self.assertEqual(store.feedback_rate("d1"), {"up": 1, "down": 1, "total": 2, "rate": .5})


if __name__ == "__main__":
    unittest.main()
