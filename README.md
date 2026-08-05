# Nexusnews

Minimal ingestion layer for RSS and JSON APIs, with deterministic normalization,
deduplication, and durable local SQLite storage. It uses only the Python 3 standard
library.

```python
from nexusnews.fetchers import RSSFetcher, UrlLibTransport
from nexusnews.models import normalize_item
from nexusnews.storage import SQLiteItemStore

raw_items = RSSFetcher(UrlLibTransport()).fetch("https://example.com/feed.xml", source="example")
with SQLiteItemStore("nexusnews.db") as store:
    inserted = store.put_many(normalize_item(raw) for raw in raw_items)
```

API sources use `APIFetcher(transport, parser)`, where `parser` converts decoded JSON
into `RawItem` values. Inject a custom `Transport` in tests or for alternative HTTP
clients. Network and XML/JSON failures raise `FetchError`; invalid normalized fields
raise `ValueError`.

Run tests without network access:

```sh
python3 -m unittest discover -s tests -v
```

## Offline end-to-end demo

Requires Python 3.10+ and no third-party packages, network, webhook, or API key.
From the repository root, run:

```sh
python3 -m nexusnews --config config.demo.json --dry-run
```

Expected result: seven structured stories are printed and written to
`var/demo-digest.txt`; the deliberately duplicated sample story appears once. Each
story includes title, source, first-party link, Chinese summary, and why it matters.
Normalized items persist in `var/demo.db`, and logs go to `var/nexusnews.log`.
Delete those three generated files when you want a clean local run.

The bundled sample uses future publication dates only so it remains inside the
rolling 24-hour selector whenever the demo is run. It is synthetic demonstration
content, not a current-news fixture. The deterministic local summarizer is useful
for validating the pipeline, but its editorial quality is below a configured LLM.

## Optional live sources and integrations

Copy `config.example.json` to `config.json` and review its source list to use live
RSS. This mode needs network access:

```sh
python3 -m nexusnews --config config.json --dry-run
```

The command stores normalized items in `var/nexusnews.db`, selects 5–10 unique
items from the last 24 hours, and writes `var/latest-digest.txt`. A run fails rather
than padding the digest with duplicates if all sources fail; if fewer than the
configured minimum unique stories pass selection, it emits an explicit empty-day
message.

For delivery, create a Feishu custom-bot webhook and export it only in the runtime
environment (never in config or source control):

```sh
export FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/...'
python3 -m nexusnews --config config.json
```

Delivery retries three times with exponential backoff. `scripts/nexusnews-daily.cron`
is an installable scheduling example; replace its absolute project path first.
Configure an OpenAI-compatible `llm_endpoint` and `llm_model` to generate the
editorial summary. Export `NEXUSNEWS_LLM_API_KEY` only in the runtime environment.
Missing credentials, invalid responses, network errors, and editorial length
violations automatically use the deterministic local fallback; secrets are never
written to SQLite or logs.

Feishu interactive callbacks should pass their verified callback fields to the
feedback recorder (the surrounding callback service is responsible for signature
verification and mapping Feishu's user ID):

```sh
python3 -m nexusnews --config config.json --record-feedback \
  '{"digest_id":"2026-08-05","scope":"item","item_id":"ITEM_ID","vote":"up","user_id":"FEISHU_USER_ID"}'
python3 -m nexusnews --config config.json --feedback-stats 2026-08-05
```

The latest vote from the same user for the same item/digest replaces the earlier
vote. Text replies such as `1好` / `1差` and `本期好` / `本期差` can be mapped to the
same recorder by the callback service.

## Optional delivery smoke test

Local Board must inject `FEISHU_WEBHOOK_URL` and, when LLM mode is enabled,
`NEXUSNEWS_LLM_API_KEY`. Validate configuration without exposing values:

```sh
test -n "$FEISHU_WEBHOOK_URL" && test -n "$NEXUSNEWS_LLM_API_KEY"
python3 -m nexusnews --config config.json --dry-run
python3 -m nexusnews --config config.json
tail -n 50 var/nexusnews.log
```

Confirm the target received 5–10 items and manually check summary/impact lengths.
`scripts/nexusnews-daily.cron` remains an optional scheduling example; production
deployment and trial monitoring are outside this local Demo. Never paste webhook
URLs, tokens, or full callback user IDs into issues or logs.
