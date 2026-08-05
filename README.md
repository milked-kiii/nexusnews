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

## End-to-end daily digest

Requires Python 3.10+ and no third-party packages. Copy `config.example.json` to
`config.json`, review the placeholder source list, then run:

```sh
python3 -m nexusnews --config config.json --dry-run
```

The command fetches configured RSS sources, stores normalized items in
`var/nexusnews.db`, selects 5–10 unique items from the last 24 hours, and writes a
Chinese preview to `var/latest-digest.txt`. Operational logs are in
`var/nexusnews.log`. A run fails rather than padding the digest with duplicates if
fewer than the configured minimum unique recent stories exist.

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

## Production smoke test and 14-day trial

Local Board must inject `FEISHU_WEBHOOK_URL` and, when LLM mode is enabled,
`NEXUSNEWS_LLM_API_KEY`. Validate configuration without exposing values:

```sh
test -n "$FEISHU_WEBHOOK_URL" && test -n "$NEXUSNEWS_LLM_API_KEY"
python3 -m nexusnews --config config.json --dry-run
python3 -m nexusnews --config config.json
tail -n 50 var/nexusnews.log
```

Confirm the target received 5–10 items and manually check summary/impact lengths.
Install `scripts/nexusnews-daily.cron` after replacing its absolute path; it fixes
the schedule at 08:30 Asia/Shanghai. For 14 days, check delivery and source errors
daily, review item feedback rate every 7 days, and investigate any failed run before
the next schedule. Never paste webhook URLs, tokens, or full callback user IDs into
issues or logs.
