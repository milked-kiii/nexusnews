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
The built-in summarizer is a deterministic Chinese fallback suitable for dry runs.
Its `Summarizer` interface is the integration point for an approved LLM-backed
editorial implementation; no model credential is required or stored by this MVP.
