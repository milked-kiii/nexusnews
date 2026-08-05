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
