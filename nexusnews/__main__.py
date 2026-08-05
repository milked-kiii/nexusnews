from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

from .config import load_config
from .fetchers import LocalOrUrlTransport, UrlLibTransport
from .pipeline import run
from .storage import SQLiteItemStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and optionally deliver a Nexusnews daily digest")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-feedback", metavar="JSON", help="record a Feishu callback payload")
    parser.add_argument("--feedback-stats", metavar="DIGEST_ID", help="print item feedback stats")
    args = parser.parse_args()
    Path("var").mkdir(exist_ok=True)
    logging.basicConfig(filename="var/nexusnews.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.record_feedback:
        payload = json.loads(args.record_feedback)
        with SQLiteItemStore(config.database) as store:
            store.record_feedback(**payload)
        print("feedback recorded")
    elif args.feedback_stats:
        with SQLiteItemStore(config.database) as store:
            print(json.dumps(store.feedback_rate(args.feedback_stats), ensure_ascii=False))
    else:
        print(run(config, LocalOrUrlTransport(UrlLibTransport()), dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
