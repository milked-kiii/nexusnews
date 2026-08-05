from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .fetchers import UrlLibTransport
from .pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and optionally deliver a Nexusnews daily digest")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    Path("var").mkdir(exist_ok=True)
    logging.basicConfig(filename="var/nexusnews.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(load_config(args.config), UrlLibTransport(), dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
