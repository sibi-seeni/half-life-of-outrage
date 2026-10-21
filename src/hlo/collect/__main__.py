"""python -m hlo.collect {discovery|comments|full|mark-done} [--first-from YYYY-MM-DD] [--first-to YYYY-MM-DD]

mark-done: one-off for an existing database, marks videos that already have comments as collected.
--first-from/--first-to: collect videos uploaded in this range first (end date exclusive).
"""
import argparse
import logging
from datetime import datetime

import sys

from hlo.collect.api import DailyQuotaExhausted
from hlo.collect.collector import collect_comments, mark_collected_from_existing
from hlo.collect.database import init_db
from hlo.collect.discovery import search_new_videos

logging.basicConfig(filename="collection_log.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Arc Raiders YouTube collection pipeline")
    parser.add_argument("mode", choices=["discovery", "comments", "full", "mark-done"])
    parser.add_argument("--first-from", type=datetime.fromisoformat, help="upload date to prioritise from")
    parser.add_argument("--first-to", type=datetime.fromisoformat, help="upload date to prioritise to (exclusive)")
    args = parser.parse_args()

    init_db()
    if args.mode == "mark-done":
        mark_collected_from_existing()
        return
    try:
        if args.mode in ("discovery", "full"):
            logging.info("Video discovery started")
            search_new_videos()
        if args.mode in ("comments", "full"):
            logging.info("Comment collection started")
            collect_comments(args.first_from, args.first_to)
    except DailyQuotaExhausted:
        logging.info("Stopped: daily quota exhausted (%s)", args.mode)
        sys.exit(2)
    logging.info("Run finished: %s", args.mode)


if __name__ == "__main__":
    main()
