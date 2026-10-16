"""python -m hlo.collect {discovery|comments|full}"""
import argparse
import logging

import sys

from hlo.collect.api import DailyQuotaExhausted
from hlo.collect.collector import collect_comments
from hlo.collect.database import init_db
from hlo.collect.discovery import search_new_videos

logging.basicConfig(filename="collection_log.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Arc Raiders YouTube collection pipeline")
    parser.add_argument("mode", choices=["discovery", "comments", "full"])
    args = parser.parse_args()

    init_db()
    try:
        if args.mode in ("discovery", "full"):
            logging.info("Video discovery started")
            search_new_videos()
        if args.mode in ("comments", "full"):
            logging.info("Comment collection started")
            collect_comments()
    except DailyQuotaExhausted:
        logging.info("Stopped: daily quota exhausted (%s)", args.mode)
        sys.exit(2)
    logging.info("Run finished: %s", args.mode)


if __name__ == "__main__":
    main()
