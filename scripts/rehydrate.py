"""Re-fetch comment text for a released ID list via comments.list (50 IDs per call, 1 quota unit).

python scripts/rehydrate.py data/release/comments_labels.csv data/rehydrated.csv
Deleted or hidden comments will be missing; report the recovery rate when reusing the data.
"""
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build


def main(src, dst):
    load_dotenv()
    yt = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])
    ids = pd.read_csv(src, dtype={"comment_id": str})["comment_id"].tolist()
    rows = []
    for i in range(0, len(ids), 50):
        resp = yt.comments().list(part="snippet", id=",".join(ids[i:i + 50]), textFormat="plainText").execute()
        rows += [{"comment_id": it["id"], "text": it["snippet"]["textDisplay"]} for it in resp.get("items", [])]
    pd.DataFrame(rows).to_csv(dst, index=False)
    print(f"Recovered {len(rows):,} / {len(ids):,} comments ({len(rows) / max(len(ids), 1):.1%})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
