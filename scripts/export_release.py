"""Build the public data release: IDs, timestamps and labels only (no comment text).

YouTube API Terms do not allow redistributing comment text; reviewers can rebuild the text with
scripts/rehydrate.py. Author hashes are re-salted so they cannot be matched to other datasets.
"""
import hashlib
import os
import secrets
from pathlib import Path

import pandas as pd

from hlo.config import DATA_DIR, DATASET_CSV, LLM_LABELS_CSV, ROBERTA_LABELS_CSV

OUT = Path(os.getenv("HLO_RELEASE_DIR", DATA_DIR / "release"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(16)  # not saved anywhere
    data = pd.read_csv(DATASET_CSV, dtype={"comment_id": str})
    llm = pd.read_csv(LLM_LABELS_CSV, dtype={"comment_id": str}).drop_duplicates("comment_id", keep="last")
    rel = data[["comment_id", "parent_id", "video_id", "channel_id", "comment_date", "likes", "author_hash"]].copy()
    rel["author_id"] = rel.pop("author_hash").map(lambda h: hashlib.sha256((salt + h).encode()).hexdigest()[:16])
    rel = rel.merge(llm[["comment_id", "game_related", "target", "category", "sentiment", "confidence",
                         "prompt_version"]], on="comment_id", how="left")
    if ROBERTA_LABELS_CSV.exists():
        rel = rel.merge(pd.read_csv(ROBERTA_LABELS_CSV, dtype={"comment_id": str}), on="comment_id", how="left")
    rel.to_csv(OUT / "comments_labels.csv", index=False)
    print(f"Wrote {len(rel):,} rows to {OUT / 'comments_labels.csv'}")


if __name__ == "__main__":
    main()
