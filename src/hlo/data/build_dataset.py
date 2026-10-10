"""Build the analysis dataset from the SQLite DB.

Ports EDA.ipynb + Data_Wrangling.ipynb into one deterministic step:
  raw comments -> non-empty -> English (langdetect, seeded) -> >=3 words -> text-deduplicated
plus spike flags (rolling z-score on raw daily volume), engagement features and LLM context
(video title, first 500 chars of description, parent comment text).

Writes data/comments_clean.csv and results/data_flow.json (every count reported in the paper).
"""
import argparse
import json
import re
import sqlite3

import numpy as np
import pandas as pd

from hlo.config import DATA_DIR, DATASET_CSV, GAME_LAUNCH, RANDOM_SEED, RESULTS_DIR, ensure_dirs

QUERY = """
SELECT c.comment_id, c.text, c.published_at AS comment_date, c.author_hash, c.parent_id,
       c.like_count AS likes, v.video_id, v.title AS video_title, v.published_at AS video_date,
       v.channel_id, v.keyword_matched, v.description AS video_description
FROM comments c JOIN videos v ON c.video_id = v.video_id
"""


def load_raw(db_path):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(QUERY, conn)
    for col in ("comment_date", "video_date"):
        df[col] = pd.to_datetime(df[col])
    return df.sort_values("comment_date").reset_index(drop=True)


def detect_language(texts):
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = RANDOM_SEED  # langdetect is non-deterministic without a seed

    def _one(t):
        if not isinstance(t, str) or len(t.strip()) < 3:
            return "unknown"
        try:
            return detect(t)
        except LangDetectException:
            return "unknown"

    return texts.map(_one)


def spike_flags(dates, launch=GAME_LAUNCH, window=30, min_periods=7, z_pre=3.0, z_post=2.5):
    """Rolling z-score of daily comment volume. Post-launch uses a lower threshold because the
    baseline volume is orders of magnitude higher. Returns a per-day frame."""
    daily = dates.dt.normalize().value_counts().sort_index()
    daily = daily.asfreq("D", fill_value=0)
    mean = daily.rolling(window, min_periods=min_periods).mean()
    std = daily.rolling(window, min_periods=min_periods).std()
    z = (daily - mean) / std
    pre = daily.index < pd.Timestamp(launch)
    is_spike = (pre & (z > z_pre)) | (~pre & (z > z_post))
    return pd.DataFrame({"date": daily.index, "daily_volume": daily.values,
                         "z_score": z.values, "is_spike": is_spike.values})


def preprocess_text(text):
    """Mask URLs and @mentions, collapse whitespace (same as the RoBERTa preprocessing)."""
    text = re.sub(r"http\S+|www\S+", "[URL]", text)
    text = re.sub(r"@[\w.-]+", "[USER]", text)
    return " ".join(text.split())


def build(db_path, min_words=3):
    flow = {}
    df = load_raw(db_path)
    flow["raw_comments"] = len(df)
    flow["raw_videos"] = int(df["video_id"].nunique())
    flow["raw_channels"] = int(df["channel_id"].nunique())
    flow["raw_commenters"] = int(df["author_hash"].nunique())
    flow["date_min"], flow["date_max"] = str(df["comment_date"].min()), str(df["comment_date"].max())
    # Parent text comes from the raw table: a reply's parent may itself be filtered out (short/non-English).
    parent_map = df.dropna(subset=["text"]).set_index("comment_id")["text"].to_dict()

    # Spikes are computed on all raw comments (attention signal, independent of language filtering).
    spikes = spike_flags(df["comment_date"])
    df["date"] = df["comment_date"].dt.normalize()
    df = df.merge(spikes[["date", "z_score", "is_spike"]], on="date", how="left")

    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    flow["non_empty"] = len(df)

    df["language"] = detect_language(df["text"])
    flow["english"] = int((df["language"] == "en").sum())
    df["word_count"] = df["text"].str.split().str.len()
    df = df[(df["language"] == "en") & (df["word_count"] >= min_words)]
    flow[f"english_min{min_words}words"] = len(df)

    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    flow["deduplicated_final"] = len(df)

    df["text_preprocessed"] = df["text"].map(preprocess_text)
    df["parent_text"] = df["parent_id"].map(parent_map).map(
        lambda t: preprocess_text(t) if isinstance(t, str) else "")
    df["video_context"] = df["video_description"].fillna("").str[:500]
    df["like_count_log"] = np.log1p(df["likes"])
    df["comment_latency_days"] = (df["comment_date"] - df["video_date"]).dt.total_seconds() / 86400

    flow["final_videos"] = int(df["video_id"].nunique())
    flow["final_channels"] = int(df["channel_id"].nunique())
    flow["final_commenters"] = int(df["author_hash"].nunique())
    flow["final_replies"] = int(df["parent_id"].notna().sum())
    flow["spike_days"] = int(spikes["is_spike"].sum())

    cols = ["comment_id", "parent_id", "parent_text", "text_preprocessed", "video_id", "video_title",
            "video_context", "channel_id", "comment_date", "video_date", "author_hash", "likes",
            "like_count_log", "comment_latency_days", "is_spike", "z_score"]
    return df[cols], spikes, flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "arc_raiders.db"))
    ap.add_argument("--min-words", type=int, default=3)
    args = ap.parse_args()
    ensure_dirs()
    df, spikes, flow = build(args.db, args.min_words)
    df.to_csv(DATASET_CSV, index=False)
    spikes.to_csv(RESULTS_DIR / "daily_spikes.csv", index=False)
    (RESULTS_DIR / "data_flow.json").write_text(json.dumps(flow, indent=2))
    print(json.dumps(flow, indent=2))


if __name__ == "__main__":
    main()
