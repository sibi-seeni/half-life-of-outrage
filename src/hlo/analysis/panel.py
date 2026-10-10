"""Join the cleaned dataset with LLM labels and build daily aspect-level series."""
import numpy as np
import pandas as pd

from hlo.config import DATASET_CSV, EVENTS_CSV, LLM_LABELS_CSV, SENTIMENT_SCORE


def load_labeled(dataset=DATASET_CSV, labels=LLM_LABELS_CSV, min_conf=None):
    data = pd.read_csv(dataset, dtype={"comment_id": str}, parse_dates=["comment_date", "video_date"])
    llm = pd.read_csv(labels, dtype={"comment_id": str}).drop_duplicates("comment_id", keep="last")
    llm = llm[llm["parse_ok"].astype(str) == "True"]
    df = data.merge(llm[["comment_id", "game_related", "target", "category", "sentiment", "confidence"]],
                    on="comment_id", how="inner")
    df["aspect"] = np.where(df["game_related"] == "No", "NotGame", df["category"])
    df["sentiment_score"] = df["sentiment"].map(SENTIMENT_SCORE)
    df["is_neg"] = (df["sentiment"] == "Neg").astype(float)
    df["weight"] = np.log1p(df["likes"]) + 1  # engagement weight; likes are as of collection time
    df["day"] = df["comment_date"].dt.normalize()
    if min_conf is not None:
        df = df[(df["aspect"] == "NotGame") | (df["confidence"] >= min_conf)]
    return df


def load_events(path=EVENTS_CSV, include=("verified", "approximate")):
    ev = pd.read_csv(path)
    ev = ev[ev["date"].notna() & ev["status"].isin(include)].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    return ev.reset_index(drop=True)


def daily_series(df, aspect=None):
    """Per-day counts, negative share, mean score and engagement-weighted versions, zero-filled index."""
    sub = df[df["aspect"] == aspect] if aspect else df[df["aspect"] != "NotGame"]
    g = sub.groupby("day")
    out = pd.DataFrame({
        "n": g.size(),
        "n_neg": g["is_neg"].sum(),
        "score_sum": g["sentiment_score"].sum(),
        "w_sum": g["weight"].sum(),
        "w_neg_sum": g.apply(lambda x: (x["weight"] * x["is_neg"]).sum(), include_groups=False),
        "w_score_sum": g.apply(lambda x: (x["weight"] * x["sentiment_score"]).sum(), include_groups=False),
        "active_videos": g["video_id"].nunique(),
    })
    if out.empty:
        return out
    out = out.asfreq("D", fill_value=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["neg_share"] = out["n_neg"] / out["n"]
        out["mean_score"] = out["score_sum"] / out["n"]
        out["w_neg_share"] = out["w_neg_sum"] / out["w_sum"]
        out["w_mean_score"] = out["w_score_sum"] / out["w_sum"]
    return out
