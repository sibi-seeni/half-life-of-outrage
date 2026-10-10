"""Pre / event / post window summaries with bootstrap CIs, raw and engagement-weighted."""
import numpy as np
import pandas as pd


def _boot_mean(x, w=None, n_boot=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None, [None, None]
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        stats.append(np.average(x[i], weights=w[i]))
    return float(np.average(x, weights=w)), [float(np.quantile(stats, .025)), float(np.quantile(stats, .975))]


def window_table(df, events, pre=7, event_half=2, post=7, aspects=None, min_n=20):
    rows = []
    aspects = aspects or sorted(a for a in df["aspect"].unique() if a != "NotGame")
    for _, ev in events.iterrows():
        delta = (df["day"] - ev["date"]).dt.days
        win = pd.cut(delta, [-pre - 0.5, -event_half - 0.5, event_half + 0.5, post + 0.5],
                     labels=["pre", "event", "post"])
        for aspect in aspects:
            for w in ["pre", "event", "post"]:
                part = df[(win == w) & (df["aspect"] == aspect)]
                row = {"event_id": ev["event_id"], "aspect": aspect, "window": w, "n": len(part)}
                if len(part) >= min_n:
                    row["neg_share"], row["neg_share_ci95"] = _boot_mean(part["is_neg"])
                    row["mean_score"], row["mean_score_ci95"] = _boot_mean(part["sentiment_score"])
                    row["w_neg_share"], row["w_neg_share_ci95"] = _boot_mean(part["is_neg"], part["weight"])
                rows.append(row)
    return pd.DataFrame(rows)


def engagement_gap(df, aspect, n_boot=1000, seed=0):
    """Weighted minus raw negative share for one aspect, with a comment-level bootstrap CI."""
    part = df[df["aspect"] == aspect]
    if part.empty:
        return None
    x, w = part["is_neg"].values, part["weight"].values
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        gaps.append(np.average(x[i], weights=w[i]) - x[i].mean())
    return {"aspect": aspect, "n": int(len(x)), "raw_neg_share": float(x.mean()),
            "weighted_neg_share": float(np.average(x, weights=w)),
            "gap": float(np.average(x, weights=w) - x.mean()),
            "gap_ci95": [float(np.quantile(gaps, .025)), float(np.quantile(gaps, .975))]}
