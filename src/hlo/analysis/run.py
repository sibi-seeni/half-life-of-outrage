"""Run the full temporal analysis and write every paper number to results/.

python -m hlo.analysis.run [--min-conf 0.8] [--horizon 60] [--tag robust_conf80]

Outputs (suffixed with --tag if given):
  results/half_life.json      per event x aspect: sentiment half-life (+CI, model-free) and attention half-life
  results/its.json            per event x aspect: level / slope change with HAC CIs
  results/windows.csv         pre / event / post negativity with CIs, raw and engagement-weighted
  results/engagement_gap.json weighted minus raw negative share per aspect
  results/label_summary.json  label distribution, confidence distribution, coverage
  figures/timeline.png, figures/decay_<event>.png
"""
import argparse
import json

import numpy as np

from hlo.analysis.decay import event_half_life
from hlo.analysis.its import segmented_regression
from hlo.analysis.panel import daily_series, load_events, load_labeled
from hlo.analysis.windows import engagement_gap, window_table
from hlo.config import FIGURES_DIR, RESULTS_DIR, ensure_dirs

ASPECTS = ["Game_Related", "Business_Model_Related", "AI_Voice_Related"]


def label_summary(df):
    game = df[df["aspect"] != "NotGame"]
    conf = game["confidence"].astype(float)
    return {
        "n_labeled": int(len(df)),
        "game_related_share": float((df["aspect"] != "NotGame").mean()),
        "target_counts": df["target"].value_counts().to_dict(),
        "aspect_counts": game["aspect"].value_counts().to_dict(),
        "sentiment_by_aspect": game.groupby("aspect")["sentiment"].value_counts(normalize=True)
                                   .round(4).unstack().to_dict(orient="index"),
        # denominators are game-related rows with a confidence value (the old notebook divided by all rows)
        "confidence_quantiles": conf.quantile([.05, .25, .5, .75, .95]).round(3).to_dict(),
        "confidence_retention": {f">={t}": float((conf >= t).mean()) for t in (.6, .7, .8, .9)},
        "confidence_unique_values": int(conf.nunique()),
    }


def plot_timeline(df, events, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    for aspect in ASPECTS:
        s = daily_series(df, aspect)
        if s.empty:
            continue
        ns = s["n_neg"].rolling(14, min_periods=1).sum() / s["n"].rolling(14, min_periods=1).sum()
        ax.plot(ns.index, ns.values, label=aspect.replace("_", " "), linewidth=1.5)
    for _, ev in events.iterrows():
        ax.axvline(ev["date"], color="0.4", linestyle=":", linewidth=0.8)
        ax.text(ev["date"], 1.0, ev["event_id"], rotation=90, va="top", ha="right", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Negative share (14-day pooled)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_decay(series, ev, res, path):
    import matplotlib.pyplot as plt
    if res.get("status") != "ok":
        return
    peak = np.datetime64(res["peak_day"])
    seg = series.loc[ev["date"] - np.timedelta64(30, "D"): peak + np.timedelta64(60, "D")]
    seg = seg[seg["n"] >= 5]
    t = (seg.index - peak).days.values
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(t, seg["neg_share"], s=np.clip(seg["n"], 5, 80), alpha=0.5, label="daily negative share")
    tt = np.arange(0, t.max() + 1)
    ax.plot(tt, res["baseline"] + res["A"] * np.exp(-res["lambda"] * tt), color="C3",
            label=f"fit, t½ = {res['half_life']:.1f} d")
    ax.axhline(res["baseline"], color="0.5", linestyle="--", linewidth=0.8, label="pre-event baseline")
    ax.set_xlabel("Days since post-event peak")
    ax.set_ylabel("Negative share")
    ax.set_title(f"{ev['label']}", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--baseline", type=int, default=30)
    ap.add_argument("--its-window", type=int, default=60)
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ensure_dirs()
    sfx = f"_{args.tag}" if args.tag else ""

    df = load_labeled(min_conf=args.min_conf)
    status = ("verified", "approximate", "unverified") if args.include_unverified else ("verified", "approximate")
    events = load_events(include=status)
    series = {a: daily_series(df, a) for a in ASPECTS}

    half, its = [], []
    for _, ev in events.iterrows():
        for aspect, s in series.items():
            if s.empty:
                continue
            key = {"event_id": ev["event_id"], "aspect": aspect, "event_aspect": ev["aspect"]}
            sent = event_half_life(s, ev["date"], "neg_share", args.baseline, args.horizon)
            attn = event_half_life(s, ev["date"], "n", args.baseline, args.horizon, n_boot=0)
            half.append(key | {"sentiment": sent, "attention": attn})
            its.append(key | segmented_regression(s, ev["date"], window=args.its_window))
            if aspect == ev["aspect"]:
                plot_decay(s, ev, sent, FIGURES_DIR / f"decay_{ev['event_id']}{sfx}.png")

    (RESULTS_DIR / f"half_life{sfx}.json").write_text(json.dumps(half, indent=2, default=str))
    (RESULTS_DIR / f"its{sfx}.json").write_text(json.dumps(its, indent=2, default=str))
    window_table(df, events).to_csv(RESULTS_DIR / f"windows{sfx}.csv", index=False)
    gaps = [g for g in (engagement_gap(df, a) for a in ASPECTS) if g]
    (RESULTS_DIR / f"engagement_gap{sfx}.json").write_text(json.dumps(gaps, indent=2))
    (RESULTS_DIR / f"label_summary{sfx}.json").write_text(json.dumps(label_summary(df), indent=2, default=str))
    plot_timeline(df, events, FIGURES_DIR / f"timeline{sfx}.png")

    for h in half:
        s = h["sentiment"]
        hl = f"{s['half_life']:.1f} d {s.get('half_life_ci95', '')}" if s.get("status") == "ok" else s["status"]
        print(f"{h['event_id']:<20} {h['aspect']:<24} sentiment t½: {hl}")


if __name__ == "__main__":
    main()
