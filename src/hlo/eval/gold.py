"""Gold-set construction and evaluation.

export:   writes a blind annotation sheet (no model labels) with two strata
          - random: uniform sample of the population (gives unbiased accuracy estimates)
          - stratified: aspect-balanced sample by LLM-predicted aspect (covers rare classes)
          plus an `overlap` flag marking rows the second annotator must also label.
evaluate: human-vs-human, human-vs-LLM and human-vs-RoBERTa agreement with bootstrap CIs,
          per-class reports, reported separately per stratum.

python -m hlo.eval.gold export --n-random 600 --n-strat 400 --n-overlap 300
python -m hlo.eval.gold evaluate --a1 annotation/annotator1.csv --a2 annotation/annotator2.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

from hlo.config import (DATASET_CSV, GOLD_TEMPLATE_CSV, LLM_LABELS_CSV, RANDOM_SEED,
                        RESULTS_DIR, ROBERTA_LABELS_CSV, ensure_dirs)
from hlo.eval.agreement import agreement_report

BLIND_COLS = ["comment_id", "video_title", "video_context", "parent_text", "text_preprocessed"]
ANNOT_COLS = ["gold_game_related", "gold_category", "gold_sentiment", "notes"]


def load_llm(path=LLM_LABELS_CSV):
    llm = pd.read_csv(path, dtype={"comment_id": str})
    return llm.drop_duplicates("comment_id", keep="last")  # retries append rows


def export(n_random, n_strat, n_overlap, seed=RANDOM_SEED):
    data = pd.read_csv(DATASET_CSV, dtype={"comment_id": str})
    llm = load_llm()
    rng = np.random.default_rng(seed)

    rand = data.sample(n=n_random, random_state=seed).assign(stratum="random")
    pool = data[~data["comment_id"].isin(rand["comment_id"])].merge(
        llm[["comment_id", "game_related", "category"]], on="comment_id")
    pool["aspect"] = np.where(pool["game_related"] == "No", "NotGame", pool["category"])
    groups = pool.groupby("aspect")
    per = n_strat // groups.ngroups
    strat = groups.apply(lambda g: g.sample(n=min(len(g), per), random_state=seed),
                         include_groups=False).reset_index(level=0, drop=True)
    strat = data[data["comment_id"].isin(strat["comment_id"])].assign(stratum="stratified")

    sheet = pd.concat([rand, strat]).sample(frac=1, random_state=seed)[BLIND_COLS + ["stratum"]]
    sheet["overlap"] = False
    sheet.loc[sheet.index[rng.permutation(len(sheet))[:n_overlap]], "overlap"] = True
    for c in ANNOT_COLS:
        sheet[c] = ""
    sheet.to_csv(GOLD_TEMPLATE_CSV, index=False)
    print(f"Wrote {len(sheet)} rows ({n_overlap} overlap) to {GOLD_TEMPLATE_CSV}. "
          "Give annotator 2 only rows with overlap == True; neither sees model labels.")


def _aspect(df, gr_col, cat_col):
    return np.where(df[gr_col].astype(str).str.capitalize() == "No", "NotGame", df[cat_col].astype(str))


def _per_class(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    out = {}
    for c in labels:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        p = tp / max((y_pred == c).sum(), 1)
        r = tp / max((y_true == c).sum(), 1)
        out[c] = {"support": int((y_true == c).sum()), "precision": round(p, 3), "recall": round(r, 3),
                  "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
    out["macro_f1"] = round(float(np.mean([v["f1"] for v in out.values()])), 3)
    return out


def evaluate(a1_path, a2_path=None):
    a1 = pd.read_csv(a1_path, dtype={"comment_id": str})
    a1["gold_aspect"] = _aspect(a1, "gold_game_related", "gold_category")
    llm = load_llm()
    llm["llm_aspect"] = _aspect(llm, "game_related", "category")
    df = a1.merge(llm[["comment_id", "llm_aspect", "sentiment"]], on="comment_id", how="left")
    report = {}

    if a2_path:  # human ceiling
        a2 = pd.read_csv(a2_path, dtype={"comment_id": str})
        a2["gold_aspect"] = _aspect(a2, "gold_game_related", "gold_category")
        hh = a1.merge(a2, on="comment_id", suffixes=("_1", "_2"))
        report["human_vs_human"] = {
            "aspect": agreement_report(hh["gold_aspect_1"].values, hh["gold_aspect_2"].values)}
        game = hh[(hh["gold_aspect_1"] != "NotGame") & (hh["gold_aspect_2"] != "NotGame")]
        report["human_vs_human"]["sentiment"] = agreement_report(
            game["gold_sentiment_1"].values, game["gold_sentiment_2"].values)

    for stratum, part in [("all", df)] + list(df.groupby("stratum")):
        part = part.dropna(subset=["llm_aspect"])
        r = {"aspect": agreement_report(part["gold_aspect"].values, part["llm_aspect"].values),
             "aspect_per_class": _per_class(part["gold_aspect"].values, part["llm_aspect"].values)}
        game = part[(part["gold_aspect"] != "NotGame") & (part["llm_aspect"] != "NotGame")]
        r["sentiment"] = agreement_report(game["gold_sentiment"].values, game["sentiment"].values)
        r["sentiment_per_class"] = _per_class(game["gold_sentiment"].values, game["sentiment"].values)
        report[f"human_vs_llm_{stratum}"] = r

    if ROBERTA_LABELS_CSV.exists():  # RoBERTa has 3 classes: fold Ambivalent into Neu for comparison
        rob = pd.read_csv(ROBERTA_LABELS_CSV, dtype={"comment_id": str})
        g = df[df["gold_aspect"] != "NotGame"].merge(rob, on="comment_id")
        gold3 = g["gold_sentiment"].replace({"Ambivalent": "Neu"}).values
        llm3 = g["sentiment"].replace({"Ambivalent": "Neu"}).values
        report["sentiment_3class"] = {
            "human_vs_roberta": agreement_report(gold3, g["roberta_label"].values),
            "human_vs_llm": agreement_report(gold3, llm3),
            "roberta_per_class": _per_class(gold3, g["roberta_label"].values),
        }

    ensure_dirs()
    (RESULTS_DIR / "gold_agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n-random", type=int, default=600)
    e.add_argument("--n-strat", type=int, default=400)
    e.add_argument("--n-overlap", type=int, default=300)
    v = sub.add_parser("evaluate")
    v.add_argument("--a1", required=True)
    v.add_argument("--a2")
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.n_random, args.n_strat, args.n_overlap)
    else:
        evaluate(args.a1, args.a2)


if __name__ == "__main__":
    main()
