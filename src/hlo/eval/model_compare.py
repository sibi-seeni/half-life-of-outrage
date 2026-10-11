"""Two-model comparison and disagreement-based selective relabeling.

evaluate  On the gold set: human vs Llama, human vs Gemma, Llama vs Gemma agreement (aspect,
          sentiment, joint label), plus a simulation of selective relabeling. A human only
          relabels comments the models disagree on; everything else keeps the shared model label.
          The resulting accuracy is compared with two baselines that send the *same number* of
          comments to a human: lowest Llama confidence first, and random.
queue     On the full dataset: writes a blind sheet of the comments where the two models
          disagree, i.e. the human relabeling workload.

python -m hlo.eval.model_compare evaluate --gold annotation/gold_adjudicated.csv
python -m hlo.eval.model_compare queue
"""
import argparse
import json

import numpy as np
import pandas as pd

from hlo.config import DATA_DIR, DATASET_CSV, LLM_LABELS_CSV, RANDOM_SEED, RESULTS_DIR, ensure_dirs
from hlo.eval.agreement import agreement_report
from hlo.eval.gold import BLIND_COLS

SECOND_LABELS_CSV = DATA_DIR / "labels_gemma.csv"


def joint_label(game_related, category, sentiment):
    """One label per comment: 'NotGame', or '<aspect>|<sentiment>' for game comments."""
    gr = pd.Series(game_related).astype(str).str.capitalize().values
    return np.where(gr == "No", "NotGame",
                    pd.Series(category).astype(str).values + "|" + pd.Series(sentiment).astype(str).values)


def _load_model(path, prefix):
    df = pd.read_csv(path, dtype={"comment_id": str}).drop_duplicates("comment_id", keep="last")
    df = df[df["parse_ok"].astype(str) == "True"]
    out = pd.DataFrame({"comment_id": df["comment_id"],
                        f"{prefix}_joint": joint_label(df["game_related"], df["category"], df["sentiment"]),
                        f"{prefix}_conf": pd.to_numeric(df["confidence"], errors="coerce")})
    out[f"{prefix}_aspect"] = out[f"{prefix}_joint"].str.split("|").str[0]
    out[f"{prefix}_sent"] = out[f"{prefix}_joint"].str.split("|").str[1]
    return out


def _pair(df, a, b):
    rep = {"joint": agreement_report(df[f"{a}_joint"].values, df[f"{b}_joint"].values),
           "aspect": agreement_report(df[f"{a}_aspect"].values, df[f"{b}_aspect"].values)}
    game = df[df[f"{a}_sent"].notna() & df[f"{b}_sent"].notna()]
    rep["sentiment_on_game"] = agreement_report(game[f"{a}_sent"].values, game[f"{b}_sent"].values)
    return rep


def selective_relabeling(df, seed=RANDOM_SEED):
    """Accuracy (vs gold joint label) when humans relabel a subset and models label the rest."""
    gold, llama, gemma = df["gold_joint"].values, df["llama_joint"].values, df["gemma_joint"].values
    n = len(df)
    disagree = llama != gemma
    k = int(disagree.sum())

    def acc_with(routed):
        final = np.where(routed, gold, llama)
        return float((final == gold).mean())

    conf = df["llama_conf"].fillna(df["llama_conf"].min()).values
    by_conf = np.zeros(n, bool)
    by_conf[np.argsort(conf, kind="stable")[:k]] = True
    rng = np.random.default_rng(seed)
    rand = np.mean([acc_with(np.isin(np.arange(n), rng.choice(n, k, replace=False))) for _ in range(200)])

    agree = ~disagree
    return {
        "n": n,
        "routed_to_human": k,
        "routed_share": k / n,
        "accuracy_llama_only": float((llama == gold).mean()),
        "accuracy_gemma_only": float((gemma == gold).mean()),
        "accuracy_when_models_agree": float((llama[agree] == gold[agree]).mean()) if agree.any() else None,
        "accuracy_llama_when_models_disagree": float((llama[disagree] == gold[disagree]).mean()) if k else None,
        "accuracy_gemma_when_models_disagree": float((gemma[disagree] == gold[disagree]).mean()) if k else None,
        "accuracy_route_disagreements": acc_with(disagree),
        "accuracy_route_lowest_llama_confidence": acc_with(by_conf),
        "accuracy_route_random_mean": float(rand),
        "errors_caught_by_disagreement": float(((llama != gold) & disagree).sum() / max((llama != gold).sum(), 1)),
    }


def evaluate(gold_path, llama_path=LLM_LABELS_CSV, gemma_path=SECOND_LABELS_CSV):
    g = pd.read_csv(gold_path, dtype={"comment_id": str})
    g = g.assign(gold_joint=joint_label(g["gold_game_related"], g["gold_category"], g["gold_sentiment"]))
    g["gold_aspect"] = g["gold_joint"].str.split("|").str[0]
    g["gold_sent"] = g["gold_joint"].str.split("|").str[1]
    df = (g[["comment_id", "stratum", "gold_joint", "gold_aspect", "gold_sent"]]
          .merge(_load_model(llama_path, "llama"), on="comment_id")
          .merge(_load_model(gemma_path, "gemma"), on="comment_id"))

    report = {}
    for stratum, part in [("all", df)] + list(df.groupby("stratum")):
        report[stratum] = {"human_vs_llama": _pair(part, "gold", "llama"),
                           "human_vs_gemma": _pair(part, "gold", "gemma"),
                           "llama_vs_gemma": _pair(part, "llama", "gemma")}
    # The random stratum is the unbiased estimate of what routing would cost on the population.
    report["selective_relabeling"] = {s: selective_relabeling(p)
                                      for s, p in [("all", df)] + list(df.groupby("stratum"))}
    ensure_dirs()
    (RESULTS_DIR / "model_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["selective_relabeling"], indent=2))


def queue(llama_path=LLM_LABELS_CSV, gemma_path=SECOND_LABELS_CSV, out=DATA_DIR / "relabel_queue.csv"):
    both = _load_model(llama_path, "llama").merge(_load_model(gemma_path, "gemma"), on="comment_id")
    dis = both[both["llama_joint"] != both["gemma_joint"]]
    data = pd.read_csv(DATASET_CSV, dtype={"comment_id": str})
    sheet = data[data["comment_id"].isin(dis["comment_id"])][BLIND_COLS].copy()
    for c in ("gold_game_related", "gold_category", "gold_sentiment", "notes"):
        sheet[c] = ""
    sheet.to_csv(out, index=False)
    summary = {"compared": len(both), "disagreements": len(dis), "share": len(dis) / max(len(both), 1),
               "aspect_disagreements": int((both["llama_aspect"] != both["gemma_aspect"]).sum())}
    ensure_dirs()
    (RESULTS_DIR / "relabel_queue_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), f"\nBlind sheet -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--gold", required=True, help="adjudicated gold sheet from hlo.eval.gold export")
    e.add_argument("--llama", default=str(LLM_LABELS_CSV))
    e.add_argument("--gemma", default=str(SECOND_LABELS_CSV))
    q = sub.add_parser("queue")
    q.add_argument("--llama", default=str(LLM_LABELS_CSV))
    q.add_argument("--gemma", default=str(SECOND_LABELS_CSV))
    args = ap.parse_args()
    if args.cmd == "evaluate":
        evaluate(args.gold, args.llama, args.gemma)
    else:
        queue(args.llama, args.gemma)


if __name__ == "__main__":
    main()
