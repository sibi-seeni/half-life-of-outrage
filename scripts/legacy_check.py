"""Can the class-work labels (prompt_sha256="legacy") be reused? Relabel a random sample and compare.

  python scripts/legacy_check.py sample            # -> data/legacy_check_ids.txt (500 legacy IDs in the dataset)
  python -m hlo.label.llm_labeler --workers 4 --ids data/legacy_check_ids.txt --out data/labels_legacy_check.csv
  python scripts/legacy_check.py compare           # -> results/legacy_check.json

Agreement is reported for the target (Game/Creator/Other), the full label the analysis uses
(target, or category+sentiment for game comments), and category and sentiment on comments
both runs call Game. Temperature is 0, so a matching pipeline should agree almost perfectly;
two Llama runs with the same prompt and inputs typically give kappa well above 0.8.
"""
import argparse
import json

import pandas as pd

from hlo.config import DATA_DIR, DATASET_CSV, LLM_LABELS_CSV, RANDOM_SEED, RESULTS_DIR, ensure_dirs
from hlo.eval.agreement import agreement_report

IDS = DATA_DIR / "legacy_check_ids.txt"
RELABEL = DATA_DIR / "labels_legacy_check.csv"
OUT = RESULTS_DIR / "legacy_check.json"


def _read(path):
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    d = d.drop_duplicates("comment_id", keep="last")
    return d[d.parse_ok == "True"]


def _full(d):
    return d.target.where(d.target != "Game", "Game|" + d.category + "|" + d.sentiment)


def sample(n):
    labels = _read(LLM_LABELS_CSV)
    ids = pd.read_csv(DATASET_CSV, dtype={"comment_id": str}, usecols=["comment_id"]).comment_id
    legacy = labels[(labels.prompt_sha256 == "legacy") & labels.comment_id.isin(ids)]
    picked = legacy.comment_id.sample(n, random_state=RANDOM_SEED)
    IDS.write_text("\n".join(picked) + "\n")
    print(f"wrote {len(picked)} of {len(legacy)} legacy IDs to {IDS}")


def compare():
    ensure_dirs()
    if not RELABEL.exists():
        raise SystemExit(f"{RELABEL} not found. Relabel the sample first:\n  python -m hlo.label.llm_labeler "
                         f"--workers 4 --ids {IDS.relative_to(DATA_DIR.parent)} --out {RELABEL.relative_to(DATA_DIR.parent)}")
    old = _read(LLM_LABELS_CSV)
    new = _read(RELABEL)
    m = old.merge(new, on="comment_id", suffixes=("_old", "_new"))
    m = m[m.prompt_sha256_old == "legacy"]
    o = m.filter(like="_old").rename(columns=lambda c: c[:-4])
    n = m.filter(like="_new").rename(columns=lambda c: c[:-4])
    game = (o.target == "Game") & (n.target == "Game")
    report = {
        "n": int(len(m)),
        "target": agreement_report(o.target.values, n.target.values),
        "full_label": agreement_report(_full(o).values, _full(n).values),
        "category_on_game": agreement_report(o.category[game].values, n.category[game].values),
        "sentiment_on_game": agreement_report(o.sentiment[game].values, n.sentiment[game].values),
        "share_neg_of_game": {"old": round(float((o.sentiment[o.target == "Game"] == "Neg").mean()), 4),
                              "new": round(float((n.sentiment[n.target == "Game"] == "Neg").mean()), 4)},
        "share_business": {"old": round(float((o.category == "Business_Model_Related").mean()), 4),
                           "new": round(float((n.category == "Business_Model_Related").mean()), 4)},
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\ntarget confusion (rows = old, cols = new):")
    print(pd.crosstab(o.target, n.target))
    print("\ncategory confusion on game comments:")
    print(pd.crosstab(o.category[game], n.category[game]))
    print("\nsentiment confusion on game comments:")
    print(pd.crosstab(o.sentiment[game], n.sentiment[game]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=500)
    sub.add_parser("compare")
    args = ap.parse_args()
    sample(args.n) if args.cmd == "sample" else compare()


if __name__ == "__main__":
    main()
