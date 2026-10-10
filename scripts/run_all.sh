#!/usr/bin/env bash
# End-to-end pipeline. Steps that call paid/quota APIs are opt-in via flags.
#   ./scripts/run_all.sh                  analysis only (needs data/comments_clean.csv + data/labels_llm.csv)
#   ./scripts/run_all.sh --collect --label   full rebuild from the YouTube API
set -euo pipefail
cd "$(dirname "$0")/.."

for arg in "$@"; do
  case $arg in
    --collect) python -m hlo.collect full ;;
    --label)   python -m hlo.data.build_dataset
               python -m hlo.label.llm_labeler
               python -m hlo.label.roberta_baseline ;;
  esac
done

python -m hlo.analysis.run
# Robustness grid reported in the appendix
python -m hlo.analysis.run --min-conf 0.8 --tag conf80
python -m hlo.analysis.run --horizon 30 --tag h30
python -m hlo.analysis.run --baseline 14 --tag base14
