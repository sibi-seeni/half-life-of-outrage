#!/bin/bash
# RoBERTa baseline on one B200 GPU. Needs data/comments_clean.csv in the repo folder and the
# environment from setup_env.sbatch. Submit from the repo folder:
#     sbatch scripts/hipergator/roberta.sbatch
#SBATCH --job-name=hlo-roberta
#SBATCH --account=eee6778
#SBATCH --qos=eee6778
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
WORK=${HLO_WORK:-/blue/eee6778/$USER/hlo}
PY="$WORK/env/bin/python"
[ -x "$PY" ] || { echo "Environment missing: run setup_env.sbatch first"; exit 1; }
export HF_HOME="$WORK/hf_cache"
export PYTHONPATH="$PWD/src"

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

test -f data/comments_clean.csv || { echo "data/comments_clean.csv not found"; exit 1; }
time "$PY" -m hlo.label.roberta_baseline --batch-size 512
ls -lh data/labels_roberta.csv
