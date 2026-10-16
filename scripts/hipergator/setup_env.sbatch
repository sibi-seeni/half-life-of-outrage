#!/bin/bash
# One-time setup: builds a Python environment for the RoBERTa job and pre-downloads the model.
# Runs on a CPU node (no GPU needed to install). Submit from the repo folder:
#     sbatch scripts/hipergator/setup_env.sbatch
#SBATCH --job-name=hlo-setup
#SBATCH --account=eee6778
#SBATCH --qos=eee6778
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
WORK=${HLO_WORK:-/blue/eee6778/$USER/hlo}
mkdir -p "$WORK"
echo "Environment and model cache go in: $WORK"

module load conda
[ -x "$WORK/env/bin/python" ] || conda create -y -p "$WORK/env" python=3.11
PY="$WORK/env/bin/python"   # call the env's python directly; `conda activate` is unreliable in batch jobs

# B200 (Blackwell) GPUs need a PyTorch build for CUDA 12.8 or newer.
"$PY" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-cache-dir "transformers>=4.45" pandas numpy python-dotenv

export HF_HOME="$WORK/hf_cache"
"$PY" - <<'EOS'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = "cardiffnlp/twitter-roberta-base-sentiment-latest"
AutoTokenizer.from_pretrained(m)
AutoModelForSequenceClassification.from_pretrained(m)
print("model cached")
EOS
"$PY" -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
echo "Setup finished."
