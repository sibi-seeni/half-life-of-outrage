"""Zero-shot RoBERTa sentiment baseline (cardiffnlp/twitter-roberta-base-sentiment-latest).

Writes comment_id + class probabilities + label + max-prob confidence. Ported from
roberta_classification.py; input text is the same `text_preprocessed` field the LLM sees.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hlo.config import DATASET_CSV, ROBERTA_LABELS_CSV, ensure_dirs

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = np.array(["Neg", "Neu", "Pos"])


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def classify(texts, model, tok, dev, batch_size=128, max_length=128):
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding="longest", truncation=True,
                  max_length=max_length, return_tensors="pt").to(dev)
        probs.append(torch.softmax(model(**enc).logits, dim=1).cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATASET_CSV))
    ap.add_argument("--out", default=str(ROBERTA_LABELS_CSV))
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    ensure_dirs()

    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval().to(dev)
    df = pd.read_csv(args.input, usecols=["comment_id", "text_preprocessed"], dtype={"comment_id": str})
    p = classify(df["text_preprocessed"].fillna("").astype(str).tolist(), model, tok, dev, args.batch_size)
    out = pd.DataFrame({"comment_id": df["comment_id"], "roberta_neg": p[:, 0], "roberta_neu": p[:, 1],
                        "roberta_pos": p[:, 2], "roberta_label": LABELS[p.argmax(1)],
                        "roberta_confidence": p.max(1)})
    out.to_csv(args.out, index=False)
    print(out["roberta_label"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
