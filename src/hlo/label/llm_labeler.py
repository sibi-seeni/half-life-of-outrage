"""Aspect + sentiment labeling with an OpenAI-compatible chat endpoint (UF NaviGator, Llama-3.3-70B).

Ported from Llama_Labeler.ipynb. Changes from the original notebook:
  * the parent comment is read from `parent_text` (the notebook read a non-existent
    `parent_context` column, so replies were labeled without their parent);
  * the whole dataset is labeled by default (the notebook capped at LIMIT rows);
  * `--no-context` drops title/description/parent for the context ablation;
  * the prompt lives in prompts/<version>.txt and its SHA-256 is stored with every label.

Prompt versions:
  absa_v1  4 categories incl. Others, one-step (early 20k-row pilot).
  absa_v2  two-step: target gate (Game / Creator / Other) then category+sentiment for Game
           comments only. This produced the final labels; non-game rows get
           game_related="No" and an empty category, which is why ~1/3 of rows have no aspect.

Usage: python -m hlo.label.llm_labeler [--model M] [--limit N] [--ids FILE] [--no-context] [--out PATH]
       Second model (gold set or full data, always into its own file):
       python -m hlo.label.llm_labeler --model gemma-4-31b-it --out data/labels_gemma.csv
Env:   NAVIGATOR_API_KEYS=key1,key2   NAVIGATOR_BASE_URL (default https://api.ai.it.ufl.edu)
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from hlo.config import CATEGORIES, DATASET_CSV, LLM_LABELS_CSV, SENTIMENTS, ensure_dirs

load_dotenv()
log = logging.getLogger("absa")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROMPT_DIR = Path(__file__).parent / "prompts"
BASE_URL = os.getenv("NAVIGATOR_BASE_URL", "https://api.ai.it.ufl.edu")
# USD per million (input, output) tokens on NaviGator; override with --price-in/--price-out
MODEL_PRICES = {
    "llama-3.3-70b-instruct": (0.40, 0.40),
    "gemma-4-31b-it": (0.12, 0.36),
}
BUDGET_PER_KEY = 24.50
MAX_RETRIES = 3
TIMEOUT = 45

OUT_FIELDS = ["comment_id", "game_related", "target", "category", "sentiment", "confidence",
              "reasoning", "parse_ok",
              "model", "prompt_version", "prompt_sha256", "context", "tokens_in", "tokens_out"]


class KeyRotator:
    """Round-robin over API keys; retired keys (budget/auth) leave the pool."""

    def __init__(self, keys):
        if not keys:
            raise ValueError("Set NAVIGATOR_API_KEYS (comma-separated).")
        self.keys, self.active, self.idx = list(keys), list(range(len(keys))), 0

    def current(self):
        if not self.active:
            raise RuntimeError("All API keys retired.")
        return self.keys[self.active[self.idx % len(self.active)]]

    def advance(self):
        if self.active:
            self.idx = (self.idx + 1) % len(self.active)

    def retire(self, key):
        ki = self.keys.index(key)
        if ki in self.active:
            self.active.remove(ki)
            log.warning("Key …%s retired, %d left", key[-4:], len(self.active))


class CostTracker:
    def __init__(self, keys, price_in, price_out):
        self.cost = {k: 0.0 for k in keys}
        self.price_in, self.price_out = price_in, price_out

    def record(self, key, t_in, t_out):
        self.cost[key] += t_in / 1e6 * self.price_in + t_out / 1e6 * self.price_out
        return self.cost[key]

    def total(self):
        return sum(self.cost.values())


def build_user_message(row, use_context=True):
    def _s(v, n=None):
        v = "" if pd.isna(v) else str(v).strip()
        return v[:n] if n else v

    parts = []
    if use_context:
        parts += [f"video_title: {_s(row.get('video_title'), 300)}",
                  f"video_context: {_s(row.get('video_context'), 400)}"]
        parent = _s(row.get("parent_text"), 300)
        if parent:
            parts.append(f"parent_context: {parent}")
    parts.append(f"comment: {_s(row.get('text_preprocessed'), 1000)}")
    return "\n".join(parts)


def call_api(key, system_prompt, user_message, model, json_mode=False, max_tokens=512):
    payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _strip_fences(raw):
    # Reasoning models (e.g. Gemma 4 with thinking on) may emit text or <think> blocks before the JSON.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in raw and not raw.startswith("```"):
        raw = raw[raw.index("```"):]
    elif not raw.startswith(("{", "```")) and "{" in raw:
        raw = raw[raw.index("{"):]
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _repair_newlines(raw):
    def _fix(m):
        return m.group(1) + m.group(2).replace("\n", " ").replace("\r", "") + m.group(3)
    return re.sub(r'("(?:game_related|target|category|sentiment|confidence|reasoning)"\s*:\s*")(.*?)("\s*[,}])',
                  _fix, raw, flags=re.DOTALL)


def _regex_fields(raw):
    fields = ("game_related", "target", "category", "sentiment", "confidence")
    found = {f: re.search(rf'"{f}"\s*:\s*"?([^",}}]+)', raw) for f in fields}
    if not ((found["category"] and found["sentiment"]) or found["game_related"]):
        return None
    reas = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*[,}]', raw, re.DOTALL)
    return {k: (m.group(1).strip() if m else "") for k, m in found.items()} | {
        "reasoning": reas.group(1) if reas else ""}


def parse_response(api_response):
    """strict JSON -> JSON after newline repair -> regex extraction. Never raises."""
    try:
        raw = _strip_fences(api_response["choices"][0]["message"]["content"].strip())
    except (KeyError, IndexError, AttributeError) as exc:
        return {"category": "PARSE_ERROR", "sentiment": "PARSE_ERROR", "confidence": "",
                "reasoning": f"bad response: {exc}", "parse_ok": False}
    label = None
    for attempt in (lambda: json.loads(raw), lambda: json.loads(_repair_newlines(raw)), lambda: _regex_fields(raw)):
        try:
            label = attempt()
        except json.JSONDecodeError:
            label = None
        if label:
            break
    if not label:
        return {"category": "PARSE_ERROR", "sentiment": "PARSE_ERROR", "confidence": "",
                "reasoning": raw[:300], "parse_ok": False}
    return normalise(label)


def normalise(label):
    """Map a parsed label onto the output schema (handles both v1 and v2 prompts)."""
    game_related = str(label.get("game_related", "Yes")).strip().capitalize()
    if game_related == "No":
        return {"game_related": "No", "target": str(label.get("target", "Other")).strip(),
                "category": "", "sentiment": "", "confidence": "", "reasoning": "", "parse_ok": True}
    cat = str(label.get("category", "")).strip()
    sent = str(label.get("sentiment", "")).strip()
    try:
        conf = float(label.get("confidence", ""))
    except (TypeError, ValueError):
        conf = ""
    return {"game_related": "Yes", "target": "Game",
            "category": cat if cat in CATEGORIES else "Others",
            "sentiment": sent if sent in SENTIMENTS else "Neu",
            "confidence": conf,
            "reasoning": str(label.get("reasoning", "")).replace("\n", " ").strip(),
            "parse_ok": True}


def label_one(row, rotator, tracker, system_prompt, model, use_context, json_mode=False, max_tokens=512):
    msg = build_user_message(row, use_context)
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        key = rotator.current()
        try:
            resp = call_api(key, system_prompt, msg, model, json_mode, max_tokens)
            usage = resp.get("usage", {})
            t_in, t_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            if tracker.record(key, t_in, t_out) >= BUDGET_PER_KEY:
                rotator.retire(key)
            else:
                rotator.advance()
            return parse_response(resp) | {"tokens_in": t_in, "tokens_out": t_out}
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status in (401, 403):
                rotator.retire(key)
            else:
                rotator.advance()
                time.sleep(attempt + random.random())
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            time.sleep(2 ** attempt + random.random())
    return {"category": "ERROR", "sentiment": "ERROR", "confidence": "",
            "reasoning": f"API failure: {last_exc}", "parse_ok": False, "tokens_in": 0, "tokens_out": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATASET_CSV))
    ap.add_argument("--out", default=str(LLM_LABELS_CSV))
    ap.add_argument("--model", default="llama-3.3-70b-instruct")
    ap.add_argument("--prompt", default="absa_v2")
    ap.add_argument("--json-mode", action="store_true", help="send response_format=json_object (v1 runs used it)")
    ap.add_argument("--no-context", action="store_true", help="comment text only (ablation)")
    ap.add_argument("--ids", help="optional file with one comment_id per line to label (e.g. the gold set)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--price-in", type=float, help="USD per 1M input tokens (default: MODEL_PRICES)")
    ap.add_argument("--price-out", type=float, help="USD per 1M output tokens (default: MODEL_PRICES)")
    args = ap.parse_args()
    if args.model not in MODEL_PRICES and (args.price_in is None or args.price_out is None):
        ap.error(f"no price known for {args.model}; pass --price-in and --price-out")
    price_in, price_out = MODEL_PRICES.get(args.model, (None, None))
    price_in = args.price_in if args.price_in is not None else price_in
    price_out = args.price_out if args.price_out is not None else price_out
    ensure_dirs()

    system_prompt = (PROMPT_DIR / f"{args.prompt}.txt").read_text()
    prompt_sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
    keys = [k.strip() for k in os.getenv("NAVIGATOR_API_KEYS", "").split(",") if k.strip()]
    rotator, tracker = KeyRotator(keys), CostTracker(keys, price_in, price_out)

    df = pd.read_csv(args.input, dtype={"comment_id": str})
    if args.ids:
        wanted = set(Path(args.ids).read_text().split())
        df = df[df["comment_id"].isin(wanted)]
    out = Path(args.out)
    done = set()
    if out.exists():  # resume, retrying rows that failed last time
        prev = pd.read_csv(out, dtype={"comment_id": str})
        done = set(prev.loc[prev["parse_ok"].astype(str) == "True", "comment_id"])
    todo = df[~df["comment_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    log.info("%d rows, %d already labeled, %d to do", len(df), len(done), len(todo))

    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for i, row in enumerate(todo.to_dict("records"), 1):
            res = label_one(row, rotator, tracker, system_prompt, args.model,
                            not args.no_context, args.json_mode, args.max_tokens)
            writer.writerow(res | {"comment_id": row["comment_id"], "model": args.model,
                                   "prompt_version": args.prompt, "prompt_sha256": prompt_sha,
                                   "context": not args.no_context})
            if i % 200 == 0:
                fh.flush()
                log.info("%d/%d labeled, cost so far $%.2f", i, len(todo), tracker.total())
    log.info("Done. Total cost $%.2f. Keep the last row per comment_id when reading (retries append).",
             tracker.total())


if __name__ == "__main__":
    main()
