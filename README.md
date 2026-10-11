# The Half-Life of Outrage

Code for *The Half-Life of Outrage: Modeling Temporal Sentiment Decay and Community Backlash in Video Game Announcements*. The study is a longitudinal, aspect-level sentiment study of YouTube discourse about **Arc Raiders** (Embark Studios), from its reveal in December 2021 through launch and the 2025–26 AI-voice controversy.

The pipeline:

1. **Collect** every comment and reply on Arc Raiders videos in the YouTube Gaming category via the YouTube Data API v3. Commenter names are SHA-256 hashed at ingestion.
2. **Build** a clean English dataset: language detection, ≥3 words, deduplication, URL/mention masking, volume-spike flags, and the video and parent-comment context.
3. **Label** each comment with Llama-3.3-70B. It first decides whether the comment is about the game, the creator, or something else. Game comments then get an aspect (gameplay / business model / AI voices), a sentiment and a confidence score. A RoBERTa model provides the baseline.
4. **Validate** against a two-annotator gold set: human–human ceiling, human–LLM and human–RoBERTa agreement (percent agreement, Cohen's κ, Gwet's AC1, Krippendorff's α, bootstrap CIs).
5. **Analyse** each event with a verified date:
   - **Sentiment half-life**: exponential decay of the excess negative share above the pre-event baseline, with a bootstrap CI and a model-free check.
   - **Attention half-life**: decay of comment volume.
   - **Interrupted time series**: level and slope change, with Newey-West errors.
   - **Pre/event/post windows**: raw and engagement-weighted.

## Repository layout

```
config/events.csv          event dates with verification status and sources (the analysis anchors)
src/hlo/
  collect/                 YouTube discovery + comment/reply collection into SQLite
  data/build_dataset.py    DB -> data/comments_clean.csv + results/data_flow.json
  label/
    llm_labeler.py         Llama-3.3-70B labeling (resumable, key rotation, cost tracking)
    prompts/absa_v2.txt    prompt used for the final labels (absa_v1 = earlier pilot)
    roberta_baseline.py    cardiffnlp/twitter-roberta-base-sentiment-latest baseline
  eval/
    agreement.py           κ, AC1, Krippendorff's α, bootstrap CIs
    gold.py                gold-set export (random + stratified, blind) and evaluation
    model_compare.py       Llama vs Gemma vs humans; disagreement-based selective relabeling
  analysis/
    panel.py               join labels, daily aspect series
    decay.py               sentiment/attention half-life
    its.py                 interrupted time series
    windows.py             event windows, engagement-weighted gap
    run.py                 runs everything, writes results/*.json and figures/
scripts/
  run_all.sh               end-to-end run + robustness grid
  export_release.py        public release: IDs + labels, no text
  rehydrate.py             rebuild comment text from released IDs
annotation/GUIDELINES.md   instructions given to human annotators
tests/                     unit tests + an end-to-end run on synthetic data
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env    # add YOUTUBE_API_KEY and NAVIGATOR_API_KEYS
pytest                  # no data or API keys needed
```

## Reproducing the paper

```bash
python -m hlo.collect full                 # YouTube API (quota-limited; rerun daily until complete)
python -m hlo.data.build_dataset           # -> data/comments_clean.csv, results/data_flow.json
python -m hlo.label.llm_labeler            # -> data/labels_llm.csv   (~$0.27 per 1k comments on NaviGator)
python -m hlo.label.roberta_baseline       # -> data/labels_roberta.csv

python -m hlo.eval.gold export             # -> data/gold_to_annotate.csv for both annotators
python -m hlo.eval.gold evaluate --a1 annotation/annotator1.csv --a2 annotation/annotator2.csv

./scripts/run_all.sh                       # main analysis + robustness grid -> results/, figures/
```

Every number in the paper is read from a file in `results/`. Nothing is copied by hand.

### Second model and selective relabeling

Gemma-4-31B labels the same comments with the same prompt (`absa_v2`). It's used to check the main labels, not to replace them:

```bash
# gold set only (a few dollars): Llama vs Gemma vs humans, and the routing simulation
cut -d, -f1 annotation/gold_adjudicated.csv | tail -n +2 > gold_ids.txt
python -m hlo.label.llm_labeler --model gemma-4-31b-it --ids gold_ids.txt --out data/labels_gemma.csv
python -m hlo.eval.model_compare evaluate --gold annotation/gold_adjudicated.csv   # -> results/model_comparison.json

# optional, full dataset: blind sheet of every comment the two models disagree on
python -m hlo.label.llm_labeler --model gemma-4-31b-it --out data/labels_gemma.csv
python -m hlo.eval.model_compare queue                                            # -> data/relabel_queue.csv
```

The routing simulation sends a human only the comments where Llama and Gemma disagree. It compares the resulting accuracy with sending the same number of comments chosen by lowest Llama confidence, or at random. Pricing per model is set in `MODEL_PRICES` in `llm_labeler.py`: Llama $0.40/$0.40 and Gemma $0.12/$0.36 per million input/output tokens. Other models need `--price-in`/`--price-out`.

Context ablation (used for the labeling-reliability analysis): `python -m hlo.label.llm_labeler --ids gold_ids.txt --no-context --out data/labels_llm_nocontext.csv`.

## Data availability

YouTube's API Terms of Service do not allow redistributing comment text. The release (`scripts/export_release.py`) contains comment, video and channel IDs, timestamps, like counts, re-salted author IDs, and all model labels. `scripts/rehydrate.py` re-fetches the text through the API. Comments deleted since collection can't be recovered, so report the recovery rate when you reuse the data.

## Ethics

The data is public comments only. Author names are never stored, only salted hashes. No individual-level analysis is done, and quoted examples in the paper are paraphrased.

## License

MIT. See `LICENSE`.
