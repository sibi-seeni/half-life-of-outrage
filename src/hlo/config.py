"""Shared paths and constants. Override paths with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.getenv("HLO_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.getenv("HLO_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.getenv("HLO_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = Path(os.getenv("HLO_FIGURES_DIR", ROOT / "figures"))
CONFIG_DIR = ROOT / "config"
DB_URL = os.getenv("HLO_DB_URL", f"sqlite:///{DATA_DIR / 'arc_raiders.db'}")

EVENTS_CSV = CONFIG_DIR / "events.csv"
DATASET_CSV = DATA_DIR / "comments_clean.csv"
LLM_LABELS_CSV = DATA_DIR / "labels_llm.csv"
ROBERTA_LABELS_CSV = DATA_DIR / "labels_roberta.csv"
GOLD_TEMPLATE_CSV = DATA_DIR / "gold_to_annotate.csv"

GAME_LAUNCH = "2025-10-30"
# Study window end (inclusive, UTC). Videos published and comments posted after this are not
# collected and are dropped when the dataset is built. Override with HLO_END_DATE=YYYY-MM-DD.
END_DATE = os.getenv("HLO_END_DATE", "2026-04-30")
CATEGORIES = ["AI_Voice_Related", "Business_Model_Related", "Game_Related", "Others"]
SENTIMENTS = ["Pos", "Neg", "Neu", "Ambivalent"]
SENTIMENT_SCORE = {"Pos": 1, "Neu": 0, "Ambivalent": 0, "Neg": -1}
RANDOM_SEED = 15


def end_datetime():
    """END_DATE as a naive UTC datetime at 23:59:59 (inclusive end of day)."""
    from datetime import datetime
    return datetime.strptime(END_DATE, "%Y-%m-%d").replace(hour=23, minute=59, second=59)


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
