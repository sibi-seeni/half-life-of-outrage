"""End-to-end run of the analysis on a synthetic labeled dataset."""
import importlib
import json

import numpy as np
import pandas as pd


def test_analysis_runs_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-06-01", "2024-11-30", freq="D")
    event = pd.Timestamp("2024-08-20")
    rows = []
    for d in days:
        for _ in range(60):
            aspect = rng.choice(["Game_Related", "Business_Model_Related", "AI_Voice_Related", "NotGame"],
                                p=[0.6, 0.2, 0.05, 0.15])
            t = (d - event).days
            p_neg = 0.3 + (0.4 * np.exp(-t * np.log(2) / 6) if (t >= 0 and aspect == "Business_Model_Related") else 0)
            rows.append({"day": d, "aspect": aspect, "neg": rng.random() < p_neg})
    syn = pd.DataFrame(rows)
    syn["comment_id"] = [f"c{i}" for i in range(len(syn))]
    data = pd.DataFrame({
        "comment_id": syn["comment_id"], "parent_id": None, "parent_text": "",
        "text_preprocessed": "x", "video_id": rng.choice(["v1", "v2", "v3"], len(syn)),
        "video_title": "t", "video_context": "", "channel_id": "ch",
        "comment_date": syn["day"] + pd.to_timedelta(rng.integers(0, 86000, len(syn)), unit="s"),
        "video_date": pd.Timestamp("2024-05-01"), "author_hash": "a",
        "likes": rng.integers(0, 50, len(syn)), "like_count_log": 0.0,
        "comment_latency_days": 1.0, "is_spike": False, "z_score": 0.0})
    labels = pd.DataFrame({
        "comment_id": syn["comment_id"],
        "game_related": np.where(syn["aspect"] == "NotGame", "No", "Yes"),
        "target": np.where(syn["aspect"] == "NotGame", "Creator", "Game"),
        "category": np.where(syn["aspect"] == "NotGame", "", syn["aspect"]),
        "sentiment": np.where(syn["aspect"] == "NotGame", "", np.where(syn["neg"], "Neg", "Pos")),
        "confidence": np.where(syn["aspect"] == "NotGame", np.nan, 0.85),
        "parse_ok": True})
    events = pd.DataFrame([{"event_id": "premium_switch", "date": "2024-08-20", "label": "Premium switch",
                            "aspect": "Business_Model_Related", "status": "verified", "source": "", "note": ""}])

    monkeypatch.setenv("HLO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HLO_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("HLO_FIGURES_DIR", str(tmp_path / "figures"))
    import hlo.config as config
    importlib.reload(config)
    config.ensure_dirs()
    data.to_csv(config.DATASET_CSV, index=False)
    labels.to_csv(config.LLM_LABELS_CSV, index=False)
    events_path = tmp_path / "events.csv"
    events.to_csv(events_path, index=False)

    import hlo.analysis.panel as panel
    import hlo.analysis.run as run
    importlib.reload(panel)
    monkeypatch.setattr(panel, "EVENTS_CSV", events_path)
    importlib.reload(run)
    monkeypatch.setattr(run, "load_events", lambda include: panel.load_events(events_path, include))
    monkeypatch.setattr("sys.argv", ["run"])
    run.main()

    half = json.loads((config.RESULTS_DIR / "half_life.json").read_text())
    biz = next(h for h in half if h["aspect"] == "Business_Model_Related")["sentiment"]
    assert biz["status"] == "ok"
    assert 3 < biz["half_life"] < 10  # true value 6 days
    game = next(h for h in half if h["aspect"] == "Game_Related")["sentiment"]
    assert game["status"] != "ok" or game["peak_excess"] < 0.1  # no injected effect on gameplay
    summary = json.loads((config.RESULTS_DIR / "label_summary.json").read_text())
    assert summary["confidence_retention"][">=0.8"] == 1.0
    assert (config.FIGURES_DIR / "timeline.png").exists()
