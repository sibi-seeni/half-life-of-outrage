import pandas as pd

from hlo.eval import gold


def test_failed_labels_are_not_a_stratum(tmp_path, monkeypatch):
    aspects = ["Game_Related"] * 20 + ["AI_Voice_Related"] * 20 + ["PARSE_ERROR"]
    data = pd.DataFrame({"comment_id": [f"c{i}" for i in range(len(aspects))], "text_preprocessed": "x",
                         "video_title": "", "video_context": "", "parent_text": ""})
    llm = data[["comment_id"]].assign(game_related=["Yes"] * 40 + [""], category=aspects,
                                      parse_ok=["True"] * 40 + ["False"])
    data.to_csv(tmp_path / "data.csv", index=False)
    monkeypatch.setattr(gold, "DATASET_CSV", tmp_path / "data.csv")
    monkeypatch.setattr(gold, "GOLD_TEMPLATE_CSV", tmp_path / "gold.csv")
    monkeypatch.setattr(gold, "load_llm", lambda: llm)

    gold.export(n_random=0, n_strat=20, n_overlap=0)
    sheet = pd.read_csv(tmp_path / "gold.csv")
    assert "c40" not in set(sheet["comment_id"])
    assert len(sheet) == 20  # 10 per real aspect, not 20 // 3
