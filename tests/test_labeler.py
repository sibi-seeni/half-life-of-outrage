import pandas as pd

from hlo.data.build_dataset import preprocess_text, spike_flags
from hlo.label.llm_labeler import build_user_message, parse_response


def _resp(content):
    return {"choices": [{"message": {"content": content}}]}


def test_parse_v2_game_comment():
    r = parse_response(_resp('{"game_related":"Yes","category":"AI_Voice_Related","sentiment":"Neg",'
                             '"confidence":"0.85","reasoning":"hates the TTS"}'))
    assert r == r | {"game_related": "Yes", "category": "AI_Voice_Related", "sentiment": "Neg",
                     "confidence": 0.85, "parse_ok": True}


def test_parse_v2_non_game_comment():
    r = parse_response(_resp('```json\n{"game_related": "No", "target": "Creator"}\n```'))
    assert r["game_related"] == "No" and r["target"] == "Creator" and r["category"] == ""


def test_parse_repairs_newlines_and_unknown_labels():
    r = parse_response(_resp('{"category":"Weird","sentiment":"Pos","confidence":"0.9",'
                             '"reasoning":"line one\nline two"}'))
    assert r["parse_ok"] and r["category"] == "Others" and "\n" not in r["reasoning"]


def test_parse_failure_is_flagged():
    assert parse_response(_resp("I cannot help with that"))["parse_ok"] is False


def test_parent_text_is_sent_to_the_model():
    row = {"video_title": "Trailer", "video_context": "desc", "parent_text": "AI voices are bad",
           "text_preprocessed": "agreed"}
    msg = build_user_message(row)
    assert "parent_context: AI voices are bad" in msg
    assert build_user_message(row, use_context=False) == "comment: agreed"


def test_preprocess_masks_urls_and_mentions():
    assert preprocess_text("@player-cw9nj see https://x.y  now") == "[USER] see [URL] now"


def test_spike_flags_zero_fills_missing_days():
    dates = pd.Series(pd.to_datetime(["2024-01-01"] * 5 + ["2024-01-03"] * 5))
    s = spike_flags(dates)
    assert len(s) == 3 and s.loc[1, "daily_volume"] == 0
