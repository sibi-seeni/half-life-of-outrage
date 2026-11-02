import json

import pandas as pd
import pytest
import requests

from hlo.data.build_dataset import preprocess_text, spike_flags
from hlo.label import llm_labeler
from hlo.label.llm_labeler import (CostTracker, KeyRotator, build_user_message, key_is_dead, label_one,
                                   parse_response)


def _resp(content):
    return {"choices": [{"message": {"content": content}}]}


def _http(status, error):
    r = requests.Response()
    r.status_code, r._content = status, json.dumps({"error": error}).encode()
    return r


def test_spent_or_unauthorised_keys_are_dead():
    assert key_is_dead(_http(422, {"type": "budget_exceeded", "message": "Budget has been exceeded!"}))
    assert key_is_dead(_http(400, {"type": "invalid_request_error",
                                   "message": "Invalid model name passed in model=llama-3.3-70b-instruct"}))
    assert key_is_dead(_http(401, {"type": "token_not_found_in_db"}))
    assert not key_is_dead(_http(429, {"type": "rate_limit_error"}))
    assert not key_is_dead(_http(502, {}))


def test_dead_key_is_dropped_and_good_key_labels(monkeypatch):
    def fake_call(key, *a, **k):
        if key == "spent":
            raise requests.exceptions.HTTPError(response=_http(422, {"type": "budget_exceeded"}))
        return _resp('{"game_related": "No", "target": "Other"}')
    monkeypatch.setattr(llm_labeler, "call_api", fake_call)
    rot = KeyRotator(["spent", "good"])
    r = label_one({"text_preprocessed": "hi"}, rot, CostTracker(rot.keys, 0, 0), "", "m", False)
    assert r["parse_ok"] and rot.active == [1]


def test_all_keys_dead_stops_instead_of_writing_errors(monkeypatch):
    def fake_call(key, *a, **k):
        raise requests.exceptions.HTTPError(response=_http(422, {"type": "budget_exceeded"}))
    monkeypatch.setattr(llm_labeler, "call_api", fake_call)
    rot = KeyRotator(["a", "b"])
    with pytest.raises(RuntimeError, match="All API keys retired"):
        label_one({"text_preprocessed": "hi"}, rot, CostTracker(rot.keys, 0, 0), "", "m", False)


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
