import numpy as np
import pandas as pd

from hlo.eval.model_compare import joint_label, selective_relabeling
from hlo.label.llm_labeler import CostTracker, parse_response


def test_joint_label():
    out = joint_label(["Yes", "No"], ["Game_Related", ""], ["Neg", ""])
    assert list(out) == ["Game_Related|Neg", "NotGame"]


def test_disagreement_routing_beats_random_when_errors_cause_disagreement():
    rng = np.random.default_rng(0)
    n = 1000
    gold = rng.choice(["A|Pos", "A|Neg", "NotGame"], n)
    llama = gold.copy()
    wrong = rng.random(n) < 0.15
    llama[wrong] = "B|Neu"
    gemma = gold.copy()
    gemma[wrong & (rng.random(n) < 0.9)] = "C|Pos"  # Gemma disagrees on 90% of Llama's errors
    gemma[~wrong & (rng.random(n) < 0.05)] = "C|Pos"  # and on 5% of correct ones
    df = pd.DataFrame({"gold_joint": gold, "llama_joint": llama, "gemma_joint": gemma,
                       "llama_conf": rng.uniform(0.8, 0.9, n)})
    r = selective_relabeling(df)
    assert r["accuracy_route_disagreements"] > r["accuracy_route_random_mean"] + 0.05
    assert r["errors_caught_by_disagreement"] > 0.85
    assert r["accuracy_llama_only"] < r["accuracy_route_disagreements"]


def test_gemma_pricing():
    t = CostTracker(["k"], 0.12, 0.36)
    assert abs(t.record("k", 1_000_000, 1_000_000) - 0.48) < 1e-9


def test_thinking_text_before_json_is_ignored():
    resp = {"choices": [{"message": {"content":
            '<think>The comment is about TTS…</think>\nHere is the label:\n'
            '{"game_related":"Yes","category":"AI_Voice_Related","sentiment":"Neg","confidence":"0.8","reasoning":"x"}'}}]}
    r = parse_response(resp)
    assert r["parse_ok"] and r["category"] == "AI_Voice_Related"
