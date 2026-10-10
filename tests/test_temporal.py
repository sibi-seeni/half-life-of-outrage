import numpy as np
import pandas as pd
import pytest

from hlo.analysis.decay import event_half_life, fit_exponential
from hlo.analysis.its import segmented_regression


def _series(neg_share, n=200, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(neg_share), freq="D")
    s = pd.DataFrame({"n": n, "neg_share": neg_share}, index=idx)
    s["n_neg"] = s["neg_share"] * s["n"]
    return s


def test_fit_recovers_lambda():
    t = np.arange(40)
    a, lam = fit_exponential(t, 0.3 * np.exp(-0.1 * t))
    assert lam == pytest.approx(0.1, rel=1e-3)
    assert a == pytest.approx(0.3, rel=1e-3)


def test_event_half_life_recovers_known_value():
    true_hl = 5.0
    lam = np.log(2) / true_hl
    base = 0.30
    pre = np.full(40, base)
    post = base + 0.35 * np.exp(-lam * np.arange(80))
    s = _series(np.concatenate([pre, post]), n=400)
    event = s.index[40]
    res = event_half_life(s, event, n_boot=200)
    assert res["status"] == "ok"
    assert res["half_life"] == pytest.approx(true_hl, rel=0.05)
    lo, hi = res["half_life_ci95"]
    assert lo < true_hl < hi
    assert res["model_free_half_life"] in range(3, 8)


def test_peak_found_after_short_lag():
    base, lam = 0.2, np.log(2) / 4
    pre = np.full(40, base)
    post = np.concatenate([[base, base + 0.1], base + 0.4 * np.exp(-lam * np.arange(78))])
    s = _series(np.concatenate([pre, post]))
    res = event_half_life(s, s.index[40], n_boot=0)
    assert res["peak_day"] == str(s.index[42].date())
    assert res["half_life"] == pytest.approx(4, rel=0.05)


def test_insufficient_data_is_reported_not_fitted():
    s = _series(np.full(45, 0.3))
    s.iloc[:40, s.columns.get_loc("n")] = 0  # no pre-event data
    assert event_half_life(s, s.index[40])["status"] == "insufficient_data"


def test_its_recovers_level_change():
    rng = np.random.default_rng(1)
    t = np.arange(-60, 61)
    y = 0.3 + 0.001 * t + 0.15 * (t >= 0) + rng.normal(0, 0.01, len(t))
    s = _series(y, start="2025-01-01")
    res = segmented_regression(s, s.index[60])
    assert res["status"] == "ok"
    assert res["level_change"]["coef"] == pytest.approx(0.15, abs=0.02)
    assert res["level_change"]["p"] < 0.001
    assert abs(res["slope_change"]["coef"]) < 0.002
