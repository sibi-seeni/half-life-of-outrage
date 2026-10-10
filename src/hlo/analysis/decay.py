"""Sentiment half-life around an event.

Replaces the original notebook fit, which modelled comment *volume* from the global volume
peak (often days before the data ended). Here the decaying quantity is the event-attributable
excess negativity of one aspect:

    E(t) = N(t) - B,     N(t) = daily negative share,  B = pre-event baseline (days -baseline..-1)
    E(t) ~ A * exp(-lambda * t),   t = days since the post-event peak (searched within `peak_search` days)
    half-life = ln 2 / lambda

The fit is weighted by daily comment counts, needs at least `min_days` days with >= `min_n`
comments, and gets a percentile CI from a day-level bootstrap (resample comments within each day,
which preserves the timeline). A model-free half-life (first day the 7-day-smoothed excess falls
below half its peak) is reported alongside, as is the same fit on volume ("attention decay").
"""
import numpy as np
from scipy.optimize import curve_fit

LN2 = np.log(2)


def _exp(t, a, lam):
    return a * np.exp(-lam * t)


def fit_exponential(t, y, w=None):
    """Weighted least squares exp fit. Returns (A, lambda) or None."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 3 or not np.isfinite(y).all() or y.max() <= 0:
        return None
    sigma = None if w is None else 1 / np.sqrt(np.maximum(np.asarray(w, float), 1e-9))
    try:
        (a, lam), _ = curve_fit(_exp, t, y, p0=[max(y[0], 1e-6), 0.1], sigma=sigma,
                                bounds=([0, 1e-6], [np.inf, 10]), maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    return a, lam


def _segment(series, event_date, baseline, horizon, peak_search, value_col, min_n):
    pre = series.loc[event_date - np.timedelta64(baseline, "D"): event_date - np.timedelta64(1, "D")]
    pre = pre[pre["n"] >= min_n]
    if len(pre) < 3:
        return None
    base = np.average(pre[value_col], weights=pre["n"]) if value_col != "n" else pre["n"].mean()
    post = series.loc[event_date: event_date + np.timedelta64(horizon + peak_search, "D")].copy()
    post = post[post["n"] >= min_n]
    if post.empty:
        return None
    early = post.loc[: event_date + np.timedelta64(peak_search, "D")]
    if early.empty:
        return None
    peak_day = early[value_col].idxmax()
    seg = post.loc[peak_day: peak_day + np.timedelta64(horizon, "D")]
    t = (seg.index - peak_day).days.values
    return base, peak_day, t, (seg[value_col] - base).values, seg["n"].values


def model_free_half_life(t, excess, smooth=7):
    """First day the centred rolling-mean excess drops below half of its starting value."""
    if len(excess) == 0 or excess[0] <= 0:
        return None
    full = np.full(t.max() + 1, np.nan)
    full[t] = excess
    sm = np.array([np.nanmean(full[max(0, i - smooth // 2): i + smooth // 2 + 1]) for i in range(len(full))])
    below = np.where(sm <= sm[0] / 2)[0]
    return int(below[0]) if len(below) else None


def event_half_life(series, event_date, value_col="neg_share", baseline=30, horizon=60,
                    peak_search=3, min_n=5, min_days=10, n_boot=500, seed=0):
    event_date = np.datetime64(event_date, "D")
    seg = _segment(series, event_date, baseline, horizon, peak_search, value_col, min_n)
    if seg is None:
        return {"status": "insufficient_data"}
    base, peak_day, t, excess, n = seg
    res = {"baseline": float(base), "peak_day": str(peak_day.date()), "peak_excess": float(excess[0]),
           "days_used": int(len(t)), "comments_used": int(n.sum()),
           "model_free_half_life": model_free_half_life(t, excess)}
    if len(t) < min_days:
        return res | {"status": "too_few_days"}
    if excess[0] <= 0:
        return res | {"status": "no_excess_at_peak"}
    fit = fit_exponential(t, excess, n if value_col != "n" else None)
    if fit is None:
        return res | {"status": "fit_failed"}
    res |= {"status": "ok", "A": float(fit[0]), "lambda": float(fit[1]), "half_life": float(LN2 / fit[1])}

    if value_col == "neg_share" and n_boot:
        rng = np.random.default_rng(seed)
        p = np.clip(excess + base, 0, 1)
        hl = []
        for _ in range(n_boot):
            yb = rng.binomial(n, p) / n - base
            fb = fit_exponential(t, yb, n)
            if fb is not None:
                hl.append(LN2 / fb[1])
        if hl:
            res["half_life_ci95"] = [float(np.quantile(hl, 0.025)), float(np.quantile(hl, 0.975))]
            res["bootstrap_success"] = len(hl) / n_boot
    return res
