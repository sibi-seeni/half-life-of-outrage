"""Interrupted time series around an event (segmented regression, Newey-West errors).

    y_t = b0 + b1 * t + b2 * post_t + b3 * (t * post_t) + e_t

b2 = immediate level change at the event, b3 = change in slope afterwards.
Days are weighted by comment counts; days with fewer than `min_n` comments are dropped.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def segmented_regression(series, event_date, value_col="neg_share", window=60, min_n=5, maxlags=7):
    event_date = pd.Timestamp(event_date)
    seg = series.loc[event_date - pd.Timedelta(days=window): event_date + pd.Timedelta(days=window)]
    seg = seg[seg["n"] >= min_n]
    if (seg.index < event_date).sum() < 7 or (seg.index >= event_date).sum() < 7:
        return {"status": "insufficient_data"}
    t = (seg.index - event_date).days.values.astype(float)
    post = (t >= 0).astype(float)
    X = sm.add_constant(np.column_stack([t, post, t * post]))
    model = sm.WLS(seg[value_col].values, X, weights=seg["n"].values).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    names = ["intercept", "pre_trend", "level_change", "slope_change"]
    out = {"status": "ok", "n_days": int(len(seg)), "n_comments": int(seg["n"].sum())}
    for i, name in enumerate(names):
        lo, hi = model.conf_int()[i]
        out[name] = {"coef": float(model.params[i]), "ci95": [float(lo), float(hi)],
                     "p": float(model.pvalues[i])}
    return out
