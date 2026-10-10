"""Agreement metrics for nominal labels with bootstrap confidence intervals."""
import numpy as np
import pandas as pd


def percent_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean())


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    p_o = (a == b).mean()
    p_e = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((p_o - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def gwet_ac1(a, b):
    """Gwet's AC1 for two raters: (p_a - p_e) / (1 - p_e), p_e = sum(pi_k (1 - pi_k)) / (q - 1)."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    q = len(cats)
    p_a = (a == b).mean()
    if q < 2:
        return 1.0
    pi = np.array([((a == c).mean() + (b == c).mean()) / 2 for c in cats])
    p_e = (pi * (1 - pi)).sum() / (q - 1)
    return float((p_a - p_e) / (1 - p_e))


def krippendorff_alpha_nominal(*raters):
    """Krippendorff's alpha (nominal) for any number of raters; NaN/None = missing."""
    data = pd.DataFrame({i: pd.Series(r, dtype="object") for i, r in enumerate(raters)})
    cats = sorted({v for col in data for v in data[col].dropna()})
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for _, row in data.iterrows():
        vals = [idx[v] for v in row.dropna()]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[vals[i], vals[j]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    d_o = o.sum() - np.trace(o)
    d_e = (n * n - (n_c ** 2).sum()) / (n - 1)
    return float(1 - d_o / d_e) if d_e > 0 else 1.0


METRICS = {"percent_agreement": percent_agreement, "cohen_kappa": cohen_kappa,
           "gwet_ac1": gwet_ac1, "krippendorff_alpha": krippendorff_alpha_nominal}


def bootstrap_ci(a, b, fn, n_boot=2000, alpha=0.05, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    stats = [fn(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def agreement_report(a, b, n_boot=2000):
    """All metrics with 95% bootstrap CIs, as a dict ready for JSON."""
    out = {"n": int(len(a))}
    for name, fn in METRICS.items():
        lo, hi = bootstrap_ci(a, b, fn, n_boot)
        out[name] = {"value": round(fn(a, b), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out
