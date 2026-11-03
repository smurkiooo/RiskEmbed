"""Gini, кластерный бутстрап по клиенту, парные сравнения, бизнес-метрика."""

from __future__ import annotations

import numpy as np
import pandas as pd


def auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    """AUC через ранги (Mann–Whitney), с весами для бутстрапа."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)
    keep = w > 0
    y, score, w = y[keep], score[keep], w[keep]
    pos_w = (w * y).sum()
    neg_w = (w * (1 - y)).sum()
    if pos_w == 0 or neg_w == 0:
        return np.nan
    # взвешенные средние ранги: сортируем, для связок усредняем кумулятивный вес
    order = np.argsort(score, kind="mergesort")
    s, yy, ww = score[order], y[order], w[order]
    cum = np.cumsum(ww)
    ranks = cum - ww / 2  # ранг в «весовой» шкале
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        ranks = (np.bincount(inv, weights=ranks * ww) / np.bincount(inv, weights=ww))[inv]
    return float(((ranks * ww * yy).sum() - pos_w * pos_w / 2) / (pos_w * neg_w))


def gini(y, score, weight=None) -> float:
    return 2 * auc(y, score, weight) - 1


def gini_bootstrap(
    y, score, cluster, n_reps: int = 500, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    y = np.asarray(y)
    score = np.asarray(score)
    _, cl = np.unique(np.asarray(cluster), return_inverse=True)
    n_cl = cl.max() + 1
    rng = np.random.default_rng(seed)
    reps = np.empty(n_reps)
    for i in range(n_reps):
        cl_weight = rng.multinomial(n_cl, np.full(n_cl, 1.0 / n_cl))  # сколько раз взят каждый клиент
        reps[i] = gini(y, score, cl_weight[cl])
    lo, hi = np.nanquantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"gini": gini(y, score), "ci_low": float(lo), "ci_high": float(hi), "std": float(np.nanstd(reps))}


def paired_gini_delta_bootstrap(y, score_a, score_b, cluster, n_reps: int = 500, seed: int = 0) -> dict[str, float]:
    """Прирост Gini(b) - Gini(a) на одних и тех же бутстрап-выборках (парный тест для абляций)."""
    y, a, b = map(np.asarray, (y, score_a, score_b))
    _, cl = np.unique(np.asarray(cluster), return_inverse=True)
    n_cl = cl.max() + 1
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_reps)
    for i in range(n_reps):
        w = rng.multinomial(n_cl, np.full(n_cl, 1.0 / n_cl))[cl]
        deltas[i] = gini(y, b, w) - gini(y, a, w)
    return {
        "delta": gini(y, b) - gini(y, a),
        "ci_low": float(np.nanquantile(deltas, 0.025)),
        "ci_high": float(np.nanquantile(deltas, 0.975)),
        "p_le_0": float(np.mean(deltas <= 0)),
    }


def gini_by_period(df: pd.DataFrame, target: str, score: str, period: str) -> pd.DataFrame:
    rows = []
    for p, part in df.groupby(period, sort=True):
        rows.append({period: p, "n": len(part), "bad_rate": part[target].mean(), "gini": gini(part[target], part[score])})
    return pd.DataFrame(rows)


def approval_rate_at_bad_rate(y, score, max_bad_rate: float) -> float:
    """Бизнес-метрика: максимальная доля одобрений (одобряем самых «хороших» по скору),
    при которой доля дефолтов среди одобренных не превышает max_bad_rate."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")  # низкий скор = низкий риск
    cum_bad = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.nonzero(cum_bad <= max_bad_rate)[0]
    return float((ok.max() + 1) / len(y)) if len(ok) else 0.0
