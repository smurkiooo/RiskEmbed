"""Отбор компонент эмбеддинга по стабильности и Парето-фронт качество/PSI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskembed.features.psi import psi_numeric


def component_psi(
    values: np.ndarray, period: np.ndarray, reference_mask: np.ndarray, n_bins: int = 10, agg: str = "max"
) -> np.ndarray:
    """PSI каждой компоненты относительно эталона, агрегированный по периодам (max — худший период)."""
    values = np.asarray(values, dtype=float)
    period = np.asarray(period)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    out = np.zeros(values.shape[1])
    ref = values[reference_mask]
    for j in range(values.shape[1]):
        per_period = [psi_numeric(ref[:, j], values[period == p, j], n_bins) for p in np.unique(period)]
        out[j] = max(per_period) if agg == "max" else float(np.mean(per_period))
    return out


def select_by_psi(psi: np.ndarray, threshold: float) -> np.ndarray:
    """Маска компонент со стабильностью не хуже порога (threshold=inf — взять все)."""
    return np.asarray(psi) <= threshold


def pareto_frontier(rows: list[dict], gini_key: str = "gini", psi_key: str = "mean_psi") -> pd.DataFrame:
    """Помечает строки, которые не доминируются другими (больше Gini и меньше PSI одновременно)."""
    df = pd.DataFrame(rows)
    dominated = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        better = (df[gini_key] >= df.loc[i, gini_key]) & (df[psi_key] <= df.loc[i, psi_key])
        strictly = (df[gini_key] > df.loc[i, gini_key]) | (df[psi_key] < df.loc[i, psi_key])
        dominated[i] = bool((better & strictly).any())
    df["pareto"] = ~dominated
    return df
