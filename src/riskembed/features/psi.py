"""Population Stability Index по квантильным бинам эталона."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    ref = reference[~np.isnan(reference)]
    if len(ref) == 0:
        return np.array([-np.inf, np.inf])
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], qs, [np.inf]])


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Доли по бинам + отдельный бин для NaN (пропуск — тоже часть распределения)."""
    nan = np.isnan(values)
    counts = np.histogram(values[~nan], bins=edges)[0]
    counts = np.append(counts, nan.sum()).astype(float)
    return counts / max(counts.sum(), 1.0)


def psi_numeric(reference, current, n_bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = _bin_edges(reference, n_bins)
    e = np.clip(_shares(reference, edges), EPS, None)
    a = np.clip(_shares(current, edges), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(reference, current) -> float:
    ref = pd.Series(reference).astype("object").fillna("__nan__").value_counts(normalize=True)
    cur = pd.Series(current).astype("object").fillna("__nan__").value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    e = np.clip(ref.reindex(cats, fill_value=0).to_numpy(), EPS, None)
    a = np.clip(cur.reindex(cats, fill_value=0).to_numpy(), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(
    df: pd.DataFrame, features: list[str], period_col: str, reference_mask, n_bins: int = 10
) -> pd.DataFrame:
    """PSI признаков по периодам относительно эталона: строки — признаки, колонки — периоды.

    reference_mask должна быть выровнена с df по длине: булева Series от другого датафрейма
    сопоставилась бы по индексу и дала бы не тот эталон.
    """
    mask = np.asarray(reference_mask, dtype=bool)
    if len(mask) != len(df):
        raise ValueError(f"reference_mask ({len(mask)}) не выровнена с df ({len(df)})")
    ref = df.loc[mask]
    rows = {}
    for f in features:
        edges = _bin_edges(ref[f].to_numpy(dtype=float), n_bins)
        e = np.clip(_shares(ref[f].to_numpy(dtype=float), edges), EPS, None)
        row = {}
        for period, part in df.groupby(period_col, sort=True):
            a = np.clip(_shares(part[f].to_numpy(dtype=float), edges), EPS, None)
            row[period] = float(np.sum((a - e) * np.log(a / e)))
        rows[f] = row
    return pd.DataFrame(rows).T
