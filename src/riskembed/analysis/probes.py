"""Linear probing: что линейно восстанавливается из эмбеддинга."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    mean = X.mean(axis=0)
    std = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
    Z = (X - mean) / std
    gram = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(gram, Z.T @ (y - y.mean()))
    return np.concatenate([w / std, [y.mean() - (mean / std) @ w]]), float(y.mean())


def _predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ w[:-1] + w[-1]


def linear_probe(
    emb_train: np.ndarray, y_train: np.ndarray, emb_test: np.ndarray, y_test: np.ndarray, alpha: float = 1.0
) -> float:
    """R² линейного восстановления y из эмбеддинга, на отложенных наблюдениях."""
    ok_tr = np.isfinite(emb_train).all(axis=1) & np.isfinite(y_train)
    ok_te = np.isfinite(emb_test).all(axis=1) & np.isfinite(y_test)
    if ok_tr.sum() < 50 or ok_te.sum() < 50:
        return np.nan
    w, _ = _ridge_fit(emb_train[ok_tr], y_train[ok_tr], alpha)
    pred = _predict(w, emb_test[ok_te])
    y = y_test[ok_te]
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def probe_report(
    emb: np.ndarray, targets: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray, alpha: float = 1.0
) -> pd.DataFrame:
    """R² по каждой колонке targets. Логарифмирует сильно скошенные положительные величины."""
    rows = []
    for name in targets.columns:
        y = targets[name].to_numpy(dtype=float)
        if np.nanmin(y) >= 0 and np.nanmax(y) > 0 and (np.nanmax(y) / max(np.nanmedian(y), 1e-9) > 50):
            y, name = np.log1p(y), f"log1p({name})"
        rows.append({"target": name,
                     "r2": linear_probe(emb[train_mask], y[train_mask], emb[test_mask], y[test_mask], alpha)})
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
