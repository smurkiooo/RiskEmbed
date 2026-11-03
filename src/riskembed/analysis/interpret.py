"""SHAP по группам признаков через pred_contrib самого LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shap_by_group(model, X: np.ndarray, feature_names: list[str], groups: dict[str, list[str]],
                  max_rows: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Средний |SHAP| по группам признаков (доля от суммы по всем группам)."""
    X = np.asarray(X)
    if len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X = X[idx]
    contrib = model.predict(X, num_iteration=model.best_iteration, pred_contrib=True)[:, :-1]  # последняя колонка — base value
    mean_abs = np.abs(contrib).mean(axis=0)
    pos = {n: j for j, n in enumerate(feature_names)}
    rows = []
    for group, names in groups.items():
        cols = [pos[n] for n in names if n in pos]
        rows.append({"group": group, "n_features": len(cols), "mean_abs_shap": float(mean_abs[cols].sum())})
    df = pd.DataFrame(rows)
    df["share"] = df["mean_abs_shap"] / max(df["mean_abs_shap"].sum(), 1e-12)
    return df.sort_values("share", ascending=False).reset_index(drop=True)


def top_features_by_shap(model, X: np.ndarray, feature_names: list[str], top: int = 20,
                         max_rows: int = 20_000, seed: int = 0) -> pd.DataFrame:
    X = np.asarray(X)
    if len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X = X[idx]
    contrib = model.predict(X, num_iteration=model.best_iteration, pred_contrib=True)[:, :-1]
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(contrib).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
