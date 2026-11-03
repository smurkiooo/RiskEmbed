"""Остаточные эмбеддинги: вычитание гребневой проекции на ручные признаки."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ResidualProjection:
    feature_names: list[str]
    emb_names: list[str]
    medians: np.ndarray  # импутация пропусков ручных признаков (по train)
    missing_cols: np.ndarray  # индексы признаков, для которых добавлен индикатор пропуска
    mean_x: np.ndarray
    std_x: np.ndarray
    coef: np.ndarray  # [n_features, n_emb]
    intercept: np.ndarray  # [n_emb]
    r2: np.ndarray  # [n_emb] — доля дисперсии компоненты, объяснённая ручными признаками (train)
    alpha: float

    @classmethod
    def fit(cls, X: np.ndarray, E: np.ndarray, feature_names: list[str], emb_names: list[str],
            alpha: float = 10.0) -> "ResidualProjection":
        """X: ручные признаки [n, p] (с NaN), E: эмбеддинг [n, k]. Обе матрицы — только train."""
        # Keep the input's compact dtype; only bounded design batches become float64.
        X = np.asarray(X)
        E = np.asarray(E, dtype=np.float64)
        ok = np.isfinite(E).all(axis=1)
        if not ok.all():
            X, E = X[ok], E[ok]
        if not len(X):
            raise ValueError("No finite training embeddings")
        medians = np.zeros(X.shape[1], dtype=np.float64)
        missing = []
        for j in range(X.shape[1]):
            column = X[:, j]
            finite = np.isfinite(column)
            if finite.any():
                medians[j] = np.median(column[finite])
            if not finite.all():
                missing.append(j)
        missing_cols = np.asarray(missing, dtype=np.int64)
        width = X.shape[1] + len(missing_cols)
        sum_x, sum_x2 = np.zeros(width), np.zeros(width)
        batch_size = 8192
        for start in range(0, len(X), batch_size):
            design = cls._design(X[start:start + batch_size], medians, missing_cols)
            sum_x += design.sum(axis=0)
            sum_x2 += np.square(design).sum(axis=0)
        mean_x = sum_x / len(X)
        std_x = np.sqrt(np.maximum(sum_x2 / len(X) - mean_x**2, 0))
        std_x[std_x < 1e-9] = 1.0
        intercept = E.mean(axis=0)
        gram = alpha * np.eye(width)
        rhs = np.zeros((width, E.shape[1]))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            design = cls._design(X[start:end], medians, missing_cols)
            Z = (design - mean_x) / std_x
            gram += Z.T @ Z
            rhs += Z.T @ (E[start:end] - intercept)
        coef = np.linalg.solve(gram, rhs)
        sum_r, sum_r2 = np.zeros(E.shape[1]), np.zeros(E.shape[1])
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            Z = (cls._design(X[start:end], medians, missing_cols) - mean_x) / std_x
            resid = E[start:end] - (Z @ coef + intercept)
            sum_r += resid.sum(axis=0)
            sum_r2 += np.square(resid).sum(axis=0)
        residual_var = np.maximum(sum_r2 / len(X) - (sum_r / len(X))**2, 0)
        var = E.var(axis=0)
        r2 = 1 - residual_var / np.where(var > 1e-12, var, 1.0)
        return cls(list(feature_names), list(emb_names), medians, missing_cols, mean_x, std_x,
                   coef, intercept, r2, alpha)

    @staticmethod
    def _design(X: np.ndarray, medians: np.ndarray, missing_cols: np.ndarray) -> np.ndarray:
        """Значения с импутацией + индикаторы пропуска: иначе «признака не было» остаётся в остатке."""
        filled = np.where(np.isfinite(X), X, medians)
        if len(missing_cols) == 0:
            return filled
        return np.hstack([filled, (~np.isfinite(X[:, missing_cols])).astype(np.float64)])

    def transform(self, X: np.ndarray, E: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        E = np.asarray(E)
        out = np.empty(E.shape, dtype=np.float32)
        for start in range(0, len(X), 8192):
            end = start + 8192
            Xi = self._design(X[start:end], self.medians, self.missing_cols)
            Z = (Xi - self.mean_x) / self.std_x
            out[start:end] = E[start:end] - (Z @ self.coef + self.intercept)
        return out

    def residual_names(self, prefix: str = "res") -> list[str]:
        return [f"{prefix}_{n.split('_')[-1]}" for n in self.emb_names]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(path.with_suffix(".npz"), medians=self.medians, missing_cols=self.missing_cols,
                 mean_x=self.mean_x, std_x=self.std_x, coef=self.coef, intercept=self.intercept, r2=self.r2)
        path.with_suffix(".json").write_text(json.dumps(
            {"feature_names": self.feature_names, "emb_names": self.emb_names, "alpha": self.alpha}))

    @classmethod
    def load(cls, path: str | Path) -> "ResidualProjection":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz")) as z:
            return cls(meta["feature_names"], meta["emb_names"], z["medians"], z["missing_cols"],
                       z["mean_x"], z["std_x"], z["coef"], z["intercept"], z["r2"], meta["alpha"])
