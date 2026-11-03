"""Адаптер AmEx Default Prediction (parquet-сборка raddar)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from riskembed.config import resolve

ID_COL = "customer_ID"
DATE_COL = "S_2"
# категориальные признаки по описанию соревнования
CAT_COLS = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
INT_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT")
NEG_WEIGHT = 20.0  # вес отрицательного класса (обратная доля прореживания)


def customer_int_id(customer_id: pd.Series | np.ndarray) -> np.ndarray:
    """Детерминированный int64 из hex-хэша customer_ID (первые 15 hex-символов = 60 бит).

    Нужен для хэш-сплита (evaluation.splits.holdout_mask) и компактного хранения;
    не зависит от порядка строк и состава выборки.
    """
    return np.array([int(s[:15], 16) for s in np.asarray(customer_id)], dtype=np.int64)


def raw_paths(raw_dir: str | Path) -> tuple[Path, Path]:
    raw = resolve(raw_dir)
    return raw / "train.parquet", raw / "train_labels.csv"


def column_types(con: duckdb.DuckDBPyConnection, parquet: Path) -> dict[str, str]:
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM '{parquet}'").fetchall()}


def feature_columns(types: dict[str, str]) -> tuple[list[str], list[str]]:
    """(числовые, категориальные) колонки признаков."""
    cats = [c for c in CAT_COLS if c in types]
    nums = [c for c in types if c not in (ID_COL, DATE_COL, *cats)]
    return nums, cats


def register_amex(con: duckdb.DuckDBPyConnection, raw_dir: str | Path, name: str = "amex") -> tuple[list[str], list[str]]:
    """View `name` с нормализованными пропусками: -1 → NULL в целочисленных колонках, S_2 → DATE.

    customer_ID остаётся строкой; целочисленный id считается в Python (customer_int_id),
    для стабильного разбиения клиентов независимо от порядка строк.
    """
    parquet, _ = raw_paths(raw_dir)
    types = column_types(con, parquet)
    nums, cats = feature_columns(types)
    exprs = []
    for c in nums + cats:
        if types[c] in INT_TYPES:
            exprs.append(f"CASE WHEN {c} = -1 THEN NULL ELSE {c} END AS {c}")
        else:
            exprs.append(f"CASE WHEN isnan({c}) THEN NULL ELSE {c} END AS {c}")
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT {ID_COL}, {DATE_COL}::DATE AS {DATE_COL}, "
        + ", ".join(exprs)
        + f" FROM read_parquet('{parquet}')"
    )
    return nums, cats


def load_labels(raw_dir: str | Path) -> pd.DataFrame:
    _, labels_csv = raw_paths(raw_dir)
    lab = pd.read_csv(labels_csv)
    lab["cid"] = customer_int_id(lab[ID_COL])
    if lab["cid"].duplicated().any():
        raise ValueError("Коллизия customer_int_id — увеличьте число hex-символов")
    return lab


def aggregate_sql(nums: list[str], cats: list[str], source: str = "amex", cutoff: str | None = None) -> str:
    """Классическая витрина агрегатов по выпискам клиента (last/mean/std/min/max + категории).

    cutoff: если задан — используются только выписки с S_2 < cutoff (сдвинутый срез).
    """
    where = f"WHERE {DATE_COL} < DATE '{cutoff}'" if cutoff else ""  # source может быть подзапросом (чанк)
    sel = [
        f"{ID_COL}",
        "count(*)::FLOAT AS n_statements",
        f"date_diff('day', min({DATE_COL}), max({DATE_COL}))::FLOAT AS history_days",
        f"(date_diff('day', min({DATE_COL}), max({DATE_COL})) / nullif(count(*) - 1, 0))::FLOAT AS mean_gap_days",
    ]
    for c in nums:
        sel += [
            f"arg_max_null({c}, {DATE_COL})::FLOAT AS {c}__last",
            f"avg({c})::FLOAT AS {c}__mean",
            f"stddev_samp({c})::FLOAT AS {c}__std",
            f"min({c})::FLOAT AS {c}__min",
            f"max({c})::FLOAT AS {c}__max",
            # изменение за историю: last − first (тренд, которого нет в mean/std)
            f"(arg_max_null({c}, {DATE_COL}) - arg_min_null({c}, {DATE_COL}))::FLOAT AS {c}__delta",
        ]
    for c in cats:
        sel += [
            f"arg_max_null({c}, {DATE_COL})::FLOAT AS {c}__last",
            f"count(DISTINCT {c})::FLOAT AS {c}__nunique",
        ]
    return f"SELECT {', '.join(sel)} FROM {source} {where} GROUP BY {ID_COL}"


def aggregate_feature_names(nums: list[str], cats: list[str]) -> list[str]:
    names = ["n_statements", "history_days", "mean_gap_days"]
    names += [f"{c}__{a}" for c in nums for a in ("last", "mean", "std", "min", "max", "delta")]
    names += [f"{c}__{a}" for c in cats for a in ("last", "nunique")]
    return names


def amex_metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Метрика соревнования M = 0.5 · (G + D) с весом 20 у отрицательного класса.

    G — нормализованный взвешенный Gini, D — доля дефолтов в топ-4% скоров по весу.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    w = np.where(y[order] == 0, NEG_WEIGHT, 1.0)
    cut = np.cumsum(w) <= int(0.04 * w.sum())
    d = y[order][cut].sum() / y.sum()

    def lorenz_gini(ordering):
        target = y[ordering]
        weight = np.where(target == 0, NEG_WEIGHT, 1.0)
        random = np.cumsum(weight) / weight.sum()
        lorenz = np.cumsum(target * weight) / np.sum(target * weight)
        return np.sum((lorenz - random) * weight)

    # Use the exact competition normalization, including its finite-sample correction.
    g = lorenz_gini(order) / lorenz_gini(np.argsort(-y, kind="mergesort"))
    return {"amex_M": 0.5 * (g + d), "amex_G_weighted": g, "amex_D_top4": d}
