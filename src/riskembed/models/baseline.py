"""LightGBM и единый формат метрик для строк таблицы абляции."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from riskembed.evaluation.metrics import (
    approval_rate_at_bad_rate,
    gini,
    gini_bootstrap,
    gini_by_period,
)
from riskembed.features.psi import psi_numeric


def train_lgb(
    X_train, y_train, X_valid, y_valid, params: dict, feature_names: list[str],
    num_boost_round: int = 2000, early_stopping_rounds: int = 100, seed: int = 42,
) -> lgb.Booster:
    params = {"num_threads": 4, **params, "seed": seed, "deterministic": True, "metric": "auc"}
    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_names, free_raw_data=True)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain, free_raw_data=True)
    return lgb.train(
        params, dtrain, num_boost_round=num_boost_round, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def importance_frame(model: lgb.Booster) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": model.feature_name(), "gain": model.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["gain_share"] = imp["gain"] / max(imp["gain"].sum(), 1e-12)
    return imp


def score_report(
    scored: pd.DataFrame,
    name: str,
    main_split: str = "oot",
    target: str = "target_pd",
    bootstrap_reps: int = 500,
    seed: int = 42,
    period_col: str | None = "report_date",
    business_metric: bool = True,
) -> tuple[dict, pd.DataFrame | None]:
    """scored: client_id, split, target, score (+ period_col). Возвращает (metrics, gini по периодам main_split)."""
    metrics: dict = {"name": name}
    for split, part in scored.groupby("split"):
        metrics[f"gini_{split}"] = gini(part[target], part["score"])
        metrics[f"n_{split}"] = len(part)
        metrics[f"bad_rate_{split}"] = float(part[target].mean())
    main = scored[scored["split"] == main_split]
    boot = gini_bootstrap(main[target], main["score"], main["client_id"], n_reps=bootstrap_reps, seed=seed)
    metrics |= {f"gini_{main_split}_ci_low": boot["ci_low"], f"gini_{main_split}_ci_high": boot["ci_high"]}

    train_score = scored.loc[scored["split"] == "train", "score"]
    for split in scored["split"].unique():
        if split != "train":
            metrics[f"psi_score_{split}"] = psi_numeric(train_score, scored.loc[scored["split"] == split, "score"])
    if business_metric:
        target_bad = float(scored.loc[scored["split"] == "train", target].mean())
        metrics[f"approval_rate_{main_split}_at_train_bad_rate"] = approval_rate_at_bad_rate(main[target], main["score"], target_bad)

    monthly = None
    if period_col and period_col in main and main[period_col].nunique() > 1:
        monthly = gini_by_period(main.assign(month=main[period_col].dt.strftime("%Y-%m")), target, "score", "month")
    return metrics, monthly


