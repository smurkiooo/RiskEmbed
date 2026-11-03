"""Остаточная проекция, отбор компонент, пробы, SHAP."""

import numpy as np
import pandas as pd
import pytest

from riskembed.analysis.interpret import shap_by_group, top_features_by_shap
from riskembed.analysis.probes import linear_probe, probe_report
from riskembed.features.residual import ResidualProjection
from riskembed.features.selection import component_psi, pareto_frontier, select_by_psi


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n, p, k = 4000, 6, 4
    X = rng.normal(size=(n, p))
    noise = rng.normal(scale=0.3, size=(n, k))
    # компоненты 0,1 — линейные функции ручных признаков; 2,3 — независимая информация
    E = np.column_stack([X[:, 0] * 2 - X[:, 1], X[:, 2] + 0.5 * X[:, 3], rng.normal(size=n), rng.normal(size=n)]) + noise
    return X, E


def test_residual_removes_explainable_part(toy):
    X, E = toy
    names = [f"f{i}" for i in range(X.shape[1])]
    emb = [f"emb_{i:03d}" for i in range(E.shape[1])]
    proj = ResidualProjection.fit(X[:2000], E[:2000], names, emb, alpha=1.0)
    # компоненты 0,1 объясняются ручными признаками, 2,3 — нет
    assert proj.r2[0] > 0.9 and proj.r2[1] > 0.9
    assert abs(proj.r2[2]) < 0.05 and abs(proj.r2[3]) < 0.05

    res = proj.transform(X[2000:], E[2000:])
    # в остатке корреляция с ручными признаками исчезает
    corr_before = np.abs(np.corrcoef(np.column_stack([X[2000:, 0], E[2000:, 0]]), rowvar=False)[0, 1])
    corr_after = np.abs(np.corrcoef(np.column_stack([X[2000:, 0], res[:, 0]]), rowvar=False)[0, 1])
    assert corr_before > 0.7 and corr_after < 0.1
    assert res.shape == E[2000:].shape and np.isfinite(res).all()


def test_residual_handles_nan_and_roundtrip(toy, tmp_path):
    X, E = toy
    X = X.copy()
    X[::7, 0] = np.nan  # пропуски в ручных признаках
    names = [f"f{i}" for i in range(X.shape[1])]
    emb = [f"emb_{i:03d}" for i in range(E.shape[1])]
    proj = ResidualProjection.fit(X, E, names, emb)
    proj.save(tmp_path / "proj")
    loaded = ResidualProjection.load(tmp_path / "proj")
    np.testing.assert_allclose(proj.transform(X, E), loaded.transform(X, E), atol=1e-6)
    assert loaded.residual_names()[0] == "res_000"


def test_residual_removes_missingness_structure():
    """Пропуск ручного признака сам по себе информативен; остаток не должен его сохранять."""
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.normal(size=(n, 3))
    missing = rng.random(n) < 0.3
    X[missing, 0] = np.nan
    E = np.column_stack([np.nan_to_num(X[:, 0]) + 2.0 * missing + 0.1 * rng.normal(size=n), rng.normal(size=n)])
    proj = ResidualProjection.fit(X, E, [f"f{i}" for i in range(3)], ["emb_000", "emb_001"], alpha=1.0)
    assert proj.r2[0] > 0.95
    res = proj.transform(X, E)
    assert abs(np.corrcoef(res[:, 0], missing)[0, 1]) < 0.05
    assert proj.missing_cols.tolist() == [0]


def test_component_psi_and_selection():
    rng = np.random.default_rng(1)
    n = 6000
    period = np.repeat(["p1", "p2", "p3"], n // 3)
    stable = rng.normal(size=n)
    drifting = rng.normal(size=n) + np.repeat([0.0, 0.5, 1.5], n // 3)  # сдвиг по периодам
    values = np.column_stack([stable, drifting])
    psi = component_psi(values, period, period == "p1")
    assert psi[0] < 0.05 < psi[1]
    assert select_by_psi(psi, 0.1).tolist() == [True, False]
    assert select_by_psi(psi, np.inf).all()


def test_pareto_frontier():
    rows = [
        {"name": "a", "gini": 0.30, "mean_psi": 0.10},
        {"name": "b", "gini": 0.32, "mean_psi": 0.20},
        {"name": "c", "gini": 0.28, "mean_psi": 0.30},  # доминируется обеими
    ]
    df = pareto_frontier(rows).set_index("name")
    assert df.loc["a", "pareto"] and df.loc["b", "pareto"] and not df.loc["c", "pareto"]


def test_linear_probe_recovers_encoded_concept(toy):
    X, E = toy
    tr, te = slice(0, 2000), slice(2000, 4000)
    # X[:,0] закодирован в E → высокий R²; чистый шум — около нуля
    r2_concept = linear_probe(E[tr], X[tr, 0], E[te], X[te, 0])
    noise = np.random.default_rng(5).normal(size=len(X))
    r2_noise = linear_probe(E[tr], noise[tr], E[te], noise[te])
    # потолок R² задан шумом в эмбеддинге (проверено сравнением с обычным МНК: 0.7928)
    assert r2_concept > 0.75 and abs(r2_noise) < 0.05


def test_probe_report_logs_skewed_targets(toy):
    X, E = toy
    targets = pd.DataFrame({"skewed": np.exp(X[:, 0] * 3), "plain": X[:, 1]})
    mask_tr = np.zeros(len(X), dtype=bool)
    mask_tr[:2000] = True
    rep = probe_report(E, targets, mask_tr, ~mask_tr)
    assert set(rep["target"]) == {"log1p(skewed)", "plain"}
    assert rep["r2"].max() > 0.6


class _FakeBooster:
    """Заглушка вместо LightGBM: TreeSHAP не наша ответственность, проверяем агрегацию.

    LightGBM нельзя импортировать в одном процессе с torch, а тесты энкодера его уже загрузили.
    """

    def __init__(self, contrib: np.ndarray):
        self._contrib = contrib
        self.best_iteration = 1

    def predict(self, X, num_iteration=None, pred_contrib=False):  # noqa: ARG002
        assert pred_contrib, "shap_by_group обязан запрашивать вклад признаков"
        base = np.zeros((len(X), 1))
        return np.hstack([self._contrib[: len(X)], base])  # последняя колонка — base value


def test_shap_by_group_sums_to_one():
    rng = np.random.default_rng(0)
    names = [f"f{i}" for i in range(3)] + [f"emb_{i:03d}" for i in range(2)]
    contrib = rng.normal(size=(500, 5)) * np.array([1.0, 2.0, 0.5, 4.0, 0.1])
    model = _FakeBooster(contrib)
    X = rng.normal(size=(500, 5))
    rep = shap_by_group(model, X, names, {"manual": names[:3], "embedding": names[3:]})
    assert rep["share"].sum() == pytest.approx(1.0)
    # группа с самым большим вкладом (emb_000 весит 4.0) должна быть первой
    assert rep.iloc[0]["group"] == "embedding"
    expected = np.abs(contrib[:, 3:]).mean(axis=0).sum()
    assert rep.iloc[0]["mean_abs_shap"] == pytest.approx(expected, rel=1e-9)

    top = top_features_by_shap(model, X, names, top=3)
    assert top.iloc[0]["feature"] == "emb_000" and top["mean_abs_shap"].is_monotonic_decreasing


def test_chunked_ridge_matches_dense_reference():
    rng = np.random.default_rng(92)
    X = rng.normal(size=(9000, 5)).astype(np.float32)
    E = rng.normal(size=(9000, 3))
    X[::7, 1] = np.nan
    X[:, 4] = np.nan
    fitted = ResidualProjection.fit(X, E, [f"x{i}" for i in range(5)], [f"emb_{i}" for i in range(3)], alpha=3)
    filled = np.where(np.isfinite(X), X, fitted.medians)
    dense = np.column_stack([filled, (~np.isfinite(X[:, fitted.missing_cols])).astype(float)])
    mean, scale = dense.mean(axis=0), dense.std(axis=0)
    scale[scale < 1e-9] = 1
    Z = (dense - mean) / scale
    reference_coef = np.linalg.solve(Z.T @ Z + 3 * np.eye(Z.shape[1]), Z.T @ (E - E.mean(axis=0)))
    expected = E - (Z @ reference_coef + E.mean(axis=0))
    np.testing.assert_allclose(fitted.transform(X, E), expected, atol=1e-6)
