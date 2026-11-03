import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from riskembed.evaluation.metrics import approval_rate_at_bad_rate, auc, gini_bootstrap


def test_auc_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = np.round(rng.normal(size=2000) + y * 0.5, 1)  # много связок
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    w = rng.integers(0, 4, 2000)
    assert auc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


def test_cluster_bootstrap_wider_than_row_bootstrap():
    rng = np.random.default_rng(1)
    n_cl, per = 300, 20
    cl = np.repeat(np.arange(n_cl), per)
    latent = np.repeat(rng.normal(size=n_cl), per)  # сильная внутрикластерная корреляция
    y = (rng.random(n_cl * per) < 1 / (1 + np.exp(-(latent * 2 - 2)))).astype(int)
    s = latent + rng.normal(0, 1, n_cl * per)
    clustered = gini_bootstrap(y, s, cl, n_reps=200)
    rowwise = gini_bootstrap(y, s, np.arange(len(y)), n_reps=200)
    assert clustered["std"] > rowwise["std"] * 1.5


def test_approval_rate():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    s = np.arange(8)
    assert approval_rate_at_bad_rate(y, s, 0.0) == 3 / 8
    assert approval_rate_at_bad_rate(y, s, 0.25) == 5 / 8
