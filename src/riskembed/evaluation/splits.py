"""Deterministic customer-disjoint train/validation/test splits."""

import numpy as np
import pandas as pd


def holdout_mask(client_id: pd.Series | np.ndarray, share: float, salt: int = 0) -> np.ndarray:
    """Детерминированный хэш-сплит клиентов (не зависит от состава выборки и порядка строк)."""
    ids = np.asarray(client_id)
    if np.issubdtype(ids.dtype, np.signedinteger) and (ids < 0).any():
        raise ValueError("client_id должен быть неотрицательным целым")
    x = (ids.astype(np.uint64) + np.uint64(salt)) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(31)
    return (x % np.uint64(10_000)).astype(np.int64) < int(share * 10_000)


def make_client_split(client_id: pd.Series | np.ndarray, holdout_share: float = 0.4, salt: int = 0) -> np.ndarray:
    """Разбиение только по клиентам: train / valid / test (holdout делится пополам вторым хэшем).

    Нужно для AmEx, где официальные метки есть только для последней выписки, и она у всех
    клиентов в одном месяце: разделить срезы по времени не на чем.
    """
    hold = holdout_mask(client_id, holdout_share, salt)
    test = holdout_mask(client_id, 0.5, salt + 1)
    return np.where(~hold, "train", np.where(test, "test", "valid"))


