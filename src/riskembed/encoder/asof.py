"""Эмбеддинги выписок с выбором последнего доступного токена."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import torch

from riskembed.encoder.model import CausalSequenceEncoder
from riskembed.encoder.sequences import day_number
from riskembed.encoder.statements import StatementStore, collate_statements


def _batches(idx: np.ndarray, store, max_tokens: int) -> Iterator[list[int]]:
    """Батчи клиентов с ограничением B × T_max (длинные истории не взрывают память внимания)."""
    lengths = np.diff(store.day_offsets)[idx]
    order = idx[np.argsort(lengths, kind="stable")]
    batch, t_max = [], 0
    for i in order:
        n = int(store.day_offsets[i + 1] - store.day_offsets[i])
        if batch and max(t_max, n) * (len(batch) + 1) > max_tokens:
            yield batch
            batch, t_max = [], 0
        batch.append(int(i))
        t_max = max(t_max, n)
    if batch:
        yield batch


@torch.no_grad()
def embed_asof(
    model: CausalSequenceEncoder,
    store: StatementStore,
    spine: pd.DataFrame,
    device: str = "cpu",
    max_tokens: int = 32_000,
    prefix: str = "emb",
    collate_fn: Callable = collate_statements,
    inclusive: bool = False,
) -> pd.DataFrame:
    """spine[client_id, report_date] → spine + emb_000.. + emb_age_days (NaN, если истории до R нет).

    inclusive=False: токены с днём < R (события: срез на начало дня R).
    inclusive=True:  токены с днём ≤ R (выписки AmEx: выписка известна в свою дату).
    """
    model.eval()
    pos = store.index_of()
    spine = spine[["client_id", "report_date"]].reset_index(drop=True)
    store_idx = spine["client_id"].map(pos)
    rd = day_number(spine["report_date"])
    out = np.full((len(spine), model.cfg.emb_dim), np.nan, dtype=np.float32)
    age = np.full(len(spine), np.nan, dtype=np.float32)
    if store_idx.isna().all():
        rows_by_client = {}
    else:
        rows_by_client = pd.Series(np.arange(len(spine))).groupby(store_idx.to_numpy()).apply(np.asarray).to_dict()
    clients = np.array(sorted(int(k) for k in rows_by_client), dtype=np.int64)

    for batch_idx in _batches(clients, store, max_tokens):
        batch = collate_fn([store.client(i) for i in batch_idx], device)
        emb = model(batch).float().cpu().numpy()
        for b, i in enumerate(batch_idx):
            rows = rows_by_client[i]
            days = store.days[store.day_offsets[i] : store.day_offsets[i + 1]]
            k = np.searchsorted(days, rd[rows], side="right" if inclusive else "left") - 1  # последний токен < R (≤ R)
            ok = k >= 0
            out[rows[ok]] = emb[b, k[ok]]
            age[rows[ok]] = rd[rows[ok]] - days[k[ok]]

    cols = [f"{prefix}_{j:03d}" for j in range(out.shape[1])]
    res = pd.concat([spine, pd.DataFrame(out, columns=cols)], axis=1)
    res[f"{prefix}_age_days"] = age
    return res
