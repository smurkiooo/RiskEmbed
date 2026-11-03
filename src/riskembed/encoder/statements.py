"""CSR-хранилище выписок AmEx: квантильные коды uint8 и токенизатор."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from riskembed.data.amex import CAT_COLS, DATE_COL, ID_COL, customer_int_id
from riskembed.encoder.sequences import EPOCH

MAX_LEVELS = 255


@dataclass
class TabularVocab:
    num_cols: list[str]
    cat_cols: list[str]
    num_edges: list[list[float]]  # границы квантильных бинов по колонке
    cat_maps: list[dict[int, int]]  # сырое значение -> индекс (0 = неизвестное)

    @property
    def num_levels(self) -> list[int]:
        return [len(e) + 1 for e in self.num_edges]

    @property
    def cat_sizes(self) -> list[int]:
        return [len(m) + 1 for m in self.cat_maps]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "num_cols": self.num_cols, "cat_cols": self.cat_cols, "num_edges": self.num_edges,
            "cat_maps": [{str(k): v for k, v in m.items()} for m in self.cat_maps],
        }))

    @classmethod
    def load(cls, path: str | Path) -> "TabularVocab":
        raw = json.loads(Path(path).read_text())
        return cls(raw["num_cols"], raw["cat_cols"], raw["num_edges"],
                   [{int(float(k)): v for k, v in m.items()} for m in raw["cat_maps"]])


def quantile_edges(x: np.ndarray, max_levels: int = MAX_LEVELS) -> list[float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return []
    uniq = np.unique(x)
    if len(uniq) < max_levels:  # низкая кардинальность: границы посередине между значениями
        return ((uniq[1:] + uniq[:-1]) / 2).astype(float).tolist()
    q = np.quantile(x, np.linspace(0, 1, max_levels)[1:-1])
    return np.unique(q).astype(float).tolist()


def encode_numeric(x: np.ndarray, edges: list[float]) -> np.ndarray:
    codes = np.searchsorted(np.asarray(edges, dtype=np.float64), x, side="right") + 1
    codes[np.isnan(x)] = 0
    return codes.astype(np.uint8)


@dataclass
class StatementStore:
    """Компактное хранилище последовательностей выписок по клиентам."""

    client_ids: np.ndarray  # [N] int64
    day_offsets: np.ndarray  # [N+1]
    days: np.ndarray  # [S] int32 — дата выписки (дни от эпохи)
    num_codes: np.ndarray  # [S, n_num] uint8
    cat_idx: np.ndarray  # [S, n_cat] int16

    def __len__(self) -> int:
        return len(self.client_ids)

    def client(self, i: int) -> dict[str, np.ndarray]:
        a, b = self.day_offsets[i], self.day_offsets[i + 1]
        return {"client_id": int(self.client_ids[i]), "days": self.days[a:b],
                "num_codes": self.num_codes[a:b], "cat_idx": self.cat_idx[a:b]}

    def index_of(self) -> dict[int, int]:
        return {int(c): i for i, c in enumerate(self.client_ids)}

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in self.__dict__.items():
            np.save(d / f"{k}.npy", v)

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = True) -> "StatementStore":
        d = Path(directory)
        mode = "r" if mmap else None
        return cls(**{k: np.load(d / f"{k}.npy", mmap_mode=mode) for k in cls.__dataclass_fields__})


def crop_statements(c: dict[str, np.ndarray], a: int, b: int) -> dict[str, np.ndarray]:
    return {"client_id": c["client_id"], "days": c["days"][a:b], "num_codes": c["num_codes"][a:b], "cat_idx": c["cat_idx"][a:b]}


def _read_column(pf: pq.ParquetFile, name: str) -> np.ndarray:
    col = pf.read(columns=[name]).column(0)
    arr = col.to_numpy(zero_copy_only=False).astype(np.float32)
    if pa.types.is_integer(col.type):
        arr[arr == -1] = np.nan  # raddar: -1 = пропуск в целочисленных колонках
    return arr


def build_statement_store(
    parquet_path: str | Path, train_client_ids: set[int] | np.ndarray, sample_rows: int = 1_000_000, seed: int = 0,
    columns: list[str] | None = None, vocab: TabularVocab | None = None,
) -> tuple[StatementStore, TabularVocab]:
    """Строит store и словарь. Квантили и категории — только по выпискам train-клиентов."""
    pf = pq.ParquetFile(parquet_path)
    all_cols = [f.name for f in pf.schema_arrow]
    feat_cols = vocab.num_cols + vocab.cat_cols if vocab else (columns or [c for c in all_cols if c not in (ID_COL, DATE_COL)])
    missing = set(feat_cols) - set(all_cols)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    cat_cols = [c for c in CAT_COLS if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    if vocab:
        num_cols, cat_cols = vocab.num_cols, vocab.cat_cols

    ids_tbl = pf.read(columns=[ID_COL, DATE_COL])
    raw_ids = ids_tbl.column(0).to_numpy(zero_copy_only=False)
    codes_uniq, inv = np.unique(raw_ids, return_inverse=True)
    cid = customer_int_id(codes_uniq)[inv]
    days = (pd.to_datetime(ids_tbl.column(1).to_numpy(zero_copy_only=False), format="%Y-%m-%d").to_numpy("datetime64[D]") - EPOCH).astype(np.int32)
    del ids_tbl, raw_ids

    order = np.lexsort((days, cid))  # по клиенту, затем по дате
    cid, days = cid[order], days[order]
    client_ids, starts = np.unique(cid, return_index=True)
    day_offsets = np.append(starts, len(cid)).astype(np.int64)

    is_train = np.isin(cid, np.fromiter(train_client_ids, dtype=np.int64) if isinstance(train_client_ids, set) else train_client_ids)
    rng = np.random.default_rng(seed)
    train_rows = np.nonzero(is_train)[0]
    if not vocab and not len(train_rows):
        raise ValueError("No training customers available to fit the tokenizer")
    sample = np.sort(rng.choice(train_rows, size=min(sample_rows, len(train_rows)), replace=False))

    num_codes = np.zeros((len(cid), len(num_cols)), dtype=np.uint8)
    edges_all = []
    for j, c in enumerate(num_cols):
        x = _read_column(pf, c)[order]
        edges = vocab.num_edges[j] if vocab else quantile_edges(x[sample])
        edges_all.append(edges)
        num_codes[:, j] = encode_numeric(x, edges)

    cat_idx = np.zeros((len(cid), len(cat_cols)), dtype=np.int16)
    cat_maps = []
    for j, c in enumerate(cat_cols):
        x = _read_column(pf, c)[order]
        x = np.where(np.isnan(x), -1, x).astype(np.int64)  # пропуск в категории — отдельная категория -1
        values = np.unique(x[is_train])
        mapping = vocab.cat_maps[j] if vocab else {int(v): i + 1 for i, v in enumerate(values)}
        cat_maps.append(mapping)
        cat_idx[:, j] = pd.Series(x).map(mapping).fillna(0).to_numpy(np.int16)

    store = StatementStore(client_ids, day_offsets, days, num_codes, cat_idx)
    return store, TabularVocab(num_cols, cat_cols, edges_all, cat_maps)


class StatementTokenizer(nn.Module):
    """Выписка → токен: Linear([ранги, флаги пропуска]) + сумма эмбеддингов категорий → LayerNorm."""

    def __init__(self, vocab: TabularVocab, d: int):
        super().__init__()
        self.register_buffer("levels", torch.tensor(vocab.num_levels, dtype=torch.float32).clamp_min(1))
        self.num_proj = nn.Linear(2 * len(vocab.num_cols), d)
        sizes = vocab.cat_sizes
        self.register_buffer("cat_offsets", torch.tensor(np.concatenate([[0], np.cumsum(sizes)[:-1]]), dtype=torch.long))
        self.cat_emb = nn.Embedding(int(sum(sizes)) if sizes else 1, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch: dict[str, torch.Tensor], B: int = 0, T: int = 0) -> torch.Tensor:  # noqa: ARG002
        # B, T не нужны (форма уже в тензорах), сигнатура соответствует интерфейсу токенизатора
        codes = batch["num_codes"].float()  # [B,T,n]
        missing = (codes == 0).float()
        rank = codes / self.levels
        x = self.num_proj(torch.cat([rank, missing], dim=-1))
        if batch["cat_idx"].shape[-1]:
            x = x + self.cat_emb(batch["cat_idx"].long() + self.cat_offsets).sum(dim=-2)
        return self.norm(x)


def collate_statements(clients: list[dict[str, np.ndarray]], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    B = len(clients)
    T = max(len(c["days"]) for c in clients)
    n_num = clients[0]["num_codes"].shape[1]
    n_cat = clients[0]["cat_idx"].shape[1]
    days = np.zeros((B, T), dtype=np.int64)
    pad = np.ones((B, T), dtype=bool)
    num = np.zeros((B, T, n_num), dtype=np.uint8)
    cat = np.zeros((B, T, n_cat), dtype=np.int16)
    for b, c in enumerate(clients):
        n = len(c["days"])
        days[b, :n] = c["days"]
        days[b, n:] = c["days"][-1] if n else 0
        pad[b, :n] = False
        num[b, :n] = c["num_codes"]
        cat[b, :n] = c["cat_idx"]
    return {
        "client_id": torch.tensor([c["client_id"] for c in clients]),
        "days": torch.from_numpy(days).to(device),
        "pad": torch.from_numpy(pad).to(device),
        "num_codes": torch.from_numpy(num).to(device),
        "cat_idx": torch.from_numpy(cat).to(device),
        "lengths": torch.tensor([len(c["days"]) for c in clients]),
    }
