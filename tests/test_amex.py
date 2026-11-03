"""Адаптер AmEx на миниатюрном parquet того же формата, что сборка raddar."""

import hashlib

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from riskembed.data.amex import (
    aggregate_feature_names,
    aggregate_sql,
    amex_metric,
    customer_int_id,
    register_amex,
)
from riskembed.encoder.asof import embed_asof
from riskembed.encoder.model import CausalSequenceEncoder, EncoderConfig
from riskembed.encoder.statements import (
    StatementStore,
    StatementTokenizer,
    TabularVocab,
    build_statement_store,
    collate_statements,
    encode_numeric,
    quantile_edges,
)
from riskembed.evaluation.splits import holdout_mask, make_client_split

N_CUST = 60


def _cid(i: int) -> str:
    return hashlib.sha256(str(i).encode()).hexdigest()


@pytest.fixture(scope="module")
def amex_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_CUST):
        n = int(rng.integers(1, 14))
        dates = pd.date_range("2017-03-05", periods=13, freq="MS")[-n:] + pd.Timedelta(days=int(rng.integers(0, 20)))
        for d in dates:
            rows.append({
                "customer_ID": _cid(i), "S_2": d.strftime("%Y-%m-%d"),
                "P_2": np.nan if rng.random() < 0.1 else float(rng.normal()),
                "D_39": int(rng.integers(-1, 5)),  # -1 = пропуск
                "B_30": int(rng.integers(0, 3)),
                "D_63": int(rng.integers(0, 4)),
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=1)  # порядок строк перемешан
    table = pa.Table.from_pandas(df, preserve_index=False).cast(pa.schema([
        ("customer_ID", pa.string()), ("S_2", pa.string()), ("P_2", pa.float32()),
        ("D_39", pa.int16()), ("B_30", pa.int8()), ("D_63", pa.int8()),
    ]))
    d = tmp_path_factory.mktemp("amex")
    pq.write_table(table, d / "train.parquet")
    labels = pd.DataFrame({"customer_ID": [_cid(i) for i in range(N_CUST)], "target": rng.integers(0, 2, N_CUST)})
    labels.to_csv(d / "train_labels.csv", index=False)
    return d, df


def test_register_converts_minus_one_to_null(amex_dir):
    d, df = amex_dir
    con = duckdb.connect()
    nums, cats = register_amex(con, d)
    assert nums == ["P_2", "D_39"] and cats == ["B_30", "D_63"]
    n_null = con.execute("SELECT count(*) FILTER (WHERE D_39 IS NULL), count(*) FILTER (WHERE D_39 = -1) FROM amex").fetchone()
    assert n_null == ((df.D_39 == -1).sum(), 0)


def test_aggregates_match_pandas(amex_dir):
    d, df = amex_dir
    con = duckdb.connect()
    nums, cats = register_amex(con, d)
    agg = con.execute(aggregate_sql(nums, cats)).df().set_index("customer_ID")
    assert list(agg.columns) == aggregate_feature_names(nums, cats)
    x = df.assign(D_39=df.D_39.replace(-1, np.nan)).sort_values(["customer_ID", "S_2"])
    for cid, g in x.groupby("customer_ID"):
        r = agg.loc[cid]
        assert r["n_statements"] == len(g)
        # last — значение ПОСЛЕДНЕЙ выписки, даже если это пропуск
        np.testing.assert_equal(np.isnan(r["P_2__last"]), np.isnan(g.P_2.iloc[-1]))
        if not np.isnan(g.P_2.iloc[-1]):
            assert r["P_2__last"] == pytest.approx(g.P_2.iloc[-1], rel=1e-5)
        if g.D_39.notna().any():
            assert r["D_39__mean"] == pytest.approx(g.D_39.mean(), rel=1e-5)
        assert r["B_30__nunique"] == g.B_30.nunique()


def test_aggregates_cutoff_excludes_later_statements(amex_dir):
    d, df = amex_dir
    con = duckdb.connect()
    nums, cats = register_amex(con, d)
    cut = con.execute(aggregate_sql(nums, cats, cutoff="2017-12-01")).df().set_index("customer_ID")
    x = df[df.S_2 < "2017-12-01"]
    assert set(cut.index) == set(x.customer_ID)
    counts = x.groupby("customer_ID").size()
    assert (cut.loc[counts.index, "n_statements"] == counts).all()


def test_statement_store_order_codes_and_train_only_vocab(amex_dir, tmp_path):
    d, df = amex_dir
    ids = customer_int_id(pd.Series([_cid(i) for i in range(N_CUST)]))
    train_ids = set(ids[: N_CUST // 2].tolist())
    store, vocab = build_statement_store(d / "train.parquet", train_ids, sample_rows=10_000)
    assert len(store) == N_CUST and store.day_offsets[-1] == len(df)
    for i in range(len(store)):  # даты внутри клиента строго возрастают
        days = store.days[store.day_offsets[i] : store.day_offsets[i + 1]]
        assert (np.diff(days) > 0).all()
    j = vocab.num_cols.index("D_39")
    raw = df.assign(cid=customer_int_id(df.customer_ID)).sort_values(["cid", "S_2"])
    assert ((store.num_codes[:, j] == 0) == (raw.D_39 == -1).to_numpy()).all()  # -1 → код пропуска
    assert store.num_codes.max() <= max(vocab.num_levels)
    store.save(tmp_path / "s")
    vocab.save(tmp_path / "v.json")
    s2, v2 = StatementStore.load(tmp_path / "s"), TabularVocab.load(tmp_path / "v.json")
    assert (s2.num_codes == store.num_codes).all() and v2.cat_maps == vocab.cat_maps


def test_quantile_encoding_edges():
    x = np.array([np.nan, 0.0, 1.0, 1.0, 2.0])
    e = quantile_edges(x)
    assert e == [0.5, 1.5]
    assert encode_numeric(np.array([np.nan, -5, 0.7, 99.0]), e).tolist() == [0, 1, 2, 3]
    big = np.random.default_rng(0).normal(size=100_000)
    codes = encode_numeric(big, quantile_edges(big))
    assert codes.min() == 1 and codes.max() == 254 and np.bincount(codes)[1:].min() > 300


def test_statement_encoder_asof_leakage(amex_dir):
    d, _ = amex_dir
    ids = customer_int_id(pd.Series([_cid(i) for i in range(N_CUST)]))
    store, vocab = build_statement_store(d / "train.parquet", set(ids.tolist()), sample_rows=10_000)
    torch.manual_seed(0)
    model = CausalSequenceEncoder(StatementTokenizer(vocab, 16), EncoderConfig(d_model=16, n_heads=2, emb_dim=8, dropout=0.0)).eval()
    spine = pd.DataFrame({"client_id": store.client_ids, "report_date": pd.Timestamp("2017-12-01")})
    base = embed_asof(model, store, spine, collate_fn=collate_statements, inclusive=True)

    # портим все выписки ПОСЛЕ даты среза
    days = store.days
    cut = (pd.Timestamp("2017-12-01") - pd.Timestamp("1970-01-01")).days
    tampered = StatementStore(store.client_ids, store.day_offsets, days, store.num_codes.copy(), store.cat_idx.copy())
    tampered.num_codes[days > cut] = 7
    tampered.cat_idx[days > cut] = 1
    after = embed_asof(model, tampered, spine, collate_fn=collate_statements, inclusive=True)
    cols = [c for c in base if c.startswith("emb_")]
    np.testing.assert_allclose(base[cols].to_numpy(), after[cols].to_numpy(), atol=1e-5)

    # выписка ровно в дату среза входит при inclusive=True и не входит при inclusive=False
    i = 0
    last_day = int(store.days[store.day_offsets[i + 1] - 1])
    sp = pd.DataFrame({"client_id": [store.client_ids[i]], "report_date": [pd.Timestamp("1970-01-01") + pd.Timedelta(days=last_day)]})
    inc = embed_asof(model, store, sp, collate_fn=collate_statements, inclusive=True)
    exc = embed_asof(model, store, sp, collate_fn=collate_statements, inclusive=False)
    assert inc["emb_age_days"].iloc[0] == 0
    assert np.isnan(exc["emb_age_days"].iloc[0]) or exc["emb_age_days"].iloc[0] > 0


def _official_amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Референсная реализация с Kaggle (переписана компактно, логика без изменений)."""
    df = pd.DataFrame({"target": y_true, "prediction": y_pred}).sort_values("prediction", ascending=False)
    df["weight"] = np.where(df.target == 0, 20, 1)
    cutoff = int(0.04 * df.weight.sum())
    d = (df.loc[df.weight.cumsum() <= cutoff, "target"] == 1).sum() / (df.target == 1).sum()

    def wgini(frame):
        frame = frame.copy()
        frame["random"] = (frame.weight / frame.weight.sum()).cumsum()
        total_pos = (frame.target * frame.weight).sum()
        frame["lorentz"] = (frame.target * frame.weight).cumsum() / total_pos
        return ((frame.lorentz - frame.random) * frame.weight).sum()

    g = wgini(df) / wgini(df.assign(prediction=df.target).sort_values("prediction", ascending=False))
    return 0.5 * (g + d)


def test_amex_metric_matches_official():
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.26).astype(int)
    s = rng.normal(size=20_000) + 1.2 * y
    assert amex_metric(y, s)["amex_M"] == pytest.approx(_official_amex_metric(y, s), abs=1e-12)


def test_client_split_proportions_and_independent_hashes():
    ids = np.random.default_rng(0).integers(0, 2**60, 200_000)
    split = make_client_split(ids, 0.4)
    shares = pd.Series(split).value_counts(normalize=True)
    assert abs(shares["train"] - 0.6) < 0.01 and abs(shares["valid"] - 0.2) < 0.01 and abs(shares["test"] - 0.2) < 0.01
    # второй хэш (salt+1) не должен коррелировать с первым — иначе valid/test смещены
    assert abs(np.corrcoef(holdout_mask(ids, 0.4), holdout_mask(ids, 0.5, 1))[0, 1]) < 0.01


def test_customer_int_id_deterministic_and_collision_free():
    ids = [_cid(i) for i in range(50_000)]
    a = customer_int_id(pd.Series(ids))
    assert (a == customer_int_id(np.array(ids))).all() and len(np.unique(a)) == len(ids) and (a >= 0).all()


def test_source_audit_accepts_fixture_and_rejects_missing_labels(amex_dir):
    from riskembed.data.validation import audit_amex

    directory, _ = amex_dir
    report = audit_amex(directory, require_full=False)
    assert report["passed"] and report["coverage"] == {"unlabeled": 0, "missing_history": 0}
    assert not audit_amex(directory, require_full=True)["passed"]  # a miniature fixture is not the full source
    # Missing labels must be detected even when raw statements themselves look valid.
    path = directory / "train_labels.csv"
    original = path.read_bytes()
    try:
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        broken = audit_amex(directory, require_full=False)
        assert not broken["passed"] and not broken["checks"]["exact_label_coverage"]
    finally:
        path.write_bytes(original)


def test_pretraining_reads_only_selected_customers(amex_dir):
    from riskembed.encoder.pretrain import PretrainConfig, pretrain
    from riskembed.encoder.statements import crop_statements

    directory, _ = amex_dir
    ids = customer_int_id(pd.Series([_cid(i) for i in range(N_CUST)]))
    store, vocab = build_statement_store(directory / "train.parquet", ids[:30], sample_rows=10_000)
    indices = np.nonzero(np.isin(store.client_ids, ids[:30]) & (np.diff(store.day_offsets) >= 2))[0]
    accessed = []
    original_client = store.client

    def record_client(i):
        accessed.append(i)
        return original_client(i)

    store.client = record_client
    model = CausalSequenceEncoder(StatementTokenizer(vocab, 16),
                                  EncoderConfig(d_model=16, n_heads=2, n_layers=1, emb_dim=8, dropout=0.0))
    history = pretrain(model, store, collate_statements, crop_statements,
                       PretrainConfig(epochs=2, batch_size=4, max_steps_per_epoch=2, seed=7),
                       client_indices=indices, log=lambda _: None)
    assert accessed and set(accessed).issubset(set(indices))
    assert all(np.isfinite(epoch["loss"]) for epoch in history)


def test_frozen_tokenizer_reproduces_training_store(amex_dir):
    directory, _ = amex_dir
    ids = customer_int_id(pd.Series([_cid(i) for i in range(N_CUST)]))
    original, vocab = build_statement_store(directory / "train.parquet", ids[:30])
    frozen, _ = build_statement_store(directory / "train.parquet", np.empty(0, dtype=np.int64), vocab=vocab)
    np.testing.assert_array_equal(original.num_codes, frozen.num_codes)
    np.testing.assert_array_equal(original.cat_idx, frozen.cat_idx)
