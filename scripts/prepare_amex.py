"""Подготовка AmEx: DQ-профиль, метки и сплит, витрина агрегатов, хранилище выписок.

    .venv/bin/python scripts/prepare_amex.py [--quick]
"""

import argparse
import json
import shutil
import time

import duckdb
import numpy as np

from riskembed.config import load_config, resolve
from riskembed.data.amex import (
    ID_COL,
    aggregate_sql,
    load_labels,
    raw_paths,
    register_amex,
)
from riskembed.encoder.statements import build_statement_store
from riskembed.evaluation.splits import holdout_mask, make_client_split


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    a = cfg.amex
    work = resolve(a.work_dir) / ("quick" if args.quick else "full")
    rep_dir = resolve(cfg.reports_dir) / ("amex_training_quick" if args.quick else "amex_training")
    for d in (work, rep_dir):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{a.duckdb_memory}'")
    con.execute(f"SET threads = {a.duckdb_threads}")
    con.execute(f"SET temp_directory = '{work / 'duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order = false")
    nums, cats = register_amex(con, a.raw_dir)

    # ---- метки, сплит, подвыборка --------------------------------------------
    labels = load_labels(a.raw_dir)
    labels["split"] = make_client_split(labels["cid"], a.holdout_client_share)
    if args.quick:
        labels = labels[holdout_mask(labels["cid"], 0.03, salt=99)].reset_index(drop=True)
    con.register("labels_df", labels[[ID_COL]])
    con.execute(f"CREATE OR REPLACE VIEW amex_sel AS SELECT a.* FROM amex a SEMI JOIN labels_df l USING ({ID_COL})")

    # ---- DQ-профиль (блокирующие проверки для этого источника) ---------------
    parquet, _ = raw_paths(a.raw_dir)
    prof = con.execute(f"""
        SELECT count(*) AS rows, count(DISTINCT {ID_COL}) AS customers,
               count(*) - count(DISTINCT ({ID_COL}, S_2)) AS duplicate_customer_date,
               count(*) FILTER (WHERE S_2 IS NULL) AS null_dates,
               min(S_2)::VARCHAR AS min_date, max(S_2)::VARCHAR AS max_date
        FROM amex_sel""").df().iloc[0].to_dict()
    null_share = con.execute(
        "SELECT " + ", ".join(f"avg(({c} IS NULL)::INT) AS {c}" for c in nums + cats) + " FROM amex_sel"
    ).df().iloc[0]
    last_month = con.execute(f"SELECT substr(max(S_2)::VARCHAR, 1, 7) m, count(*) FROM amex_sel GROUP BY {ID_COL}").df()
    checks = {
        "customers_match_labels": int(prof["customers"]) == len(labels),
        "no_duplicate_customer_date": int(prof["duplicate_customer_date"]) == 0,
        "no_null_dates": int(prof["null_dates"]) == 0,
        "no_fully_null_columns": bool((null_share < 1.0).all()),
    }
    profile = {
        **{k: (int(v) if isinstance(v, (np.integer, int)) else v) for k, v in prof.items()},
        "positive_rate_in_sample": float(labels["target"].mean()),
        "split_counts": labels["split"].value_counts().to_dict(),
        "last_statement_month_counts": last_month["m"].value_counts().to_dict(),
        "columns_numeric": len(nums), "columns_categorical": len(cats),
        "null_share_top10": null_share.sort_values(ascending=False).head(10).round(4).to_dict(),
        "columns_null_share_gt_0.9": int((null_share > 0.9).sum()),
        "checks": checks,
    }
    (rep_dir / "dq_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False, default=str))
    log(f"DQ: {checks}")
    if not all(checks.values()):
        raise SystemExit(f"DQ заблокировал подготовку: {checks}")

    # ---- витрина агрегатов: последний срез и сдвинутый ------------------------
    labels[[ID_COL, "cid", "target", "split"]].to_parquet(work / "labels.parquet", index=False)
    # агрегат из ~1100 колонок по 459 тыс. групп не помещается в память DuckDB целиком,
    # поэтому считаем чанками по хэшу клиента и пишем партициями (как в проде по бакетам).
    n_chunks = 1 if args.quick else a.agg_chunks
    for name, cutoff in (("agg_last", None), ("agg_shift", a.shift_cutoff)):
        d = work / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for k in range(n_chunks):
            bucket = f"AND hash({ID_COL}) % {n_chunks} = {k}" if n_chunks > 1 else ""
            sql = aggregate_sql(nums, cats, f"(SELECT * FROM amex_sel WHERE 1 = 1 {bucket})", cutoff)
            con.execute(f"COPY ({sql}) TO '{d / f'part-{k}.parquet'}' (FORMAT PARQUET)")
        log(f"витрина {name} (cutoff={cutoff}): {n_chunks} партиций")

    # ---- store выписок (квантили/категории только по train-клиентам) ----------
    train_ids = labels.loc[labels["split"] == "train", "cid"].to_numpy()
    # Quick mode reads only selected clients when quantizing the statement columns.
    source = parquet
    if args.quick:
        source = work / "selected_source.parquet"
        con.execute(f"COPY (SELECT r.* FROM read_parquet('{parquet}') r SEMI JOIN labels_df USING ({ID_COL})) TO '{source}' (FORMAT PARQUET)")
    store, vocab = build_statement_store(source, train_ids, seed=cfg.seed)
    if args.quick:
        source.unlink()
    if args.quick:  # оставить только выбранных клиентов
        from riskembed.encoder.statements import StatementStore

        keep = np.isin(store.client_ids, labels["cid"].to_numpy())
        lens = np.diff(store.day_offsets)
        rows = np.repeat(keep, lens)
        store = StatementStore(store.client_ids[keep], np.append(0, np.cumsum(lens[keep])), store.days[rows],
                               store.num_codes[rows], store.cat_idx[rows])
    shutil.rmtree(work / "statements", ignore_errors=True)
    store.save(work / "statements")
    vocab.save(work / "vocab.json")
    shutil.rmtree(work / "duckdb_tmp", ignore_errors=True)
    log(f"store: {len(store):,} клиентов, {len(store.days):,} выписок, "
        f"{store.num_codes.shape[1]} числовых + {store.cat_idx.shape[1]} категориальных признаков")


if __name__ == "__main__":
    main()
