"""Обучение на AmEx: предобучение трёх вариантов энкодера и таблица абляции.

    .venv/bin/python scripts/train_amex.py [--quick] [--force]

Строки абляции: агрегаты; + CoLES+VICReg; + CoLES без VICReg; + позиции вместо времени;
только эмбеддинг. Артефакты энкодеров кэшируются в artifacts/.
"""

import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from riskembed.config import load_config, resolve
from riskembed.data.amex import ID_COL, amex_metric
from riskembed.evaluation.metrics import gini, paired_gini_delta_bootstrap
from riskembed.features.psi import psi_numeric
from riskembed.provenance import experiment_signature

# lightgbm и torch нельзя загружать в одном процессе, поэтому импорты обеих — ленивые

LAST_DATE = pd.Timestamp("2018-03-31")


def agg_files(path) -> list:
    """Витрина лежит директорией партиций (или одним файлом — совместимость со старыми прогонами)."""
    return sorted(path.glob("*.parquet")) if path.is_dir() else [path.with_suffix(".parquet")]


def pretrain_stage(cfg, a, args, labels, art, log) -> None:
    """Этап 1 (torch): CoLES+VICReg для каждого варианта + эмбеддинги на последнюю дату и на сдвинутый срез."""
    import torch

    from riskembed.encoder.asof import embed_asof
    from riskembed.encoder.model import CausalSequenceEncoder, EncoderConfig
    from riskembed.encoder.pretrain import (
        PretrainConfig,
        embedding_diagnostics,
        pretrain,
    )
    from riskembed.encoder.statements import (
        StatementStore,
        StatementTokenizer,
        TabularVocab,
        collate_statements,
        crop_statements,
    )

    torch.set_num_threads(6)
    work = resolve(a.work_dir) / ("quick" if args.quick else "full")
    store = StatementStore.load(work / "statements")
    vocab = TabularVocab.load(work / "vocab.json")
    shift_date = pd.Timestamp(a.shift_cutoff) - pd.Timedelta(days=1)  # inclusive → выписки строго до cutoff
    spine_last = pd.DataFrame({"client_id": labels["cid"], "report_date": LAST_DATE})
    spine_shift = pd.DataFrame({"client_id": labels["cid"], "report_date": shift_date})
    # предобучение только на train-клиентах: valid и test — «новые» клиенты, энкодер их не видел
    pos = store.index_of()
    train_idx = np.array([pos[c] for c in labels.loc[labels["split"] == "train", "cid"] if c in pos], dtype=np.int64)
    pre = dict(a.pretrain)
    if args.quick:
        pre |= {"epochs": 1, "batch_size": 128}
    emb_files, pretrain_report = {}, {}
    for name, var in a.variants.items():
        vdir = art / name
        emb_files[name] = vdir / "embeddings.parquet"
        signature = experiment_signature(
            {"seed": cfg.seed, "encoder": dict(a.encoder), "pretrain": pre, "variant": dict(var)},
            [work / "labels.parquet", work / "vocab.json", *sorted((work / "statements").glob("*.npy"))])
        cached_report = vdir / "report.json"
        cache_matches = (cached_report.exists() and
                         json.loads(cached_report.read_text()).get("signature") == signature)
        if emb_files[name].exists() and (vdir / "encoder.pt").exists() and cache_matches and not args.force:
            pretrain_report[name] = json.loads((vdir / "report.json").read_text())
            log(f"{name}: артефакты найдены, пропускаю предобучение")
            continue
        vdir.mkdir(parents=True, exist_ok=True)
        enc_cfg = EncoderConfig(**dict(a.encoder), time_encoding=var["time_encoding"])
        torch.manual_seed(cfg.seed)
        model = CausalSequenceEncoder(StatementTokenizer(vocab, enc_cfg.d_model), enc_cfg)
        pcfg = PretrainConfig(**pre, vicreg_var=var["vicreg_var"], vicreg_cov=var["vicreg_cov"], seed=cfg.seed)
        log(f"{name}: предобучение {pcfg.to_dict()}")
        hist = pretrain(model, store, collate_statements, crop_statements, pcfg, client_indices=train_idx, log=log)
        torch.save(model.state_dict(), vdir / "encoder.pt")

        e_last = embed_asof(model, store, spine_last, collate_fn=collate_statements, inclusive=True, max_tokens=64_000)
        e_shift = embed_asof(model, store, spine_shift, collate_fn=collate_statements, inclusive=True, max_tokens=64_000)
        cols = [c for c in e_last if c.startswith("emb_") and c != "emb_age_days"]
        emb = pd.DataFrame({"cid": labels["cid"].to_numpy()})
        for c in cols:
            emb[f"{name}__{c}"] = e_last[c].to_numpy(np.float32)
            emb[f"{name}__{c}__shift"] = e_shift[c].to_numpy(np.float32)
        emb.to_parquet(emb_files[name], index=False)
        train_emb = e_last.loc[(labels["split"] == "train").to_numpy(), cols].to_numpy()
        pretrain_report[name] = {"signature": signature, "train_clients": len(train_idx), "encoder": enc_cfg.to_dict(), "pretrain": pcfg.to_dict(), "history": hist,
                                 "diagnostics_train": embedding_diagnostics(train_emb),
                                 "shift_coverage": float(e_shift[cols[0]].notna().mean())}
        (vdir / "report.json").write_text(json.dumps(pretrain_report[name], indent=2, default=float))
        log(f"{name}: диагностика {pretrain_report[name]['diagnostics_train']}")
        del model, e_last, e_shift, emb
        gc.collect()


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", choices=["all", "pretrain", "boost"], default="all")
    args = ap.parse_args()
    if args.stage == "all":
        # ЭТАПЫ ИДУТ РАЗНЫМИ ПРОЦЕССАМИ: LightGBM в несколько потоков зависает в процессе,
        # где уже работал torch (два OpenMP-runtime в одном процессе, macOS). См. DEVLOG, аудит A2.
        flags = [f for f, on in (("--quick", args.quick), ("--force", args.force)) if on]
        for stage in ("pretrain", "boost"):
            subprocess.run([sys.executable, __file__, "--stage", stage, *flags], check=True)
        return
    a = cfg.amex
    tag = "quick" if args.quick else "full"
    work = resolve(a.work_dir) / tag
    art = resolve("artifacts") / f"amex_{tag}"
    out = resolve(cfg.reports_dir) / ("amex_training_quick" if args.quick else "amex_training")
    art.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{args.stage:8s} {time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731

    labels = pd.read_parquet(work / "labels.parquet")
    emb_files = {name: art / name / "embeddings.parquet" for name in a.variants}
    if args.stage == "pretrain":
        pretrain_stage(cfg, a, args, labels, art, log)
        return
    from riskembed.models.baseline import importance_frame, score_report, train_lgb

    pretrain_report = {name: json.loads((art / name / "report.json").read_text()) for name in a.variants}

    # ---- 2. матрица признаков: агрегаты + слоты эмбеддингов --------------------
    agg_parts = agg_files(work / "agg_last")
    agg_names = [n for n in pq.ParquetFile(agg_parts[0]).schema_arrow.names if n != ID_COL]
    emb_names = {name: [f"{name}__emb_{j:03d}" for j in range(a.encoder.emb_dim)] for name in a.variants}
    all_names = agg_names + [c for cols in emb_names.values() for c in cols]
    col_of = {n: j for j, n in enumerate(all_names)}
    n = len(labels)
    row_of = pd.Series(np.arange(n), index=labels[ID_COL].to_numpy())

    def load_agg(parts: list, X: np.ndarray, row_pos: np.ndarray | None = None) -> None:
        """Колонки витрины (одна или несколько партиций) → столбцы 0..len(agg_names) матрицы X."""
        for part in parts:
            pf = pq.ParquetFile(part)
            ids = pf.read(columns=[ID_COL]).column(0).to_numpy(zero_copy_only=False)
            pos = row_of.reindex(ids).to_numpy()
            ok = ~np.isnan(pos)
            target = pos[ok].astype(np.int64) if row_pos is None else row_pos[pos[ok].astype(np.int64)]
            keep = target >= 0
            for j, name in enumerate(agg_names):
                col = pf.read(columns=[name]).column(0).to_numpy(zero_copy_only=False).astype(np.float32)[ok]
                X[target[keep], j] = col[keep]

    X = np.full((n, len(all_names)), np.nan, dtype=np.float32)
    load_agg(agg_parts, X)
    for name in a.variants:
        e = pd.read_parquet(emb_files[name], columns=["cid"] + emb_names[name])
        assert (e["cid"].to_numpy() == labels["cid"].to_numpy()).all()
        X[:, [col_of[c] for c in emb_names[name]]] = e[emb_names[name]].to_numpy(np.float32)
    log(f"матрица признаков {X.shape}, {X.nbytes / 2**30:.2f} ГБ")

    split = labels["split"].to_numpy()
    y = labels["target"].to_numpy().astype(np.float32)
    idx = {s: np.nonzero(split == s)[0] for s in ("train", "valid", "test")}
    test_idx = idx["test"]

    # сдвинутый срез — только для test-клиентов (экономия памяти)
    test_pos = np.full(n, -1, dtype=np.int64)
    test_pos[test_idx] = np.arange(len(test_idx))
    Xs = np.full((len(test_idx), len(all_names)), np.nan, dtype=np.float32)
    load_agg(agg_files(work / "agg_shift"), Xs, row_pos=test_pos)
    for name in a.variants:
        shift_cols = [f"{c}__shift" for c in emb_names[name]]
        e = pd.read_parquet(emb_files[name], columns=shift_cols)
        Xs[:, [col_of[c] for c in emb_names[name]]] = e.to_numpy(np.float32)[test_idx]
    has_shift = ~np.isnan(Xs[:, col_of["n_statements"]])
    log(f"сдвинутый срез: {has_shift.mean():.1%} test-клиентов имеют выписки до {a.shift_cutoff}")

    # ---- 3. строки абляции -------------------------------------------------------
    rows_def = [
        ("01_agg_baseline", agg_names),
        ("02_agg+coles_vicreg", agg_names + emb_names["coles_vicreg"]),
        ("03_agg+coles_plain", agg_names + emb_names["coles_plain"]),
        ("04_agg+positions_vicreg", agg_names + emb_names["positions_vicreg"]),
        ("05_coles_vicreg_only", emb_names["coles_vicreg"]),
    ]
    b = a.boosting
    results, test_scores, shift_scores = [], {}, {}

    def predict(model, M, rows, cols):
        res = np.empty(len(rows), dtype=np.float64)
        for s in range(0, len(rows), 50_000):
            r = rows[s : s + 50_000]
            res[s : s + 50_000] = model.predict(M[np.ix_(r, cols)], num_iteration=model.best_iteration)
        return res

    for name, feats in rows_def:
        t1 = time.time()
        cols = np.array([col_of[f] for f in feats])
        model = train_lgb(X[np.ix_(idx["train"], cols)], y[idx["train"]], X[np.ix_(idx["valid"], cols)], y[idx["valid"]],
                          dict(b.params), feats, b.num_boost_round, b.early_stopping_rounds, cfg.seed)
        scored = pd.concat([
            pd.DataFrame({"client_id": labels["cid"].to_numpy()[idx[s]], "split": s, "target": y[idx[s]],
                          "score": predict(model, X, idx[s], cols)}) for s in ("train", "valid", "test")
        ], ignore_index=True)
        metrics, _ = score_report(scored, name, "test", "target", a.bootstrap_reps, cfg.seed, period_col=None, business_metric=False)
        model_dir = art / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_dir / "model.txt"), num_iteration=model.best_iteration)
        ts = scored.loc[scored.split == "test"]
        ts.to_parquet(out / f"{name}_test.parquet", index=False)
        test_scores[name] = ts["score"].to_numpy()
        metrics |= {k: float(v) for k, v in amex_metric(ts["target"].to_numpy(), test_scores[name]).items()}

        sh = predict(model, Xs, np.nonzero(has_shift)[0], cols)
        shift_scores[name] = sh
        y_shift = y[test_idx][has_shift]
        metrics |= {
            "gini_test_shift": gini(y_shift, sh),
            "gini_test_same_clients_last": gini(y_shift, test_scores[name][has_shift]),
            "psi_score_test_vs_shift": psi_numeric(test_scores[name][has_shift], sh),
            "n_features": len(feats), "best_iteration": model.best_iteration, "seconds": round(time.time() - t1, 1),
        }
        imp = importance_frame(model)
        metrics["embedding_gain_share"] = float(imp.loc[imp.feature.str.contains("__emb_"), "gain_share"].sum())
        imp.head(30).to_csv(out / f"importance_{name}.csv", index=False)
        results.append(metrics)
        log(f"{name}: gini test {metrics['gini_test']:.4f} [{metrics['gini_test_ci_low']:.4f}; {metrics['gini_test_ci_high']:.4f}] "
            f"shift {metrics['gini_test_shift']:.4f} iters {model.best_iteration} ({metrics['seconds']}s)")
        del model
        gc.collect()

    # парные приросты к бейзлайну на одних и тех же бутстрап-выборках
    y_test, cid_test = y[test_idx], labels["cid"].to_numpy()[test_idx]
    base = "01_agg_baseline"
    for r in results[1:]:
        d = paired_gini_delta_bootstrap(y_test, test_scores[base], test_scores[r["name"]], cid_test, a.bootstrap_reps, cfg.seed)
        ds = paired_gini_delta_bootstrap(y_test[has_shift], shift_scores[base], shift_scores[r["name"]], cid_test[has_shift],
                                         a.bootstrap_reps, cfg.seed)
        r |= {"delta_gini_test": d["delta"], "delta_ci_low": d["ci_low"], "delta_ci_high": d["ci_high"], "delta_p_le_0": d["p_le_0"],
              "delta_gini_shift": ds["delta"], "delta_shift_ci_low": ds["ci_low"], "delta_shift_ci_high": ds["ci_high"]}

    res = pd.DataFrame(results)
    res.to_csv(out / "ablation.csv", index=False)
    (out / "pretrain_report.json").write_text(json.dumps(pretrain_report, indent=2, default=float))
    write_report(out, res, pretrain_report, labels, has_shift, a, args.quick)
    log(f"готово → {out / 'README.md'}")


def write_report(out, res, pretrain_report, labels, has_shift, a, quick) -> None:
    def fmt_delta(r, key, lo, hi):
        return "—" if pd.isna(r.get(key)) else f"{r[key]:+.4f} [{r[lo]:+.4f}; {r[hi]:+.4f}]"

    lines = [
        f"# AmEx: CoLES + VICReg, таблица абляции{' (QUICK: подвыборка, не для выводов)' if quick else ''}", "",
        "Сплит только по клиентам (хэш): " + ", ".join(f"{k} {v:,}" for k, v in labels["split"].value_counts().items()) +
        f". Out-of-time по официальным меткам невозможен (последняя выписка у всех в 2018-03), поэтому устойчивость "
        f"проверяется сдвинутым срезом: признаки по выпискам до {a.shift_cutoff}, те же метки "
        f"({has_shift.mean():.1%} test-клиентов имеют такие выписки).", "",
        "## Абляция (test)", "",
        "| строка | признаков | Gini test [95% CI] | ΔGini к бейзлайну [95% CI] | AmEx M | Gini сдвиг | ΔGini сдвиг [95% CI] | PSI скора last→сдвиг | доля gain эмбеддинга | деревьев |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {r['name']} | {r['n_features']} | {r['gini_test']:.4f} [{r['gini_test_ci_low']:.4f}; {r['gini_test_ci_high']:.4f}] | "
            f"{fmt_delta(r, 'delta_gini_test', 'delta_ci_low', 'delta_ci_high')} | {r['amex_M']:.4f} | {r['gini_test_shift']:.4f} | "
            f"{fmt_delta(r, 'delta_gini_shift', 'delta_shift_ci_low', 'delta_shift_ci_high')} | {r['psi_score_test_vs_shift']:.4f} | "
            f"{r['embedding_gain_share']:.3f} | {r['best_iteration']} |"
        )
    lines += ["", "## Предобучение энкодеров", "",
              "| вариант | эпох | лосс (последняя эпоха) | NCE | std компонент (ср./мин.) | доля схлопнутых | ср. |corr| | эфф. ранг / dim | сек/эпоху |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, rep in pretrain_report.items():
        h, dg = rep["history"][-1], rep["diagnostics_train"]
        lines.append(f"| {name} | {len(rep['history'])} | {h['loss']:.4f} | {h['nce']:.4f} | {dg['std_mean']:.3f} / {dg['std_min']:.3f} | "
                     f"{dg['collapsed_dims_share']:.2f} | {dg['abs_offdiag_corr_mean']:.3f} | {dg['effective_rank']:.1f} / {dg['dim']} | "
                     f"{np.mean([x['seconds'] for x in rep['history']]):.0f} |")
    (out / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
