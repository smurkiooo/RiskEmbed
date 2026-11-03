"""Анализ на AmEx: остаточные эмбеддинги, стабильность компонент, SHAP, пробы.

    .venv/bin/python scripts/analyze_amex.py [--quick]

Эмбеддинги берутся из artifacts/ (посчитаны на этапе обучения), torch не нужен.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from riskembed.analysis.interpret import shap_by_group, top_features_by_shap
from riskembed.analysis.probes import probe_report
from riskembed.config import load_config, resolve
from riskembed.data.amex import ID_COL, amex_metric
from riskembed.evaluation.metrics import paired_gini_delta_bootstrap
from riskembed.features.psi import psi_numeric
from riskembed.features.residual import ResidualProjection
from riskembed.models.baseline import score_report, train_lgb

VARIANT = "coles_vicreg"
PSI_SAMPLE = 50_000  # train-клиентов для PSI «последний срез → сдвинутый»


def agg_files(path):
    return sorted(path.glob("*.parquet")) if path.is_dir() else [path.with_suffix(".parquet")]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    a = cfg.amex
    tag = "quick" if args.quick else "full"
    work = resolve(a.work_dir) / tag
    art = resolve("artifacts") / f"amex_{tag}"
    out = resolve(cfg.reports_dir) / ("amex_analysis_quick" if args.quick else "amex_analysis")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731

    labels = pd.read_parquet(work / "labels.parquet")
    n = len(labels)
    row_of = pd.Series(np.arange(n), index=labels[ID_COL].to_numpy())
    agg_parts = agg_files(work / "agg_last")
    agg_names = [c for c in pq.ParquetFile(agg_parts[0]).schema_arrow.names if c != ID_COL]

    def load_agg(parts, X, row_pos=None):
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

    emb_cols = [f"{VARIANT}__emb_{j:03d}" for j in range(a.encoder.emb_dim)]
    emb = pd.read_parquet(art / VARIANT / "embeddings.parquet",
                          columns=emb_cols + [f"{c}__shift" for c in emb_cols])
    E = emb[emb_cols].to_numpy(np.float64)
    E_shift = emb[[f"{c}__shift" for c in emb_cols]].to_numpy(np.float64)

    split = labels["split"].to_numpy()
    y = labels["target"].to_numpy(np.float32)
    idx = {s: np.nonzero(split == s)[0] for s in ("train", "valid", "test")}

    X_agg = np.full((n, len(agg_names)), np.nan, dtype=np.float32)
    load_agg(agg_parts, X_agg)
    log(f"агрегаты загружены: {X_agg.shape}, {X_agg.nbytes / 2**30:.2f} ГБ")

    # ---- остаточная проекция (обучается только на train) ----------------------
    res_names = [f"res_{j:03d}" for j in range(len(emb_cols))]
    proj = ResidualProjection.fit(X_agg[idx["train"]], E[idx["train"]], agg_names, emb_cols, alpha=100.0)
    proj.save(art / "residual_projection")
    R = proj.transform(X_agg, E)
    log(f"R² компонент: медиана {np.median(proj.r2):.3f}, мин {proj.r2.min():.3f}, макс {proj.r2.max():.3f}")

    # PSI компонент «последний срез → сдвинутый» (доступная на AmEx ось времени). Сравниваются только
    # клиенты, у которых есть выписки до cutoff: иначе NaN-бин один даёт PSI ≈ 0.2 любой компоненте.
    # Остаток сдвинутого среза считается по агрегатам того же сдвинутого среза.
    has_shift = np.isfinite(E_shift).all(axis=1)
    rng = np.random.default_rng(cfg.seed)
    cand = idx["train"][has_shift[idx["train"]]]
    sample = np.sort(rng.choice(cand, size=min(PSI_SAMPLE, len(cand)), replace=False))
    sample_pos = np.full(n, -1, dtype=np.int64)
    sample_pos[sample] = np.arange(len(sample))
    X_shift = np.full((len(sample), len(agg_names)), np.nan, dtype=np.float32)
    load_agg(agg_files(work / "agg_shift"), X_shift, row_pos=sample_pos)
    R_shift = proj.transform(X_shift, E_shift[sample])
    psi_raw = np.array([psi_numeric(E[sample, j], E_shift[sample, j]) for j in range(E.shape[1])])
    psi_res = np.array([psi_numeric(R[sample, j], R_shift[:, j]) for j in range(R.shape[1])])
    comp = pd.DataFrame({"component": emb_cols, "residual": res_names,
                         "r2_explained_by_aggregates": proj.r2, "psi_raw": psi_raw, "psi_residual": psi_res})
    comp.to_csv(out / "components.csv", index=False)
    log(f"PSI компонент на {len(sample):,} клиентах: сырых медиана {np.median(psi_raw):.4f}, "
        f"остаточных {np.median(psi_res):.4f}; клиентов без сдвинутого среза {1 - has_shift.mean():.1%}")
    del X_shift, R_shift

    # ---- матрица признаков: агрегаты + сырой эмбеддинг + остаток ---------------
    all_names = agg_names + emb_cols + res_names
    col_of = {c: j for j, c in enumerate(all_names)}
    X = np.full((n, len(all_names)), np.nan, dtype=np.float32)
    X[:, : len(agg_names)] = X_agg
    X[:, [col_of[c] for c in emb_cols]] = E.astype(np.float32)
    X[:, [col_of[c] for c in res_names]] = R.astype(np.float32)
    del X_agg
    log(f"матрица признаков {X.shape}, {X.nbytes / 2**30:.2f} ГБ")

    rows_def = [("01_agg_baseline", agg_names), ("02_agg+emb_raw", agg_names + emb_cols),
                ("03_agg+emb_residual", agg_names + res_names), ("04_residual_only", res_names)]

    b = a.boosting
    results, scores, models = [], {}, {}

    def predict(model, cols, rows):
        res = np.empty(len(rows))
        for s in range(0, len(rows), 50_000):
            r = rows[s : s + 50_000]
            res[s : s + 50_000] = model.predict(X[np.ix_(r, cols)], num_iteration=model.best_iteration)
        return res

    for name, feats in rows_def:
        t1 = time.time()
        cols = np.array([col_of[f] for f in feats])
        model = train_lgb(X[np.ix_(idx["train"], cols)], y[idx["train"]], X[np.ix_(idx["valid"], cols)], y[idx["valid"]],
                          dict(b.params), feats, b.num_boost_round, b.early_stopping_rounds, cfg.seed)
        scored = pd.concat([pd.DataFrame({"client_id": labels["cid"].to_numpy()[idx[s]], "split": s,
                                          "target": y[idx[s]], "score": predict(model, cols, idx[s])})
                            for s in ("train", "valid", "test")], ignore_index=True)
        m, _ = score_report(scored, name, "test", "target", a.bootstrap_reps, cfg.seed, period_col=None, business_metric=False)
        destination = art / "analysis_models" / name
        destination.mkdir(parents=True, exist_ok=True)
        model.save_model(str(destination / "model.txt"), num_iteration=model.best_iteration)
        ts = scored[scored.split == "test"]
        ts.to_parquet(out / f"{name}_test.parquet", index=False)
        scores[name] = ts["score"].to_numpy()
        models[name] = (model, feats, cols)
        m |= {k: float(v) for k, v in amex_metric(ts["target"].to_numpy(), scores[name]).items()}
        m |= {"n_features": len(feats), "best_iteration": model.best_iteration,
              "n_neuro_features": len([f for f in feats if f.startswith(("res_", VARIANT))]),
              "seconds": round(time.time() - t1, 1)}
        results.append(m)
        log(f"{name}: gini test {m['gini_test']:.4f} [{m['gini_test_ci_low']:.4f}; {m['gini_test_ci_high']:.4f}] ({m['seconds']}s)")

    y_test, cid_test = y[idx["test"]], labels["cid"].to_numpy()[idx["test"]]
    for r in results[1:]:
        d = paired_gini_delta_bootstrap(y_test, scores["01_agg_baseline"], scores[r["name"]], cid_test, a.bootstrap_reps, cfg.seed)
        r |= {"delta_gini_test": d["delta"], "delta_ci_low": d["ci_low"], "delta_ci_high": d["ci_high"], "delta_p_le_0": d["p_le_0"]}
    res_df = pd.DataFrame(results)
    res_df.to_csv(out / "ablation.csv", index=False)

    # ---- SHAP по группам и пробы ------------------------------------------------
    model, feats, cols = models["03_agg+emb_residual"]
    shap_rows = idx["test"][: 20_000]
    shap_groups = shap_by_group(model, X[np.ix_(shap_rows, cols)], feats,
                                {"агрегаты выписок": agg_names, "остаточный эмбеддинг": res_names})
    shap_groups.to_csv(out / "shap_groups.csv", index=False)
    top_features_by_shap(model, X[np.ix_(shap_rows, cols)], feats, top=20).to_csv(out / "shap_top_features.csv", index=False)

    probe_targets = pd.DataFrame({name: X[:, col_of[name]] for name in
                                  ["n_statements", "history_days", "mean_gap_days", "P_2__last", "P_2__mean", "B_1__last",
                                   "D_39__mean", "S_3__std"] if name in col_of})
    probes = probe_report(E, probe_targets, split == "train", split == "test")
    probes.to_csv(out / "probes.csv", index=False)
    log(f"пробы: {dict(zip(probes['target'], probes['r2'].round(3), strict=True))}")

    write_report(out, res_df, comp, shap_groups, probes, len(agg_names), a.shift_cutoff, args.quick)
    (out / "summary.json").write_text(json.dumps({"rows": results, "components_r2_median": float(np.median(proj.r2))},
                                                 indent=2, ensure_ascii=False, default=float))
    log(f"готово → {out / 'README.md'}")


def write_report(out, res, comp, shap_groups, probes, n_agg, shift_cutoff, quick) -> None:
    def delta(r):
        return "—" if pd.isna(r.get("delta_gini_test")) else (
            f"{r['delta_gini_test']:+.4f} [{r['delta_ci_low']:+.4f}; {r['delta_ci_high']:+.4f}]")

    lines = [
        f"# AmEx: остаточные эмбеддинги и стабильность компонент{' (QUICK)' if quick else ''}", "",
        "## Сравнение на test-клиентах, исключённых из обучения энкодера и бустинга", "",
        "| строка | признаков | нейро | Gini test [95% CI] | ΔGini к бейзлайну | P(Δ≤0) | AmEx M | деревьев |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        p_le = "—" if pd.isna(r.get("delta_p_le_0")) else f"{r['delta_p_le_0']:.3f}"
        lines.append(f"| {r['name']} | {r['n_features']} | {r['n_neuro_features']} | "
                     f"{r['gini_test']:.4f} [{r['gini_test_ci_low']:.4f}; {r['gini_test_ci_high']:.4f}] | {delta(r)} | {p_le} | "
                     f"{r['amex_M']:.4f} | {r['best_iteration']} |")
    lines += ["", "## Связь эмбеддинга с агрегатами", "",
              f"- R² компоненты по {n_agg} агрегатам: медиана **{comp['r2_explained_by_aggregates'].median():.3f}**, "
              f"минимум {comp['r2_explained_by_aggregates'].min():.3f}, максимум {comp['r2_explained_by_aggregates'].max():.3f}",
              f"- PSI компонент (последний срез → срез по выпискам до {shift_cutoff}, одни и те же train-клиенты): "
              f"сырых — медиана {comp['psi_raw'].median():.4f}, максимум {comp['psi_raw'].max():.4f}; "
              f"остаточных — медиана {comp['psi_residual'].median():.4f}, максимум {comp['psi_residual'].max():.4f}", "",
              "## SHAP по группам (строка 03, test)", "", "| группа | признаков | средний \\|SHAP\\| | доля |", "|---|---|---|---|"]
    for _, r in shap_groups.iterrows():
        lines.append(f"| {r['group']} | {r['n_features']} | {r['mean_abs_shap']:.4f} | {r['share']:.1%} |")
    lines += ["", "## Линейные пробы (R² восстановления агрегата из эмбеддинга, test)", "",
              "| что восстанавливаем | R² |", "|---|---|"]
    for _, r in probes.iterrows():
        lines.append(f"| {r['target']} | {r['r2']:.3f} |")
    (out / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
