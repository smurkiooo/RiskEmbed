"""Collect measured results, align paired comparisons and build offline Markdown/HTML reports."""
import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

from riskembed.config import load_config, resolve
from riskembed.data.amex import amex_metric
from riskembed.evaluation.metrics import paired_gini_delta_bootstrap


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    root = resolve(cfg.reports_dir)
    if not args.quick:
        for script, extra in [('audit_leakage.py', []), ('audit_label_shuffle.py', ['--source', 'full'])]:
            subprocess.run([sys.executable, str(resolve('scripts') / script), *extra], check=True)
    leakage_path = root / 'leakage_audit.json'
    control_path = root / 'label_shuffle_control.json'
    leakage = json.loads(leakage_path.read_text()) if leakage_path.exists() and not args.quick else None
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    suffix = "_quick" if args.quick else ""
    training_dir = root / f"amex_training{suffix}"
    analysis_dir = root / f"amex_analysis{suffix}"
    supervised_dir = root / f"amex_supervised{suffix}"
    train = pd.read_csv(training_dir / "ablation.csv")
    analysis = pd.read_csv(analysis_dir / "ablation.csv")
    supervised = pd.read_csv(supervised_dir / "ablation.csv")
    # Recompute the competition metric from saved scores using the exact Lorenz normalization.
    for frame, directory in [(train, training_dir), (analysis, analysis_dir), (supervised, supervised_dir)]:
        for index, row in frame.iterrows():
            scored = pd.read_parquet(directory / f"{row['name']}_test.parquet")
            for key, value in amex_metric(scored.target, scored.score).items():
                frame.loc[index, key] = value
        frame.to_csv(directory / 'ablation.csv', index=False)
    baseline = pd.read_parquet(training_dir / "01_agg_baseline_test.parquet").rename(columns={"client_id": "cid"})
    for index, row in supervised.iterrows():
        prediction = pd.read_parquet(supervised_dir / f"{row['name']}_test.parquet")
        paired = baseline.merge(prediction, on="cid", suffixes=("_base", "_dl"), validate="one_to_one")
        if len(paired) != len(baseline) or len(paired) != len(prediction):
            raise ValueError("Different test customers in the paired comparison")
        if not np.array_equal(paired.target_base, paired.target_dl):
            raise ValueError("Target mismatch in paired predictions")
        delta = paired_gini_delta_bootstrap(paired.target_base, paired.score_base, paired.score_dl,
                                            paired.cid, cfg.amex.bootstrap_reps, cfg.seed)
        for key, value in {"delta_gini_test": delta["delta"], "delta_ci_low": delta["ci_low"],
                           "delta_ci_high": delta["ci_high"], "delta_p_le_0": delta["p_le_0"]}.items():
            supervised.loc[index, key] = value
    supervised.to_csv(supervised_dir / "ablation.csv", index=False)
    frames = [("Предобучение и бустинг", train), ("Прямой supervised Transformer", supervised),
              ("Остаточные эмбеддинги", analysis)]
    audit = json.loads((root / "data_audit.json").read_text())
    mode = "quick" if args.quick else "full"
    note = ("QUICK: запуск на подвыборке для проверки работоспособности. Для оценки качества нужен полный режим." if args.quick else
            "Полный источник AmEx. Локальный test состоит из отдельных клиентов официального train; leaderboard и OOT здесь не оцениваются.")
    lines = ["# AmEx: результаты эксперимента", "", note, "",
             f"Checkpoint / число деревьев выбираются только на valid. Один seed: {cfg.seed}; интервалы отражают перевыборку клиентов, не обучение между seed.",
             "Сдвинутый срез содержит сокращённую историю с прежними метками. Для временной валидации он не подходит.", "",
             f"Источник: {audit['profile']['n_rows']:,} выписок, {audit['profile']['customers']:,} клиентов.",
             f"SHA256 parquet: `{audit['sha256']}`.", "",
             "Из-за прореженного отрицательного класса score без коррекции/калибровки не является банковской PD.", ""]
    for title, frame in frames:
        cols = [c for c in ["name", "gini_valid", "gini_test", "gini_test_ci_low", "gini_test_ci_high", "delta_gini_test",
                            "delta_ci_low", "delta_ci_high", "amex_M", "best_epoch", "seconds"] if c in frame]
        lines += [f"## {title}", "", frame[cols].fillna("—").to_markdown(index=False, floatfmt=".5f"), ""]
    combined = pd.concat([train, analysis[analysis.name == "03_agg+emb_residual"], supervised], ignore_index=True)
    best = combined.loc[combined.gini_test.idxmax()]
    base_gini = float(train.loc[train.name == "01_agg_baseline", "gini_test"].iloc[0])
    conclusion = (f"LightGBM на агрегатах: Gini {base_gini:.5f}. Наибольший Gini среди сравниваемых моделей: "
                  f"{best['name']} — {best['gini_test']:.5f}. "
                  "Это сравнение при фиксированном бюджете и одном seed; выбор модели по test остаётся исследовательским наблюдением.")
    lines += ["## Что означают результаты", "", conclusion, "",
              "Пользу модификации оценивайте по парному CI разницы вместе с самим приростом Gini. "
              "Исторический test уже просматривался; новые исследования требуют заранее зафиксированного протокола."]
    target = root / f"EXPERIMENT{suffix}.md"
    target.write_text("\n".join(lines))
    tag = "quick" if args.quick else "full"
    labels = pd.read_parquet(resolve(cfg.amex.work_dir) / tag / "labels.parquet")
    curves = {name: json.loads((supervised_dir / f"{name}_training.json").read_text())
              for name in supervised.name}
    components = pd.read_csv(analysis_dir / "components.csv")
    payload = clean_json({"mode": mode, "result_note": note, "results": combined.to_dict("records"),
                          "conclusion": conclusion, "curves": curves, "seed": cfg.seed, "config": dict(cfg), "leakage_audit": leakage, "shuffle_control": control,
                          "bootstrap_reps": cfg.amex.bootstrap_reps, "source_sha256": audit["sha256"],
                          "splits": labels.split.value_counts().to_dict(),
                          "r2_median": float(components.r2_explained_by_aggregates.median()),
                          "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                          "packages": {name: version(name) for name in ["numpy", "pandas", "torch", "lightgbm", "duckdb", "pyarrow"]}},
                          "built_at": datetime.now(UTC).isoformat(timespec="seconds")})
    (root / f"experiment{suffix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    template = resolve("docs/project_overview.template.html").read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = template.replace("__PROJECT_DATA__", encoded)
    destination = resolve("project_overview_quick.html" if args.quick else "project_overview.html")
    destination.write_text(html)
    print(f"{target}\n{destination}")


if __name__ == "__main__":
    main()
