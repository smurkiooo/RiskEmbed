"""Independent checks on actual customer splits, feature artifacts and saved predictions.

This is evidence against implementation leaks, not proof of semantic availability of anonymized features.
"""
import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from riskembed.config import load_config, resolve
from riskembed.data.amex import customer_int_id
from riskembed.evaluation.metrics import auc
from riskembed.evaluation.splits import make_client_split


def main():
    cfg = load_config()
    work = resolve(cfg.amex.work_dir) / "full"
    labels = pd.read_parquet(work / "labels.parquet")
    original = pd.read_csv(resolve(cfg.amex.raw_dir) / "train_labels.csv")
    original["cid"] = customer_int_id(original.customer_ID)
    groups = {s: set(labels.loc[labels.split == s, "cid"]) for s in ("train", "valid", "test")}
    overlaps = {f"{a}_{b}": len(groups[a] & groups[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    restored = labels.merge(original[["cid", "target"]], on="cid", suffixes=("_prepared", "_original"), validate="one_to_one")
    raw_names = pq.ParquetFile(resolve(cfg.amex.raw_dir) / "train.parquet").schema_arrow.names
    agg = work / "agg_last"
    names = pq.ParquetFile(next(agg.glob('*.parquet'))).schema_arrow.names
    features = [name for name in names if name != 'customer_ID']
    forbidden = [name for name in raw_names + features if name.lower() in {'target','label','split','fold','prediction','score','cid'}]
    # Full-row aggregate fingerprints can reveal duplicate feature profiles across nominally different clients.
    with duckdb.connect() as con:
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=1")
        con.register('label_frame', labels[['customer_ID','split','target']])
        con.read_parquet(str(agg / '*.parquet')).create_view('aggregates')
        expr = ','.join('"' + name.replace('"','""') + '"' for name in features)
        duplicate_profiles = con.execute(f'''WITH h AS (
            SELECT customer_ID, hash({expr}) AS fingerprint FROM aggregates
        ), d AS (
            SELECT fingerprint, count(*) AS n, count(DISTINCT split) AS n_splits
            FROM h JOIN label_frame USING(customer_ID) GROUP BY fingerprint HAVING count(*)>1
        ) SELECT count(*) AS duplicate_groups,
                 coalesce(sum(n),0) AS clients_in_duplicate_groups,
                 count(*) FILTER (WHERE n_splits>1) AS groups_crossing_splits FROM d''').df().iloc[0].to_dict()
    # One raw aggregate, with direction and imputation fitted on train only.
    p2 = pd.concat([pd.read_parquet(part, columns=['customer_ID','P_2__last']) for part in sorted(agg.glob('*.parquet'))])
    p2 = labels[['customer_ID','split','target']].merge(p2, on='customer_ID', validate='one_to_one')
    reference = p2[p2.split == 'train']
    median = float(reference.P_2__last.median())
    direction = 1 if auc(reference.target, reference.P_2__last.fillna(median)) >= 0.5 else -1
    single_feature = {split: auc(frame.target, direction*frame.P_2__last.fillna(median))
                      for split,frame in p2.groupby('split')}
    score_path = resolve(cfg.reports_dir) / 'amex_supervised' / 'transformer_scratch_test.parquet'
    scores = pd.read_parquet(score_path)
    checked = scores.merge(original[['cid','target']], on='cid', suffixes=('_saved','_original'), validate='one_to_one')
    metric_own = auc(checked.target_original, checked.score)
    metric_sklearn = roc_auc_score(checked.target_original, checked.score)
    checks = {
        'customer_splits_disjoint': not any(overlaps.values()),
        'unique_raw_and_prepared_ids': original.cid.is_unique and labels.cid.is_unique,
        'all_original_labels_present': len(restored) == len(original) == len(labels),
        'prepared_labels_equal_original': bool((restored.target_prepared == restored.target_original).all()),
        'split_recomputes_without_target': bool(np.array_equal(labels.split, make_client_split(labels.cid, cfg.amex.holdout_client_share))),
        'no_explicit_target_or_split_in_features': not forbidden,
        'test_scores_cover_only_test_customers': set(scores.cid) == groups['test'],
        'saved_test_labels_equal_original': bool((checked.target_saved == checked.target_original).all()),
        'metric_matches_independent_sklearn': bool(abs(metric_own-metric_sklearn)<1e-12),
    }
    report = {'checks': checks, 'all_implementation_checks_passed': all(checks.values()),
              'single_feature_P_2_last_auc': single_feature, 'overlap_counts': overlaps, 'forbidden_columns': forbidden,
              'split_bad_rates': labels.groupby('split').target.mean().to_dict(),
              'duplicate_aggregate_profiles': {k:int(v) for k,v in duplicate_profiles.items()},
              'scratch_test_auc_recomputed_from_original_labels': metric_sklearn,
              'scratch_test_gini_recomputed':2*metric_sklearn-1,
              'limitations':['No original CSV for value-level verification of third-party preprocessing.',
                             'Anonymized feature availability cannot be semantically established from code alone.',
                             'Customer holdout is not temporal holdout; test has been viewed during previous research.',
                             'Duplicate profiles use a 64-bit screening hash; any matches require value-level investigation.']}
    path=resolve(cfg.reports_dir)/'leakage_audit.json'
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not all(checks.values()):
        raise SystemExit('Leakage audit failed')


if __name__=='__main__':
    main()
