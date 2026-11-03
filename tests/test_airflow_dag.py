"""Validate the real DAG when the optional Airflow SDK is installed."""
import runpy
from itertools import pairwise

import pytest

from riskembed.config import PROJECT_ROOT


def test_airflow_experiment_dependencies():
    pytest.importorskip("airflow.sdk")
    dag = runpy.run_path(str(PROJECT_ROOT / "dags" / "amex_experiment.py"))["dag_object"]
    dag.validate()
    expected = ["audit_data", "prepare_data", "pretrain_encoder", "train_transformers", "train_boosting", "analyze_embeddings", "build_report"]
    assert set(dag.task_ids) == set(expected)
    for current, following in pairwise(expected):
        assert dag.get_task(current).downstream_task_ids == {following}
    assert dag.max_active_runs == 1 and not dag.catchup
