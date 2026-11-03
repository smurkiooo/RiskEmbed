"""One manually triggered research workflow, with no daily scoring or model registry."""
from datetime import datetime, timezone

from airflow.sdk import Param, dag, task

from riskembed.experiment import run_stage


@dag(
    dag_id="amex_experiment",
    schedule=None,
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    params={"quick": Param(True, type="boolean"), "force": Param(False, type="boolean")},
    default_args={"owner": "riskembed", "retries": 0},
    tags=["portfolio", "amex", "ml", "dl"],
)
def amex_experiment():
    @task
    def execute(stage: str, params: dict | None = None):
        params = params or {}
        run_stage(stage, quick=bool(params.get("quick", True)), force=bool(params.get("force", False)))

    audit = execute.override(task_id="audit_data")("audit")
    prepare = execute.override(task_id="prepare_data")("prepare")
    pretrain = execute.override(task_id="pretrain_encoder")("pretrain")
    supervised = execute.override(task_id="train_transformers")("supervised")
    boost = execute.override(task_id="train_boosting")("boost")
    analyze = execute.override(task_id="analyze_embeddings")("analyze")
    report = execute.override(task_id="build_report")("report")
    audit >> prepare >> pretrain >> supervised >> boost >> analyze >> report


dag_object = amex_experiment()
