"""Shared experiment stages for the local CLI and Airflow; heavy libraries stay in subprocesses."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime

from riskembed.config import PROJECT_ROOT, load_config, resolve

STAGES = {
    "audit": ("audit_amex.py",),
    "prepare": ("prepare_amex.py",),
    "pretrain": ("train_amex.py", "--stage", "pretrain"),
    "supervised": ("train_supervised.py",),
    "boost": ("train_amex.py", "--stage", "boost"),
    "analyze": ("analyze_amex.py",),
    "report": ("make_report.py",),
}


def run_stage(stage: str, quick: bool = False, force: bool = False) -> None:
    script, *args = STAGES[stage]
    flags = ["--quick"] if quick and stage != "audit" else []
    if force and stage == "pretrain":
        flags.append("--force")
    cfg = load_config()
    manifest = resolve(cfg.reports_dir) / ("run_quick.json" if quick else "run_full.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(manifest.read_text()) if manifest.exists() else {"mode": "quick" if quick else "full", "stages": {}}
    record = {"status": "running", "started_at": datetime.now(UTC).isoformat(), "config": dict(cfg)}
    state["stages"][stage] = record
    manifest.write_text(json.dumps(state, indent=2))
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / script), *args, *flags],
                       cwd=PROJECT_ROOT, check=True)
        record["status"] = "passed"
    except BaseException as error:
        record |= {"status": "failed", "error": str(error)}
        raise
    finally:
        record |= {"finished_at": datetime.now(UTC).isoformat(), "seconds": round(time.monotonic() - started, 2)}
        latest = json.loads(manifest.read_text()) if manifest.exists() else state
        latest["stages"][stage] = record
        manifest.write_text(json.dumps(latest, indent=2))
