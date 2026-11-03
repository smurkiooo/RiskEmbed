"""Validate the complete local AmEx source before preparation or training."""
import json

from riskembed.config import load_config, resolve
from riskembed.data.validation import audit_amex


def main():
    cfg = load_config()
    report = audit_amex(resolve(cfg.amex.raw_dir))
    out = resolve(cfg.reports_dir) / "data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "profile": report["profile"]}, indent=2))
    if not report["passed"]:
        raise SystemExit("AmEx validation failed; see reports/data_audit.json")


if __name__ == "__main__":
    main()
