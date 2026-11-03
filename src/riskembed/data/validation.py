"""Structural validation and fingerprinting of the local AmEx training source."""
from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from riskembed.data.amex import CAT_COLS

EXPECTED_ROWS = 5_531_451
EXPECTED_CUSTOMERS = 458_913


def audit_amex(raw_dir: str | Path, require_full: bool = True) -> dict:
    raw = Path(raw_dir)
    parquet, labels = raw / "train.parquet", raw / "train_labels.csv"
    pf = pq.ParquetFile(parquet)
    names = pf.schema_arrow.names
    required = {"customer_ID", "S_2"}
    if not required.issubset(names):
        raise ValueError(f"Missing key columns: {sorted(required - set(names))}")
    with duckdb.connect() as con:
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=2")
        con.read_parquet(str(parquet)).create_view("x")
        con.read_csv(str(labels), header=True).create_view("y")

        def query(sql):
            return con.execute(sql).df().to_dict("records")

        profile = query("""SELECT count(*) AS n_rows, count(DISTINCT customer_ID) AS customers,
            count(*) - count(DISTINCT (customer_ID, S_2)) AS duplicates,
            count(*) FILTER (WHERE customer_ID IS NULL OR S_2 IS NULL) AS null_keys,
            count(*) FILTER (WHERE try_cast(S_2 AS DATE) IS NULL) AS invalid_dates,
            min(try_cast(S_2 AS DATE))::VARCHAR AS min_date,
            max(try_cast(S_2 AS DATE))::VARCHAR AS max_date FROM x""")[0]
        lab = query("""SELECT count(*) AS n_rows, count(DISTINCT customer_ID) AS customers,
            avg(target) AS positive_rate,
            count(*) FILTER (WHERE customer_ID IS NULL OR target IS NULL OR target NOT IN (0,1)) AS invalid
            FROM y""")[0]
        coverage = query("""SELECT
            (SELECT count(*) FROM (SELECT DISTINCT customer_ID FROM x) a ANTI JOIN y USING(customer_ID)) AS unlabeled,
            (SELECT count(*) FROM y ANTI JOIN x USING(customer_ID)) AS missing_history""")[0]
        lengths = query("""SELECT n, count(*) AS customers FROM
            (SELECT customer_ID, count(*) AS n FROM x GROUP BY customer_ID) GROUP BY n ORDER BY n""")
        last_months = query("""SELECT strftime(d,'%Y-%m') AS statement_month, count(*) AS customers FROM
            (SELECT customer_ID, max(try_cast(S_2 AS DATE)) AS d FROM x GROUP BY customer_ID)
            GROUP BY statement_month ORDER BY statement_month""")
    features = [n for n in names if n not in required]
    checks = {
        "nonempty": profile["n_rows"] > 0,
        "unique_customer_dates": profile["duplicates"] == 0,
        "valid_keys_and_dates": profile["null_keys"] == profile["invalid_dates"] == 0,
        "unique_binary_labels": lab["n_rows"] == lab["customers"] and lab["invalid"] == 0,
        "both_classes": lab["positive_rate"] is not None and 0 < lab["positive_rate"] < 1,
        "exact_label_coverage": coverage["unlabeled"] == coverage["missing_history"] == 0,
        "history_length_1_to_13": bool(lengths) and all(1 <= r["n"] <= 13 for r in lengths),
        "feature_names_only": all(n.startswith(("B_", "D_", "P_", "R_", "S_")) for n in features),
    }
    if require_full:
        checks |= {
            "full_row_count": profile["n_rows"] == EXPECTED_ROWS,
            "full_customer_count": profile["customers"] == EXPECTED_CUSTOMERS,
            "full_feature_schema": len(features) == 188 and set(CAT_COLS).issubset(features),
            "expected_dates": profile["min_date"] == "2017-03-01" and profile["max_date"] == "2018-03-31",
            "last_statement_march_2018": last_months == [{"statement_month": "2018-03", "customers": EXPECTED_CUSTOMERS}],
        }
    fingerprint = hashlib.sha256()
    with parquet.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            fingerprint.update(block)
    label_hash = hashlib.sha256(labels.read_bytes()).hexdigest()
    return {
        "passed": all(checks.values()), "checks": checks, "profile": profile, "labels": lab,
        "coverage": coverage, "lengths": lengths, "last_months": last_months,
        "file_bytes": parquet.stat().st_size, "sha256": fingerprint.hexdigest(), "labels_sha256": label_hash,
        "schema": {f.name: str(f.type) for f in pf.schema_arrow},
        "limitation": "Structural completeness is checked; value-level equivalence to the official CSV is not verified.",
    }
