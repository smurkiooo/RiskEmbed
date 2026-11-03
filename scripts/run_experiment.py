"""Run the same stages locally as the Airflow DAG."""
import argparse

from riskembed.experiment import STAGES, run_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute encoder artifacts")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    args = parser.parse_args()
    for stage in STAGES if args.stage == "all" else [args.stage]:
        run_stage(stage, args.quick, args.force)


if __name__ == "__main__":
    main()
