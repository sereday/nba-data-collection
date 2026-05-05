"""NBA Data Pipeline Runner

This script provides a command-line interface to run the NBA data collection and cleaning pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from collection import main
from config import load_job_request, get_skipped_stages


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the NBA data pipeline with optional stage selection."
    )
    parser.add_argument(
        "--config",
        default="job_request.json",
        help="Path to the JSON job configuration file.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        help="Pipeline stages to execute, e.g. collect.",
    )
    parser.add_argument(
        "--skip-stages",
        nargs="+",
        help="Pipeline stages to skip, e.g. clean.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-fetch and overwrite files that already exist on disk.",
    )
    return parser.parse_args()


def _check_run_name(job: dict) -> None:
    """Warn and prompt if GPM will run with an already-used run_name."""
    if "gpm" in get_skipped_stages(job):
        return
    run_name = job.get("run_name", "").strip()
    if not run_name:
        return
    history_path = ROOT / "results" / "top10_history.csv"
    if not history_path.exists():
        return
    try:
        import pandas as pd
        history = pd.read_csv(history_path, usecols=["run_name"])
        if run_name not in history["run_name"].values:
            return
    except Exception:
        return

    print(
        f"\n  WARNING: run_name '{run_name}' already has a successful GPM result in top10_history.csv.\n"
        f"  Change run_name in job_request.json to avoid a duplicate entry.\n"
    )
    answer = input("  Continue anyway? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    job = load_job_request(args.config)

    if args.stages:
        job["stages"] = args.stages
    if args.skip_stages:
        job["skip_stages"] = args.skip_stages
    if args.overwrite:
        job["overwrite"] = True

    _check_run_name(job)

    main(job)
