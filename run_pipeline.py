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
from config import load_job_request


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
        help="Pipeline stages to skip, e.g. collect.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    job = load_job_request(args.config)

    if args.stages:
        job["stages"] = args.stages
    if args.skip_stages:
        job["skip_stages"] = args.skip_stages

    main(job)
