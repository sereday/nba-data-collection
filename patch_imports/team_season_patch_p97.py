"""Standalone runner: pre-1996-97 team season stats backfill.

Usage:
    python patch_imports/team_season_patch_p97.py
    python patch_imports/team_season_patch_p97.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import load_job_request
from patch import run_team_patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing output files (default: skip)")
    args = parser.parse_args()
    run_team_patch(load_job_request(), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
