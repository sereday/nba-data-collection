"""Validation module for the NBA data pipeline.

Checks that all expected files have been imported and are non-empty.
Covers season-level data (import stage) and game-level data.

Usage:
    python src/validate.py
    python src/validate.py --game-level   # include game-level file checks
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from collection import GAME_LEVEL_DATA_TYPES, SEASON_TYPE_MIN_YEAR, VALID_SEASON_TYPES
from config import (
    build_season_plan,
    get_output_directory,
    get_output_format,
    load_job_request,
)


@dataclass
class ValidationReport:
    ok: int = 0
    missing: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.ok + len(self.missing) + len(self.empty)

    def print_summary(self, label: str) -> None:
        print(f"\n{label}")
        print("-" * len(label))
        print(f"  OK:      {self.ok:,}")
        print(f"  Empty:   {len(self.empty):,}")
        print(f"  Missing: {len(self.missing):,}")
        print(f"  Total:   {self.total:,}")
        if self.missing:
            print(f"\n  Missing files ({len(self.missing)}):")
            for f in sorted(self.missing)[:50]:
                print(f"    {f}")
            if len(self.missing) > 50:
                print(f"    ... and {len(self.missing) - 50} more")
        if self.empty:
            print(f"\n  Empty files ({len(self.empty)}):")
            for f in sorted(self.empty)[:50]:
                print(f"    {f}")
            if len(self.empty) > 50:
                print(f"    ... and {len(self.empty) - 50} more")


def _row_count(filepath: Path, output_format: str) -> int:
    try:
        if output_format == "csv":
            df = pd.read_csv(filepath, nrows=1)
            if len(df) == 0:
                df = pd.read_csv(filepath)
        else:
            df = pd.read_parquet(filepath)
        return len(df)
    except Exception:
        return 0


def validate_season_level(job: Dict[str, Any]) -> ValidationReport:
    """Check season-level files produced by the import stage."""
    output_dir = get_output_directory(job)
    output_format = get_output_format(job)
    seasons = build_season_plan(job)
    season_types = list(job.get("season_types", []))
    data_types = [dt for dt in job.get("data_types", []) if dt not in GAME_LEVEL_DATA_TYPES]

    report = ValidationReport()

    for season in seasons:
        season_year = int(season[:4])

        if "rosters" in data_types:
            filepath = output_dir / f"{season}_rosters.{output_format}"
            if not filepath.exists():
                report.missing.append(filepath.name)
            elif _row_count(filepath, output_format) == 0:
                report.empty.append(filepath.name)
            else:
                report.ok += 1

        for season_type in season_types:
            min_year = SEASON_TYPE_MIN_YEAR.get(season_type)
            if min_year is not None and season_year < min_year:
                continue

            for data_type in data_types:
                if data_type == "rosters":
                    continue
                filepath = output_dir / f"{season}_{season_type}_{data_type}.{output_format}"
                if not filepath.exists():
                    report.missing.append(filepath.name)
                elif _row_count(filepath, output_format) == 0:
                    report.empty.append(filepath.name)
                else:
                    report.ok += 1

    return report


def validate_game_level(job: Dict[str, Any]) -> ValidationReport:
    """Check game-level files produced by the import stage."""
    output_dir = get_output_directory(job)
    output_format = get_output_format(job)
    seasons = build_season_plan(job)
    season_types = list(job.get("season_types", []))
    game_data_types = [dt for dt in job.get("data_types", []) if dt in GAME_LEVEL_DATA_TYPES]

    if not game_data_types:
        print("No game-level data types configured — skipping game-level validation.")
        return ValidationReport()

    report = ValidationReport()

    for season in seasons:
        season_year = int(season[:4])
        for season_type in season_types:
            min_year = SEASON_TYPE_MIN_YEAR.get(season_type)
            if min_year is not None and season_year < min_year:
                continue

            game_log = output_dir / f"{season}_{season_type}_players.{output_format}"
            if not game_log.exists():
                continue

            if output_format == "csv":
                game_ids = pd.read_csv(game_log, usecols=["GAME_ID"])["GAME_ID"].astype(str).str.zfill(10).unique().tolist()
            else:
                game_ids = pd.read_parquet(game_log, columns=["GAME_ID"])["GAME_ID"].astype(str).str.zfill(10).unique().tolist()

            for data_type in game_data_types:
                _, min_season = GAME_LEVEL_DATA_TYPES[data_type]
                if min_season is not None and season_year < min_season:
                    continue

                game_dir = output_dir / "game_level" / season / season_type / data_type
                for game_id in game_ids:
                    filepath = game_dir / f"{game_id}.{output_format}"
                    rel = f"game_level/{season}/{season_type}/{data_type}/{game_id}.{output_format}"
                    if not filepath.exists():
                        report.missing.append(rel)
                    elif _row_count(filepath, output_format) == 0:
                        report.empty.append(rel)
                    else:
                        report.ok += 1

    return report


def run_validation(job: Dict[str, Any], include_game_level: bool = False) -> None:
    print("Running pipeline validation...\n")

    season_report = validate_season_level(job)
    season_report.print_summary("Season-level data (import stage)")

    if include_game_level:
        game_report = validate_game_level(job)
        game_report.print_summary("Game-level data (import stage)")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-level", action="store_true", default=False,
                        help="Also validate game-level files (slower — reads game log files)")
    args = parser.parse_args()

    job = load_job_request()
    run_validation(job, include_game_level=args.game_level)


if __name__ == "__main__":
    main()
