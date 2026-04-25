"""NBA data collection module.

This module handles fetching data from the NBA API and saving it to disk.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerBioStats, leaguegamelog

from config import (
    build_season_plan,
    get_output_directory,
    get_output_format,
    get_requested_stages,
    get_skipped_stages,
    load_job_request,
)
from cleaning import run_clean_stage

VALID_SEASON_TYPES = {
    "Regular": "Regular Season",
    "Regular Season": "Regular Season",
    "Playoffs": "Playoffs",
    "Preseason": "Pre Season",
    "Pre Season": "Pre Season",
    "All-Star": "All Star",
    "All Star": "All Star",
    "AllStar": "All Star",
}

PLAYER_OR_TEAM_MAP = {
    "players": "P",
    "teams": "T",
}

ALL_PIPELINE_STAGES = ["collect", "clean"]


def fetch_league_game_log(season: str, season_type: str, player_or_team: str) -> Optional[pd.DataFrame]:
    """Fetch NBA league game log data from the API."""
    api_season_type = VALID_SEASON_TYPES.get(season_type, season_type)
    print(f"Fetching {player_or_team} game logs for {season} ({season_type})...")

    try:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=api_season_type,
            player_or_team_abbreviation=player_or_team,
        )
        df = log.get_data_frames()[0]
        print(f"  Retrieved {len(df):,} records")
        return df
    except Exception as error:
        print(f"  Error fetching {season_type} for {season}: {error}")
        return None


def fetch_player_bio_data(season: str, season_type: str) -> Optional[pd.DataFrame]:
    """Fetch NBA player bio stats data from the API."""
    api_season_type = VALID_SEASON_TYPES.get(season_type, season_type)
    print(f"Fetching player bios for {season} ({season_type})...")

    try:
        log = LeagueDashPlayerBioStats(
            season=season,
            season_type_all_star=api_season_type,
            per_mode_simple="Totals",
        )
        df = log.get_data_frames()[0]
        print(f"  Retrieved {len(df):,} records")
        return df
    except Exception as error:
        print(f"  Error fetching player bios for {season} ({season_type}): {error}")
        return None


def save_data(df: pd.DataFrame, output_dir: Path, filename: str, output_format: str) -> None:
    """Persist a DataFrame to disk in CSV or Parquet format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.{output_format}"

    if output_format == "csv":
        df.to_csv(filepath, index=False)
    elif output_format == "parquet":
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"  Saved to {filepath}")


def get_enabled_stages(job: Dict[str, Any]) -> list[str]:
    """Determine which pipeline stages are enabled based on job config."""
    requested = get_requested_stages(job)
    skipped = get_skipped_stages(job)
    enabled: list[str] = []

    for stage in requested:
        if stage in skipped:
            print(f"Skipping configured stage: {stage}")
            continue
        if stage not in ALL_PIPELINE_STAGES:
            print(f"Ignoring unknown stage: {stage}")
            continue
        enabled.append(stage)

    if not enabled:
        raise ValueError(
            "No pipeline stages enabled. Use 'stages' or remove 'skip_stages' from job_request.json."
        )

    return enabled


def run_collect_stage(job: Dict[str, Any]) -> None:
    """Run the data collection stage, fetching and saving data from NBA API."""
    output_dir = get_output_directory(job)
    output_format = get_output_format(job)
    seasons = build_season_plan(job)
    season_types = list(job["season_types"])
    data_types = list(job["data_types"])

    print(f"Running collect stage for {len(seasons)} season(s): {seasons}")
    print(f"Output format: {output_format}")
    print(f"Target data types: {data_types}\n")

    for season in seasons:
        for season_type in season_types:
            for data_type in data_types:
                if data_type in ["players", "teams"]:
                    df = fetch_league_game_log(season, season_type, PLAYER_OR_TEAM_MAP[data_type])
                elif data_type == "player_bios":
                    df = fetch_player_bio_data(season, season_type)
                else:
                    print(f"Unknown data type: {data_type}")
                    continue

                if df is not None:
                    filename = f"{season}_{season_type}_{data_type}"
                    save_data(df, output_dir, filename, output_format)


def run_job(job: Dict[str, Any]) -> None:
    """Execute the enabled pipeline stages."""
    enabled_stages = get_enabled_stages(job)
    print(f"Enabled pipeline stages: {enabled_stages}\n")

    if "collect" in enabled_stages:
        run_collect_stage(job)

    if "clean" in enabled_stages:
        run_clean_stage(job)

    print("\nPipeline completed successfully.")


def main(job: Dict[str, Any] | None = None) -> None:
    """Main entry point for the data pipeline."""
    if job is None:
        job = load_job_request()
    run_job(job)
