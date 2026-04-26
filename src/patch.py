"""Pre-1996-97 backfill logic for the NBA data pipeline.

Provides run_patch_stage(job, overwrite=False) for use by the pipeline,
and individual functions used by the standalone patch_imports/ scripts.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from nba_api.stats.endpoints import CommonTeamYears, PlayerCareerStats, TeamYearByYearStats

from config import get_output_directory, get_output_format

PRE97_CUTOFF = 1996

CAREER_TABLE_SEASON_TYPES = {
    0: "Regular",
    2: "Playoffs",
}

TEAM_SEASON_TYPES = [
    ("Regular Season", "Regular"),
    ("Playoffs", "Playoffs"),
]


def get_pre97_player_ids(data_dir: Path, output_format: str) -> list[int]:
    player_ids: set[int] = set()
    for f in sorted(data_dir.glob(f"*_*_players.{output_format}")):
        season = f.stem.split("_")[0]
        if int(season[:4]) < PRE97_CUTOFF:
            df = pd.read_csv(f, usecols=["PLAYER_ID"]) if output_format == "csv" else pd.read_parquet(f, columns=["PLAYER_ID"])
            player_ids.update(df["PLAYER_ID"].tolist())
    return sorted(player_ids)


def get_pre97_team_ids() -> list[int]:
    df = CommonTeamYears(league_id="00").get_data_frames()[0]
    return df[df["MIN_YEAR"].astype(int) < PRE97_CUTOFF]["TEAM_ID"].tolist()


def run_player_patch(job: Dict[str, Any], overwrite: bool = False) -> None:
    output_dir = get_output_directory(job)
    output_format = get_output_format(job)

    player_ids = get_pre97_player_ids(output_dir, output_format)
    print(f"Found {len(player_ids):,} unique pre-1996-97 players\n")

    buckets: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)

    for i, player_id in enumerate(player_ids):
        print(f"[{i + 1}/{len(player_ids)}] Player {player_id}...", end=" ", flush=True)
        try:
            result = PlayerCareerStats(player_id=player_id, per_mode36="Totals")
            all_tables = result.get_data_frames()
            for table_idx, season_type in CAREER_TABLE_SEASON_TYPES.items():
                if table_idx >= len(all_tables):
                    continue
                df = all_tables[table_idx]
                if df.empty:
                    continue
                df = df[df["SEASON_ID"].apply(lambda s: int(str(s)[:4]) < PRE97_CUTOFF)]
                if df.empty:
                    continue
                for season, group in df.groupby("SEASON_ID"):
                    buckets[(str(season), season_type)].append(group)
            print("ok")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.6)

    print(f"\nSaving {len(buckets)} season-type files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for (season, season_type), frames in sorted(buckets.items()):
        filepath = output_dir / f"{season}_{season_type}_player_season_stats.{output_format}"
        if filepath.exists() and not overwrite:
            print(f"  Skipping {filepath.name} (already exists)")
            continue
        combined = pd.concat(frames, ignore_index=True)
        if output_format == "csv":
            combined.to_csv(filepath, index=False)
        else:
            combined.to_parquet(filepath, index=False)
        print(f"  Saved {filepath} ({len(combined):,} rows)")


def run_team_patch(job: Dict[str, Any], overwrite: bool = False) -> None:
    output_dir = get_output_directory(job)
    output_format = get_output_format(job)

    team_ids = get_pre97_team_ids()
    print(f"Found {len(team_ids)} franchises active before 1996-97\n")

    buckets: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)

    for i, team_id in enumerate(team_ids):
        print(f"[{i + 1}/{len(team_ids)}] Team {team_id}...")
        for api_season_type, label in TEAM_SEASON_TYPES:
            try:
                result = TeamYearByYearStats(
                    team_id=team_id,
                    per_mode_simple="Totals",
                    season_type_all_star=api_season_type,
                )
                df = result.get_data_frames()[0]
                if df.empty:
                    continue
                df = df[df["YEAR"].apply(lambda y: int(str(y)[:4]) < PRE97_CUTOFF)]
                if df.empty:
                    continue
                for season, group in df.groupby("YEAR"):
                    buckets[(str(season), label)].append(group)
                print(f"  {label}: {len(df)} pre-97 seasons")
            except Exception as e:
                print(f"  ERROR {label}: {e}")
        time.sleep(0.6)

    print(f"\nSaving {len(buckets)} season-type files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for (season, season_type), frames in sorted(buckets.items()):
        filepath = output_dir / f"{season}_{season_type}_team_season_stats.{output_format}"
        if filepath.exists() and not overwrite:
            print(f"  Skipping {filepath.name} (already exists)")
            continue
        combined = pd.concat(frames, ignore_index=True)
        if output_format == "csv":
            combined.to_csv(filepath, index=False)
        else:
            combined.to_parquet(filepath, index=False)
        print(f"  Saved {filepath} ({len(combined):,} rows)")


def run_patch_stage(job: Dict[str, Any], overwrite: bool = False) -> None:
    print("--- Player season patch (pre-1996-97) ---")
    run_player_patch(job, overwrite=overwrite)
    print("\n--- Team season patch (pre-1996-97) ---")
    run_team_patch(job, overwrite=overwrite)
