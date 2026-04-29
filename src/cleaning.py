"""NBA data cleaning module.

This module handles loading collected data, merging datasets, and producing cleaned player-level data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config import get_output_directory, get_output_format


def run_clean_stage(job: Dict[str, Any]) -> None:
    output_dir = get_output_directory(job)
    output_format = get_output_format(job)

    print("Running clean stage using all available saved data files")
    print(f"Output format: {output_format}\n")

    # Load all data files from the output directory
    all_player_dfs = []
    all_team_dfs = []
    all_bio_dfs = []
    all_roster_dfs = []
    all_player_season_dfs = []
    all_team_season_dfs = []

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    for filepath in sorted(output_dir.iterdir()):
        if filepath.suffix not in {".csv", ".parquet"}:
            continue

        parts = filepath.stem.split("_")

        # Parse filename: either {season}_{season_type}_{data_type} or {season}_{data_type}
        # data_type may contain underscores (e.g. "player_season_stats")
        data_type = None
        season_type = None

        # Try season-type-aware format first: need at least 3 parts
        if len(parts) >= 3:
            for i in range(len(parts) - 1, 1, -1):
                potential_data_type = "_".join(parts[i:])
                if potential_data_type in {"players", "teams", "player_bios", "player_season_stats", "team_season_stats"}:
                    data_type = potential_data_type
                    season_type = parts[i - 1]
                    season = "_".join(parts[:i - 1])
                    break

        # Try season-only format: {season}_rosters
        if data_type is None and len(parts) >= 2:
            potential_data_type = "_".join(parts[1:])
            if potential_data_type == "rosters":
                data_type = potential_data_type
                season_type = None
                season = parts[0]

        if data_type is None:
            continue

        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_parquet(filepath)

        df["season"] = season
        df["season_type"] = season_type

        if data_type == "players":
            all_player_dfs.append(df)
        elif data_type == "teams":
            all_team_dfs.append(df)
        elif data_type == "player_bios":
            all_bio_dfs.append(df)
        elif data_type == "rosters":
            all_roster_dfs.append(df)
        elif data_type == "player_season_stats":
            all_player_season_dfs.append(df)
        elif data_type == "team_season_stats":
            all_team_season_dfs.append(df)

    if not all_player_dfs:
        print("No player data found, skipping clean stage.")
        return

    # Concatenate all player data across seasons and types
    player_df = pd.concat(all_player_dfs, ignore_index=True)

    # Concatenate all roster data
    if all_roster_dfs:
        roster_df = pd.concat(all_roster_dfs, ignore_index=True)
        # Outer join player to roster on PLAYER_ID (and TEAM_ID if needed)
        # Roster data typically includes player info like position, height, weight, etc.
        player_df = player_df.merge(roster_df, on=['PLAYER_ID'], how='left', suffixes=('', '_roster'))
    else:
        print("No roster data found, proceeding without roster join.")

    # Keep bio data available but don't join it to player data
    if all_bio_dfs:
        bio_df = pd.concat(all_bio_dfs, ignore_index=True)
        print(f"Bio data available with {len(bio_df)} records (not joined to player data)")
    else:
        print("No bio data found.")

    # Concatenate season stats data
    if all_player_season_dfs:
        player_season_df = pd.concat(all_player_season_dfs, ignore_index=True)
        print(f"Player season stats available with {len(player_season_df)} records")
    else:
        print("No player season stats found.")

    if all_team_season_dfs:
        team_season_df = pd.concat(all_team_season_dfs, ignore_index=True)
        print(f"Team season stats available with {len(team_season_df)} records")
    else:
        print("No team season stats found.")

    # Concatenate all team data
    if all_team_dfs:
        team_df = pd.concat(all_team_dfs, ignore_index=True)

        # Create opp_id map
        team_grouped = team_df.groupby('GAME_ID')['TEAM_ID'].agg(['min', 'max']).reset_index()
        opp_map1 = team_grouped.rename(columns={'min': 'TEAM_ID', 'max': 'OPP_ID'})
        opp_map2 = team_grouped.rename(columns={'max': 'TEAM_ID', 'min': 'OPP_ID'})
        opp_map = pd.concat([opp_map1, opp_map2], ignore_index=True)
        opp_gm_map = {f"{row['GAME_ID']}_{row['TEAM_ID']}": row['OPP_ID'] for _, row in opp_map.iterrows()}

        # Add opp_id to team_df
        team_df['opp_id'] = team_df.apply(lambda row: opp_gm_map.get(f"{row['GAME_ID']}_{row['TEAM_ID']}", None), axis=1)

        # Join player_df with team_df on TEAM_ID and GAME_ID, prefix team stats with tm_
        tm_cols = [col for col in team_df.columns if col not in ['GAME_ID', 'TEAM_ID', 'season', 'season_type']]
        tm_rename = {col: f"tm_{col}" for col in tm_cols}
        team_df_tm = team_df.rename(columns=tm_rename)
        player_df = player_df.merge(team_df_tm, on=['GAME_ID', 'TEAM_ID'], how='left', suffixes=('', '_tm'))

        # Join again on opp_id, prefix with opp_
        opp_rename = {col: f"opp_{col}" for col in tm_cols}
        team_df_opp = team_df.rename(columns=opp_rename)
        team_df_opp = team_df_opp.rename(columns={'TEAM_ID': 'opp_id'})
        player_df = player_df.merge(team_df_opp, left_on=['GAME_ID', 'tm_opp_id'], right_on=['GAME_ID', 'opp_id'], how='left', suffixes=('', '_opp'))

    # Derive is_home from MATCHUP once so downstream stages don't need text parsing
    if "MATCHUP" in player_df.columns:
        player_df["is_home"] = (~player_df["MATCHUP"].str.contains("@", regex=False)).astype("int8")

    # Drop redundant columns
    redundant_cols = [
        'tm_VIDEO_AVAILABLE', 'tm_SEASON_ID', 'tm_TEAM_ABBREVIATION', 'tm_TEAM_NAME',
        'tm_GAME_DATE', 'tm_MATCHUP', 'tm_WL', 'season_tm', 'season_type_tm', 'tm_opp_id',
        'opp_VIDEO_AVAILABLE', 'opp_SEASON_ID', 'opp_GAME_DATE', 'opp_MATCHUP', 'opp_WL',
        'season_opp', 'season_type_opp', 'opp_opp_id'
    ]
    player_df = player_df.drop(columns=[col for col in redundant_cols if col in player_df.columns], errors='ignore')

    # Save the cleaned data
    cleaned_filename = "cleaned_player_data"
    save_data(player_df, output_dir, cleaned_filename, output_format)


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