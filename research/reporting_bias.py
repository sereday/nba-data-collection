"""
Reporting bias analysis using the player aggregate logs produced by the
imputation pipeline (validations/player_agg_logs.csv).

For each player-team-season-season_type row the agg logs contain:
  GP_log              — games played (total rows in game logs)
  {stat}_log_sum      — sum of stat across reported games
  {stat}_log_count    — number of games where stat was non-null

This script derives:
  coverage  = log_count / GP_log          (fraction of games where stat was recorded)
  mean      = log_sum   / log_count       (average value when recorded)

Output (all saved to validations/):
  reporting_bias_player.csv   — coverage + mean per player-team-season-season_type
  reporting_bias_season.csv   — coverage aggregated to season x season_type
  reporting_bias_team.csv     — coverage aggregated to team x season x season_type

Usage:
    python research/reporting_bias.py
    python research/reporting_bias.py --agg validations/player_agg_logs.csv --out validations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAT_COLS = [
    "MIN",
    "FGM", "FGA",
    "FG3M", "FG3A",
    "FTM", "FTA",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
]


def build_coverage(agg: pd.DataFrame) -> pd.DataFrame:
    rows = agg.copy()
    for stat in STAT_COLS:
        cnt_col = f"{stat}_log_count"
        sum_col = f"{stat}_log_sum"
        if cnt_col not in rows.columns:
            continue
        rows[f"{stat}_coverage"] = (rows[cnt_col] / rows["GP_log"].replace(0, np.nan)).round(4)
        if sum_col in rows.columns:
            rows[f"{stat}_mean"] = (rows[sum_col] / rows[cnt_col].replace(0, np.nan)).round(4)

    keep = (
        ["PLAYER_ID", "TEAM_ID", "season", "season_type", "GP_log"]
        + [c for c in rows.columns if c.endswith("_coverage") or c.endswith("_mean")]
    )
    return rows[[c for c in keep if c in rows.columns]]


def season_summary(player_df: pd.DataFrame) -> pd.DataFrame:
    cov_cols = [c for c in player_df.columns if c.endswith("_coverage")]
    # weight by GP_log so high-minute seasons count more
    def wavg(df, col):
        w = df["GP_log"]
        return (df[col] * w).sum() / w.sum() if w.sum() > 0 else np.nan

    records = []
    for (season, stype), grp in player_df.groupby(["season", "season_type"]):
        row = {"season": season, "season_type": stype,
               "total_player_seasons": len(grp),
               "total_GP": grp["GP_log"].sum()}
        for col in cov_cols:
            row[col] = round(wavg(grp, col), 4)
        records.append(row)
    return pd.DataFrame(records).sort_values(["season", "season_type"]).reset_index(drop=True)


def team_summary(player_df: pd.DataFrame) -> pd.DataFrame:
    cov_cols = [c for c in player_df.columns if c.endswith("_coverage")]
    def wavg(df, col):
        w = df["GP_log"]
        return (df[col] * w).sum() / w.sum() if w.sum() > 0 else np.nan

    records = []
    for (team, season, stype), grp in player_df.groupby(["TEAM_ID", "season", "season_type"]):
        row = {"TEAM_ID": team, "season": season, "season_type": stype,
               "player_seasons": len(grp),
               "total_GP": grp["GP_log"].sum()}
        for col in cov_cols:
            row[col] = round(wavg(grp, col), 4)
        records.append(row)
    return pd.DataFrame(records).sort_values(["season", "season_type", "TEAM_ID"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agg", default="validations/player_agg_logs.csv",
                        help="Path to player_agg_logs.csv (default: validations/player_agg_logs.csv)")
    parser.add_argument("--out", default="validations",
                        help="Output directory (default: validations)")
    args = parser.parse_args()

    agg_path = Path(args.agg)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not agg_path.exists():
        sys.exit(
            f"Agg logs not found at {agg_path}.\n"
            "Run the imputation pipeline first to generate validations/player_agg_logs.csv."
        )

    print(f"Loading {agg_path} ...", flush=True)
    agg = pd.read_csv(agg_path, low_memory=False)
    print(f"  {len(agg):,} rows, {agg['season'].nunique()} seasons", flush=True)

    print("Building coverage + mean per player-team-season ...", flush=True)
    player_df = build_coverage(agg)
    player_df.to_csv(out_dir / "reporting_bias_player.csv", index=False)
    print(f"  -> {out_dir}/reporting_bias_player.csv  ({len(player_df):,} rows)")

    print("Aggregating to season level ...", flush=True)
    s_df = season_summary(player_df)
    s_df.to_csv(out_dir / "reporting_bias_season.csv", index=False)
    print(f"  -> {out_dir}/reporting_bias_season.csv  ({len(s_df):,} rows)")

    print("Aggregating to team level ...", flush=True)
    t_df = team_summary(player_df)
    t_df.to_csv(out_dir / "reporting_bias_team.csv", index=False)
    print(f"  -> {out_dir}/reporting_bias_team.csv  ({len(t_df):,} rows)")

    # quick console summary: season-level AST and STL coverage
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    spot_cols = ["season", "season_type"] + [
        c for c in s_df.columns
        if c.endswith("_coverage") and any(s in c for s in ["AST", "STL", "BLK", "OREB"])
    ]
    print()
    print(s_df[[c for c in spot_cols if c in s_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
