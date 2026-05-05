"""
Data quality summary: per player / season / season_type / stat.

Columns in output:
  PLAYER_ID, TEAM_ID, season, season_type, stat
  pct_not_null      – % of logged games where stat is not null
  avg_nonNull       – mean of non-null logged values
  total_games       – official GP from season stats
  season_avg        – season total / GP  (per-game season average)
  null_games        – total_games - logged non-null count
  inferred_avg_null – (season_total - log_sum) / null_games

Usage:
  python research/data_quality.py
  python research/data_quality.py --agg validations/player_agg_logs.csv \
      --season data/player_season_stats_combined.csv \
      --out results/data_quality_summary.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
         "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

JOIN_KEYS = ["PLAYER_ID", "TEAM_ID", "season", "season_type"]


def _load_season_stats_combined(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        ext = os.path.splitext(path)[1]
        return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path, low_memory=False)

    # Fallback: load from individual files under data/
    data_dir = os.path.join(_ROOT, "data")
    frames = []
    for fname in os.listdir(data_dir):
        if "player_season_stats" in fname and not fname.startswith("."):
            fpath = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(fpath, low_memory=False) if fname.endswith(".csv") else pd.read_parquet(fpath)
                parts = fname.replace("_player_season_stats", "").split("_")
                # filename: {season}_{season_type}_player_season_stats.csv
                # season like 1970-71, season_type like Regular/Playoffs
                df["season"] = parts[0]
                df["season_type"] = "_".join(parts[1:]).split(".")[0]
                frames.append(df)
            except Exception as e:
                warnings.warn(f"Skipping {fname}: {e}")
    if not frames:
        raise FileNotFoundError(f"No season stats found at {path} or in {data_dir}")
    print(f"  Loaded {len(frames)} individual season stats files (combined file not found)")
    return pd.concat(frames, ignore_index=True)


def build_summary(agg_path: str, season_path: str) -> pd.DataFrame:
    agg = pd.read_csv(agg_path, low_memory=False)
    season = _load_season_stats_combined(season_path)

    # Standardise join key types
    for df in (agg, season):
        df["PLAYER_ID"] = df["PLAYER_ID"].astype(str)
        df["TEAM_ID"]   = df["TEAM_ID"].astype(str)

    # Keep only the columns we need from season stats
    season_stat_cols = [s for s in STATS if s in season.columns]
    name_col = ["PLAYER"] if "PLAYER" in season.columns else []
    season_keep = JOIN_KEYS + name_col + ["GP"] + season_stat_cols
    season = season[[c for c in season_keep if c in season.columns]].copy()
    season = season.rename(columns={s: f"{s}_season_total" for s in season_stat_cols})
    season = season.rename(columns={"GP": "GP_season"})
    if name_col:
        season = season.rename(columns={"PLAYER": "player_name"})

    merged = agg.merge(season, on=JOIN_KEYS, how="inner")
    print(f"  Merged: {len(merged):,} rows (agg={len(agg):,}, season={len(season):,})")

    rows = []
    for stat in STATS:
        log_sum_col   = f"{stat}_log_sum"
        log_count_col = f"{stat}_log_count"
        season_col    = f"{stat}_season_total"

        if log_count_col not in merged.columns:
            continue

        name_cols = ["player_name"] if "player_name" in merged.columns else []
        sub = merged[JOIN_KEYS + name_cols + ["GP_log", "GP_season"]].copy()
        sub["stat"] = stat

        log_count  = merged[log_count_col].fillna(0)
        log_sum    = merged[log_sum_col].fillna(0) if log_sum_col in merged.columns else pd.Series(0, index=merged.index)
        gp_season  = merged["GP_season"].fillna(0)
        gp_log     = merged["GP_log"].fillna(0)

        def _div(num, den):
            num = pd.to_numeric(num, errors="coerce")
            den = pd.to_numeric(den, errors="coerce")
            return (num / den.replace(0, np.nan)).round(4)

        # Use GP_season as the consistent denominator for all coverage metrics
        season_total  = pd.to_numeric(merged[season_col], errors="coerce") if season_col in merged.columns else pd.Series(np.nan, index=merged.index)
        null_games    = (pd.to_numeric(gp_season, errors="coerce") - pd.to_numeric(log_count, errors="coerce")).clip(lower=0)
        missing_total = (season_total - log_sum)

        sub["pct_not_null"]      = _div(log_count * 100, gp_season).clip(0, 100).round(1)
        sub["avg_nonNull"]       = _div(log_sum, log_count)
        sub["total_games"]       = gp_season
        sub["season_avg"]        = _div(season_total, gp_season)
        sub["null_games"]        = null_games
        sub["inferred_avg_null"] = _div(missing_total, null_games).clip(lower=0)

        rows.append(sub)

    result = pd.concat(rows, ignore_index=True)
    col_order = JOIN_KEYS + ["player_name", "stat", "pct_not_null", "avg_nonNull",
                             "total_games", "season_avg", "null_games", "inferred_avg_null"]
    return result[[c for c in col_order if c in result.columns]].sort_values(
        JOIN_KEYS + ["stat"]
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agg",    default=os.path.join(_ROOT, "validations", "player_agg_logs.csv"))
    parser.add_argument("--season", default=os.path.join(_ROOT, "data", "player_season_stats_combined.csv"))
    parser.add_argument("--out",    default=os.path.join(_ROOT, "results", "data_quality_summary.csv"))
    args = parser.parse_args()

    for path, label in [(args.agg, "--agg")]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("Building data quality summary...")
    summary = build_summary(args.agg, args.season)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Saved {len(summary):,} rows → {args.out}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
