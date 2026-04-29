"""
Summarize the percentage of games with fully-missing data for every stat,
broken down by season and season type.

A game is counted as "missing" for a given stat when every player row for
that game has NaN for that stat.

Output: validations/missing_stats_by_season.csv
        columns: season, season_type, stat, total_games, missing_games, pct_missing

Usage:
    python research/imputation_research.py
    python research/imputation_research.py --data-dir data --out validations
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ── parameters ────────────────────────────────────────────────────────────────
STAT_COLS = [
    "MIN",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS", "FANTASY_PTS",
]

COMBINED_FILE = "all_players.csv"
OUTPUT_FILE   = "missing_stats_by_season.csv"


# ── helpers ───────────────────────────────────────────────────────────────────
def load_combined(data_dir: Path) -> pd.DataFrame:
    combined_path = data_dir / COMBINED_FILE
    if combined_path.exists():
        print(f"Loading {combined_path} ...", flush=True)
        return pd.read_csv(combined_path, low_memory=False)

    print(f"{combined_path} not found — reading individual files ...", flush=True)
    pattern = str(data_dir / "*_players.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No *_players.csv files found in {data_dir}")

    dfs = []
    for f in files:
        m = re.match(r"^(\d{4}-\d{2})_(.+)_players\.csv$", os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f, low_memory=False)
        df["season"]      = m.group(1)
        df["season_type"] = m.group(2)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"Saved {len(combined):,} rows -> {combined_path}", flush=True)
    return combined


def missing_pct_for_stat(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    if stat not in df.columns:
        return pd.DataFrame(columns=["season", "season_type", "total_games",
                                     "missing_games", "pct_missing"])

    game_missing = (
        df.groupby(["season", "season_type", "GAME_ID"])[stat]
        .apply(lambda x: x.isna().all())
        .reset_index(name="missing")
    )
    summary = (
        game_missing
        .groupby(["season", "season_type"])
        .agg(total_games=("GAME_ID", "count"),
             missing_games=("missing", "sum"))
        .assign(pct_missing=lambda d: (d.missing_games / d.total_games * 100).round(2))
        .reset_index()
    )
    summary.insert(2, "stat", stat)
    return summary


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing *_players.csv files (default: data)")
    parser.add_argument("--out", default="validations",
                        help="Directory for output CSV (default: validations)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined(data_dir)

    present_stats = [s for s in STAT_COLS if s in df.columns]
    missing_stats = [s for s in STAT_COLS if s not in df.columns]
    if missing_stats:
        print(f"Warning: columns not found in data and skipped: {missing_stats}")

    print(f"Computing missing-game rates for {len(present_stats)} stats ...", flush=True)
    parts = [missing_pct_for_stat(df, stat) for stat in present_stats]
    result = pd.concat(parts, ignore_index=True)

    result = result.sort_values(["season", "season_type", "stat"]).reset_index(drop=True)

    out_path = out_dir / OUTPUT_FILE
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result):,} rows -> {out_path}")

    # print a readable pivot: rows = season+stype, cols = stat pct_missing
    pivot = result.pivot_table(
        index=["season", "season_type"],
        columns="stat",
        values="pct_missing",
        aggfunc="first",
    )
    pivot = pivot[[s for s in STAT_COLS if s in pivot.columns]]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)
    print()
    print(pivot.to_string())


if __name__ == "__main__":
    main()
