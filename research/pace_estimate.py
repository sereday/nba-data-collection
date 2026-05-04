"""Estimate team season pace from team_season_stats CSVs and save to data/team_season_pace.csv.

Pace proxy: possessions per game = (FGA + 0.44*FTA - OREB + TOV) / GP
This is Oliver's possession estimate applied to season totals.
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "team_season_pace.csv"


def main():
    frames = []
    for p in sorted(DATA_DIR.glob("*_Regular_team_season_stats.csv")):
        m = re.match(r"(\d{4}-\d{2})_Regular", p.name)
        if not m:
            continue
        
        df = pd.read_csv(p)
        # Ensure required keys exist; fill missing historical stats with 0 for the formula
        for col in ["TEAM_ID", "GP", "FGA", "FTA", "OREB", "TOV"]:
            if col not in df.columns:
                df[col] = 0
        
        df = df[["TEAM_ID", "GP", "FGA", "FTA", "OREB", "TOV"]].copy()
        df["season"] = m.group(1)
        frames.append(df)

    if not frames:
        print(f"No team_season_stats files found in {DATA_DIR}")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df["pace_est"] = (all_df["FGA"] + 0.44 * all_df["FTA"] - all_df["OREB"] + all_df["TOV"]) / all_df["GP"]

    result = all_df[["TEAM_ID", "season", "pace_est"]].copy()
    result.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(result)} team-season rows → {OUT_PATH}")
    print(f"  pace_est range: {result['pace_est'].min():.1f} – {result['pace_est'].max():.1f} possessions/game")


if __name__ == "__main__":
    main()
