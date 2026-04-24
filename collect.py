import json
import os
from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import time


def load_job_request(filepath="job_request.json"):
    """Load configuration from job_request.json"""
    with open(filepath, "r") as f:
        return json.load(f)


def fetch_league_game_log(season, season_type, player_or_team):
    """Fetch league game log for a season"""
    print(f"Fetching {player_or_team} game logs for {season} ({season_type})...")
    
    # Map season_type to the correct API parameter
    season_type_map = {
        "Regular": "Regular Season",
        "Playoffs": "Playoffs",
        "Pre Season": "Pre Season",
        "All Star": "All Star"
    }
    
    api_season_type = season_type_map.get(season_type, season_type)
    
    try:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=api_season_type,
            player_or_team_abbreviation=player_or_team
        )
        df = log.get_data_frames()[0]
        print(f"  ✓ Retrieved {len(df)} records")
        return df
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def save_data(df, output_dir, season, season_type, data_type, format="csv"):
    """Save dataframe to file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename = f"{season}_{season_type}_{data_type}.{format}"
    filepath = os.path.join(output_dir, filename)
    
    if format == "csv":
        df.to_csv(filepath, index=False)
    elif format == "parquet":
        df.to_parquet(filepath, index=False)
    
    print(f"  Saved to {filepath}")


def main():
    job = load_job_request()
    
    output_dir = job.get("output_dir", "./data")
    output_format = job.get("output_format", "csv")
    
    for season in job["seasons"]:
        for season_type in job["season_types"]:
            # Players
            if "players" in job["data_types"]:
                df = fetch_league_game_log(season, season_type, "P")
                if df is not None:
                    save_data(df, output_dir, season, season_type, "players", output_format)
                time.sleep(1)  # Be nice to the API
            
            # Teams
            if "teams" in job["data_types"]:
                df = fetch_league_game_log(season, season_type, "T")
                if df is not None:
                    save_data(df, output_dir, season, season_type, "teams", output_format)
                time.sleep(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
