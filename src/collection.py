"""NBA data collection module.

This module handles fetching data from the NBA API and saving it to disk.
"""

from __future__ import annotations

import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    BoxScoreHustleV2,
    BoxScoreMatchupsV3,
    BoxScoreMiscV2,
    BoxScorePlayerTrackV3,
    BoxScoreTraditionalV3,
    GameRotation,
    LeagueDashPlayerBioStats,
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    LeagueLeaders,
    PlayByPlayV3,
    PlayerIndex,
    leaguegamelog,
)

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
    "PlayIn": "PlayIn",
    "Play-In": "PlayIn",
    "IST": "IST",
    "NBA Cup": "IST",
}

PLAYER_OR_TEAM_MAP = {
    "players": "P",
    "teams": "T",
}

# Earliest season start year each season type is available
SEASON_TYPE_MIN_YEAR: dict[str, int] = {
    "Preseason": 2003,
    "PlayIn":    2020,
    "IST":       2023,
}

ALL_PIPELINE_STAGES = ["import", "patch", "import_validate", "clean", "impute", "features", "gpm"]

# Game-level data types: name → (endpoint_class, min_season_year_or_None)
# min_season_year: earliest season start year supported (None = no restriction)
GAME_LEVEL_DATA_TYPES: dict[str, tuple] = {
    "boxscore_quarters":    (BoxScoreTraditionalV3, 1996),
    "boxscore_matchups":    (BoxScoreMatchupsV3, 2013),
    "boxscore_playertrack": (BoxScorePlayerTrackV3, 2013),
    "boxscore_misc":        (BoxScoreMiscV2, 2013),
    "boxscore_hustle":      (BoxScoreHustleV2, 2013),
    "game_rotation":        (GameRotation, 2005),
    "playbyplay":           (PlayByPlayV3, 2019),
}

# Season-level data types only available from a certain year.
# Types not listed here are attempted for all seasons.
SEASON_DATA_TYPE_MIN_YEAR: dict[str, int] = {
    "team_season_stats": 1996,  # LeagueDashTeamStats; pre-97 covered by patch stage
    "player_bios":       1996,  # LeagueDashPlayerBioStats; PBP era only
}


# ---------------------------------------------------------------------------
# Adaptive rate-limit delay
# ---------------------------------------------------------------------------

class _AdaptiveDelay:
    """Per-worker adaptive delay with exponential backoff on failures.

    Success path: self-tunes toward the fastest recently-proven-safe delay.
    Failure path: exp(cons_failures**1.25 * (1+rand)) — raises RuntimeError
                  after max_failures consecutive failures.
    """

    def __init__(self, max_failures: int = 5):
        self.cons_failures = 0
        self.last_pause = 0.5
        self._recent: deque = deque(maxlen=10)
        self.max_failures = max_failures

    def after_success(self) -> float:
        rand = random.uniform(0, 1)
        rand2 = 0.0572 + random.uniform(0, 0.1) + random.uniform(0, 0.85)
        base = self.last_pause / (1 + rand)
        if self._recent:
            base = max(base, rand2 * min(self._recent))
        pause = max(base, 0.1 + random.uniform(0, 0.1))
        self.last_pause = pause
        self._recent.append(pause)
        self.cons_failures = 0
        return pause

    def after_failure(self) -> float:
        self.cons_failures += 1
        if self.cons_failures >= self.max_failures:
            raise RuntimeError(f"{self.cons_failures} consecutive failures")
        rand = random.uniform(0, 1)
        rand2 = 0.0572 + random.uniform(0, 0.1) + random.uniform(0, 0.85)
        pause = np.exp(self.cons_failures ** 1.25 * (1 + rand)) - rand2
        self.last_pause = pause
        return pause


# ---------------------------------------------------------------------------
# Season-level fetch functions
# ---------------------------------------------------------------------------

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


def fetch_roster_data(season: str) -> Optional[pd.DataFrame]:
    """Fetch NBA player roster data for all active players in a single call."""
    print(f"Fetching rosters for {season}...")

    try:
        result = PlayerIndex(season=season)
        df = result.get_data_frames()[0]
        df = df.rename(columns={"PERSON_ID": "PLAYER_ID"})
        print(f"  Retrieved {len(df):,} roster records")
        return df
    except Exception as error:
        print(f"  Error fetching rosters for {season}: {error}")
        return None


def fetch_player_season_stats(season: str, season_type: str) -> Optional[pd.DataFrame]:
    """Fetch NBA player season stats data from the API."""
    api_season_type = VALID_SEASON_TYPES.get(season_type, season_type)
    print(f"Fetching player season stats for {season} ({season_type})...")

    # LeagueDashPlayerStats only has data from 1996-97 onward (PBP era).
    # LeagueLeaders covers earlier seasons.
    use_leaders = int(season[:4]) < 1996

    try:
        if use_leaders:
            stats = LeagueLeaders(
                season=season,
                season_type_all_star=api_season_type,
                per_mode48="Totals",
                stat_category_abbreviation="PTS",
            )
        else:
            stats = LeagueDashPlayerStats(
                season=season,
                season_type_all_star=api_season_type,
                per_mode_detailed="Totals",
            )
        df = stats.get_data_frames()[0]
        print(f"  Retrieved {len(df):,} player season records")
        return df
    except Exception as error:
        print(f"  Error fetching player season stats for {season} ({season_type}): {error}")
        return None


def fetch_team_season_stats(season: str, season_type: str) -> Optional[pd.DataFrame]:
    """Fetch NBA team season stats data from the API."""
    api_season_type = VALID_SEASON_TYPES.get(season_type, season_type)
    print(f"Fetching team season stats for {season} ({season_type})...")

    try:
        stats = LeagueDashTeamStats(
            season=season,
            season_type_all_star=api_season_type,
            per_mode_detailed="Totals",
        )
        df = stats.get_data_frames()[0]
        print(f"  Retrieved {len(df):,} team season records")
        return df
    except Exception as error:
        print(f"  Error fetching team season stats for {season} ({season_type}): {error}")
        return None


# ---------------------------------------------------------------------------
# Game-level fetch functions
# ---------------------------------------------------------------------------

def get_game_ids_for_season(data_dir: Path, season: str, season_type: str, output_format: str) -> list[str]:
    """Read game IDs from an already-collected player game log file."""
    filepath = data_dir / f"{season}_{season_type}_players.{output_format}"
    if not filepath.exists():
        return []
    if output_format == "csv":
        df = pd.read_csv(filepath, usecols=["GAME_ID"])
    else:
        df = pd.read_parquet(filepath, columns=["GAME_ID"])
    return sorted(df["GAME_ID"].astype(str).str.zfill(10).unique().tolist())


def fetch_boxscore_quarters(game_id: str) -> Optional[pd.DataFrame]:
    """Fetch per-quarter player stats for a game, including OT periods.

    range_type='1' activates period filtering; without it the API ignores
    start_period/end_period and returns full-game totals for every period.
    A non-existent period (e.g. OT on a regulation game) raises an error,
    which is the natural loop terminator.
    """
    periods = []
    for period in range(1, 15):
        try:
            result = BoxScoreTraditionalV3(
                game_id=game_id,
                range_type="1",
                start_period=str(period),
                end_period=str(period),
            )
            df = result.get_data_frames()[0]
            if df.empty:
                break
            df = df.copy()
            df["period"] = period
            periods.append(df)
            time.sleep(0.3)
        except Exception:
            break
    if not periods:
        raise RuntimeError(f"No period data returned for game {game_id}")
    return pd.concat(periods, ignore_index=True)


def fetch_game_level_data(game_id: str, data_type: str) -> Optional[pd.DataFrame]:
    """Fetch a single game's data for the given data type."""
    if data_type == "boxscore_quarters":
        return fetch_boxscore_quarters(game_id)

    endpoint_class, _ = GAME_LEVEL_DATA_TYPES[data_type]
    result = endpoint_class(game_id=game_id)
    if data_type == "game_rotation":
        tables = [df for df in result.get_data_frames() if not df.empty]
        return pd.concat(tables, ignore_index=True) if tables else None
    df = result.get_data_frames()[0]
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

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


def _run_season_game_level(
    season: str,
    season_type: str,
    applicable_types: list[str],
    output_dir: Path,
    output_format: str,
    overwrite: bool,
    max_failures: int,
) -> None:
    """Worker: fetch all game-level data for one (season, season_type) batch."""
    delay = _AdaptiveDelay(max_failures=max_failures)

    game_ids = get_game_ids_for_season(output_dir, season, season_type, output_format)
    if not game_ids:
        return

    random.shuffle(game_ids)
    print(f"{season} {season_type}: {len(game_ids)} games")

    for game_id in game_ids:
        for data_type in applicable_types:
            game_dir = output_dir / "game_level" / season / season_type / data_type
            filepath = game_dir / f"{game_id}.{output_format}"
            if filepath.exists() and not overwrite:
                continue

            try:
                df = fetch_game_level_data(game_id, data_type)
                if df is not None and len(df) > 0:
                    game_dir.mkdir(parents=True, exist_ok=True)
                    if output_format == "csv":
                        df.to_csv(filepath, index=False)
                    else:
                        df.to_parquet(filepath, index=False)
                time.sleep(delay.after_success())
            except Exception as exc:
                print(f"    Error {data_type} {game_id}: {exc}")
                try:
                    pause = delay.after_failure()
                    print(f"    Backing off {pause:.1f}s ({delay.cons_failures} consecutive)")
                    time.sleep(pause)
                except RuntimeError as fatal:
                    print(f"    {fatal} — aborting {season} {season_type}")
                    return


def run_import_stage(job: Dict[str, Any], overwrite: bool = False) -> None:
    """Run season-level and game-level data collection in one pass.

    Season-level types (players, teams, rosters, player_season_stats,
    team_season_stats) are fetched per-season / per-season-type.
    Game-level types (boxscore_matchups, game_rotation, playbyplay) are
    fetched per-game-id. Existing files are skipped unless overwrite=True.
    """
    output_dir = get_output_directory(job)
    output_format = get_output_format(job)
    seasons = build_season_plan(job)
    season_types = list(job["season_types"])
    data_types = list(job["data_types"])

    game_data_types = [dt for dt in data_types if dt in GAME_LEVEL_DATA_TYPES]
    season_data_types = [dt for dt in data_types if dt not in GAME_LEVEL_DATA_TYPES]

    print(f"Running import stage for {len(seasons)} season(s): {seasons[0]}–{seasons[-1]}")
    print(f"Season-level types: {season_data_types}")
    print(f"Game-level types:   {game_data_types}")
    print(f"Output format: {output_format}")
    print(f"Overwrite: {overwrite}\n")

    for season in seasons:
        season_year = int(season[:4])

        # --- Season-level ---
        if "rosters" in season_data_types:
            filepath = output_dir / f"{season}_rosters.{output_format}"
            if not filepath.exists() or overwrite:
                df = fetch_roster_data(season)
                if df is not None:
                    save_data(df, output_dir, f"{season}_rosters", output_format)
            else:
                print(f"  Skipping {filepath.name} (already exists)")

        for season_type in season_types:
            min_year = SEASON_TYPE_MIN_YEAR.get(season_type)
            if min_year is not None and season_year < min_year:
                continue

            for data_type in season_data_types:
                if data_type == "rosters":
                    continue

                min_data_year = SEASON_DATA_TYPE_MIN_YEAR.get(data_type)
                if min_data_year is not None and season_year < min_data_year:
                    continue

                filepath = output_dir / f"{season}_{season_type}_{data_type}.{output_format}"
                if filepath.exists() and not overwrite:
                    print(f"  Skipping {filepath.name} (already exists)")
                    continue

                if data_type in ["players", "teams"]:
                    df = fetch_league_game_log(season, season_type, PLAYER_OR_TEAM_MAP[data_type])
                elif data_type == "player_bios":
                    df = fetch_player_bio_data(season, season_type)
                elif data_type == "player_season_stats":
                    df = fetch_player_season_stats(season, season_type)
                elif data_type == "team_season_stats":
                    df = fetch_team_season_stats(season, season_type)
                else:
                    print(f"Unknown data type: {data_type}")
                    continue

                if df is not None and len(df) > 0:
                    save_data(df, output_dir, f"{season}_{season_type}_{data_type}", output_format)

    # --- Game-level (parallel by season) ---
    if not game_data_types:
        return

    season_tasks = []
    for season in seasons:
        season_year = int(season[:4])
        applicable = [
            dt for dt in game_data_types
            if GAME_LEVEL_DATA_TYPES[dt][1] is None or season_year >= GAME_LEVEL_DATA_TYPES[dt][1]
        ]
        if not applicable:
            continue
        for season_type in season_types:
            min_year = SEASON_TYPE_MIN_YEAR.get(season_type)
            if min_year is not None and season_year < min_year:
                continue
            season_tasks.append((season, season_type, applicable))

    if not season_tasks:
        return

    max_workers = int(job.get("max_workers", 4))
    max_failures = int(job.get("max_failures", 5))
    print(f"\nGame-level import: {len(season_tasks)} batches across {max_workers} workers\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_season_game_level,
                season, season_type, applicable,
                output_dir, output_format, overwrite, max_failures,
            ): (season, season_type)
            for season, season_type, applicable in season_tasks
        }
        for future in as_completed(futures):
            s, st = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  Worker failed for {s} {st}: {exc}")


def run_job(job: Dict[str, Any]) -> None:
    """Execute the enabled pipeline stages."""
    enabled_stages = get_enabled_stages(job)
    overwrite = bool(job.get("overwrite", False))
    print(f"Enabled pipeline stages: {enabled_stages}\n")

    if "import" in enabled_stages:
        run_import_stage(job, overwrite=overwrite)

    if "patch" in enabled_stages:
        from patch import run_patch_stage
        run_patch_stage(job, overwrite=overwrite)

    if "import_validate" in enabled_stages:
        from validate import run_validation
        run_validation(job)

    if "clean" in enabled_stages:
        run_clean_stage(job)

    if "impute" in enabled_stages:
        from impute import run_impute_stage
        run_impute_stage(job)

    if "features" in enabled_stages:
        from features import run_features_stage
        run_features_stage(job)

    if "gpm" in enabled_stages:
        from gpm import run_gpm_stage
        run_gpm_stage(job)

    print("\nPipeline completed successfully.")


def main(job: Dict[str, Any] | None = None) -> None:
    """Main entry point for the data pipeline."""
    if job is None:
        job = load_job_request()
    run_job(job)
