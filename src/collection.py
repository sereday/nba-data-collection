"""NBA data collection module.

This module handles fetching data from the NBA API and saving it to disk.
"""

from __future__ import annotations

import os
import random
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    HustleStatsBoxScore,
    BoxScoreMatchupsV3,
    BoxScoreMiscV3,
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
    "boxscore_misc":        (BoxScoreMiscV3, 2013),
    "boxscore_hustle":      (HustleStatsBoxScore, 2013),
    "game_rotation":        (GameRotation, 2005),
    "playbyplay":           (PlayByPlayV3, 2019),
}

# Game-level data types excluded from certain season types entirely.
GAME_LEVEL_EXCLUDED_SEASON_TYPES: dict[str, set] = {
    "boxscore_matchups":    {"Preseason"},
    "boxscore_playertrack": {"Preseason"},
    "boxscore_hustle":      {"Preseason"},
}

# Season-level data types only available from a certain year.
# Types not listed here are attempted for all seasons.
SEASON_DATA_TYPE_MIN_YEAR: dict[str, int] = {
    "players":             1996,  # LeagueGameLog (players); pre-97 covered by patch stage
    "teams":               1996,  # LeagueGameLog (teams); pre-97 covered by patch stage
    "player_season_stats": 1996,  # LeagueDashPlayerStats; pre-97 covered by patch stage
    "team_season_stats":   1996,  # LeagueDashTeamStats; pre-97 covered by patch stage
    "player_bios":         1996,  # LeagueDashPlayerBioStats; PBP era only
}


# ---------------------------------------------------------------------------
# Adaptive rate-limit delay
# ---------------------------------------------------------------------------

class _AbortError(Exception):
    """Raised to stop a worker immediately — abort takes priority over all other handling."""


class _VpnSwitcher:
    """Disconnect/reconnect ProtonVPN via CLI. Assumes 'protonvpn-cli' is on PATH."""

    def __init__(self, max_switches: int = 5):
        self._lock = threading.Lock()
        self._count = 0
        self.max_switches = max_switches

    @property
    def exhausted(self) -> bool:
        return self._count >= self.max_switches

    def switch(self) -> bool:
        """Disconnect then reconnect ProtonVPN to a random server for a fresh IP. Returns True on success."""
        with self._lock:
            if self._count >= self.max_switches:
                print(f"  VPN: max switches ({self.max_switches}) exhausted")
                return False
            self._count += 1
            n = self._count

        print(f"  VPN switch #{n}/{self.max_switches}: disconnecting...")
        try:
            subprocess.run(["protonvpn-cli", "disconnect"], timeout=15, capture_output=True, check=False)
            time.sleep(2)
            print(f"  VPN switch #{n}/{self.max_switches}: connecting to new server...")
            subprocess.run(["protonvpn-cli", "connect", "--random"], timeout=30, capture_output=True, check=False)
            time.sleep(5)
            print(f"  VPN switch #{n}/{self.max_switches}: connected")
            return True
        except FileNotFoundError:
            print("  VPN switch failed: 'protonvpn-cli' not found — is ProtonVPN CLI installed and on PATH?")
            return False
        except subprocess.TimeoutExpired:
            print(f"  VPN switch #{n}: timed out")
            return False
        except Exception as exc:
            print(f"  VPN switch #{n}: {exc}")
            return False


class _SharedRateState:
    """Shared across all workers — owns failure counting and backoff timing.

    Failure path: exp(cons_failures**1.25 * (1+rand)) backoff, raised as
                  _AbortError after max_failures consecutive failures.
    Pile-up guard: if a backoff is already in effect when a new failure arrives,
                   cons_failures is not incremented (same rate-limit event).
    VPN switch: at vpn_switch_threshold consecutive failures, disconnect/reconnect
                NordVPN and reset all counters (fresh IP = fresh slate).
    Global abort: if cumulative failures exceed successes and no VPN switcher is
                  configured, all workers are stopped immediately via os._exit(1).
    """

    def __init__(self, max_failures: int = 5, vpn_switch_threshold: int = 3,
                 vpn_switcher: Optional[_VpnSwitcher] = None):
        self._lock = threading.Lock()
        self.cons_failures = 0
        self.max_failures = max_failures
        self._pause_until = 0.0
        self._total_successes = 0
        self._total_failures = 0
        self._aborted = False
        self._vpn_switch_threshold = vpn_switch_threshold
        self._vpn_switcher = vpn_switcher

    def sleep(self, seconds: float) -> None:
        """Success-path sleep, interruptible if a backoff fires or abort triggers."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._aborted:
                raise _AbortError("Aborted: cumulative failures exceed successes")
            if self._pause_until > time.time():
                self._wait_backoff()
                return
            time.sleep(min(0.25, deadline - time.time()))

    def check_pause(self) -> None:
        """Block at the top of each fetch loop iteration if backing off; raise if aborted."""
        if self._aborted:
            raise _AbortError("Aborted: cumulative failures exceed successes")
        if self._pause_until > time.time():
            self._wait_backoff()

    def _wait_backoff(self) -> None:
        while self._pause_until > time.time():
            if self._aborted:
                raise _AbortError("Aborted: cumulative failures exceed successes")
            time.sleep(0.25)

    def report_success(self) -> None:
        with self._lock:
            self._total_successes += 1
            if time.time() >= self._pause_until:
                self.cons_failures = 0

    def report_failure(self) -> tuple[float, bool]:
        """Returns (pause, is_new) — is_new=False means already in backoff (pile-up).
        Raises _AbortError at max_failures or when VPN switches are exhausted.
        At vpn_switch_threshold consecutive failures, triggers a VPN server switch."""
        _try_vpn = False

        with self._lock:
            self._total_failures += 1
            if self._total_failures > self._total_successes and self._vpn_switcher is None:
                self._aborted = True
                print(
                    f"\nABORT: {self._total_failures} cumulative failures exceed "
                    f"{self._total_successes} successes — stopping all workers"
                )
                os._exit(1)
            if time.time() < self._pause_until:
                return self._pause_until - time.time(), False
            self.cons_failures += 1
            if self._vpn_switcher is not None and self.cons_failures >= self._vpn_switch_threshold:
                _try_vpn = True
            elif self.cons_failures >= self.max_failures:
                raise _AbortError(f"{self.cons_failures} consecutive failures")
            else:
                rand = random.uniform(0, 1)
                rand2 = 0.0572 + random.uniform(0, 0.1) + random.uniform(0, 0.85)
                pause = np.exp(self.cons_failures ** 1.25 * (1 + rand)) - rand2
                self._pause_until = time.time() + pause
                return pause, True

        # VPN switch outside the lock — takes several seconds
        switched = self._vpn_switcher.switch()  # type: ignore[union-attr]
        with self._lock:
            if switched:
                self._total_failures = 0
                self._total_successes = 0
                self.cons_failures = 0
                self._pause_until = 0.0
                return 0.0, True
            # Switch failed — fall back to backoff or abort
            if self.cons_failures >= self.max_failures or self._vpn_switcher.exhausted:
                raise _AbortError(
                    f"VPN switch failed after {self.cons_failures} consecutive failures"
                )
            rand = random.uniform(0, 1)
            rand2 = 0.0572 + random.uniform(0, 0.1) + random.uniform(0, 0.85)
            pause = np.exp(self.cons_failures ** 1.25 * (1 + rand)) - rand2
            self._pause_until = time.time() + pause
            return pause, True


class _WorkerDelay:
    """Per-worker success-path delay, self-tuning toward fastest proven-safe pace."""

    def __init__(self):
        self.last_pause = 0.78
        self._recent: deque = deque(maxlen=10)

    def next_pause(self) -> float:
        rand = random.uniform(0, 1)
        rand2 = 0.0572 + random.uniform(0, 0.1) + random.uniform(0, 0.85)
        base = self.last_pause / (1 + rand)
        if self._recent:
            base = max(base, rand2 * min(self._recent))
        pause = max(base, 0.16 + random.uniform(0, 0.16)) + random.uniform(0.08, 0.24)
        self.last_pause = pause
        self._recent.append(pause)
        return pause


# ---------------------------------------------------------------------------
# Ping log
# ---------------------------------------------------------------------------

class _PingLog:
    """Thread-safe buffer of per-request (duration, success) entries, flushed to CSV."""

    _HEADER = ["timestamp", "season", "season_type", "data_type", "duration_s", "success"]

    def __init__(self, path: Path, flush_every: int = 50):
        self._lock = threading.Lock()
        self._path = path
        self._flush_every = flush_every
        self._buffer: list = []
        self._header_written = path.exists()

    def record(self, season: str, season_type: str, data_type: str, duration_s: float, success: bool) -> None:
        from datetime import datetime
        entry = [datetime.now().isoformat(), season, season_type, data_type, f"{duration_s:.3f}", int(success)]
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._flush_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        import csv
        write_header = not self._header_written
        with open(self._path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(self._HEADER)
                self._header_written = True
            writer.writerows(self._buffer)
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_file(path: Path, min_bytes: int = 1536) -> bool:
    """True if the file exists and is at least min_bytes (not an empty/error save)."""
    return path.exists() and path.stat().st_size >= min_bytes


def _fetch_with_retry(fetch_fn, label: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """Retry fetch_fn() until it returns a non-empty DataFrame.
    Both None (API error) and empty DataFrame are treated as failures.
    Returns the DataFrame on success, None after all retries are exhausted."""
    for attempt in range(max_retries):
        result = fetch_fn()
        if result is not None and len(result) > 0:
            return result
        if attempt < max_retries - 1:
            reason = "empty response" if result is not None else "error"
            delay = 5 * (2 ** attempt) + random.uniform(0, 3)
            if attempt == 0:
                print(f"  {label}: {reason}, retrying in {delay:.1f}s ({attempt + 1}/{max_retries})")
            time.sleep(delay)
    return None


def _log_missing(log_path: Path, season: str, season_type: str, data_type: str) -> None:
    """Append an unexpected missing-data entry to the review log."""
    import csv
    from datetime import datetime
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "season", "season_type", "data_type"])
        writer.writerow([datetime.now().isoformat(), season, season_type, data_type])


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
            timeout=60,
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
            timeout=60,
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
        result = PlayerIndex(season=season, timeout=60)
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
                timeout=60,
            )
        else:
            stats = LeagueDashPlayerStats(
                season=season,
                season_type_all_star=api_season_type,
                per_mode_detailed="Totals",
                timeout=60,
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
            timeout=60,
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
    if not _valid_file(filepath):
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
            if not periods:
                raise  # Period 1 failed — propagate so worker triggers backoff
            break      # Period N+1 error — natural end, return what we have
    if not periods:
        raise RuntimeError(f"No period data returned for game {game_id}")
    return pd.concat(periods, ignore_index=True)


def fetch_game_level_data(game_id: str, data_type: str) -> Optional[pd.DataFrame]:
    """Fetch a single game's data for the given data type."""
    if data_type == "boxscore_quarters":
        return fetch_boxscore_quarters(game_id)

    endpoint_class, _ = GAME_LEVEL_DATA_TYPES[data_type]
    try:
        result = endpoint_class(game_id=game_id)
        if data_type == "game_rotation":
            tables = [df for df in result.get_data_frames() if not df.empty]
            return pd.concat(tables, ignore_index=True) if tables else None
        if data_type == "boxscore_hustle":
            df = result.get_data_frames()[1]  # [0] is status row, [1] is player stats
            return df if not df.empty else None
        df = result.get_data_frames()[0]
        return df if not df.empty else None
    except (KeyError, IndexError):
        return None


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
    data_type: str,
    output_dir: Path,
    output_format: str,
    overwrite: bool,
    shared: _SharedRateState,
    ping_log: _PingLog,
) -> None:
    """Worker: fetch one data_type for all games in one (season, season_type) batch."""
    local = _WorkerDelay()

    game_ids = get_game_ids_for_season(output_dir, season, season_type, output_format)
    if not game_ids:
        return

    random.shuffle(game_ids)
    print(f"{season} {season_type} [{data_type}]: {len(game_ids)} games")

    game_dir = output_dir / "game_level" / season / season_type / data_type
    for game_id in game_ids:
        filepath = game_dir / f"{game_id}.{output_format}"
        if _valid_file(filepath) and not overwrite:
            continue

        t0 = 0.0
        try:
            shared.check_pause()
            t0 = time.time()
            df = fetch_game_level_data(game_id, data_type)
            duration = time.time() - t0
            if df is not None and len(df) > 0:
                game_dir.mkdir(parents=True, exist_ok=True)
                if output_format == "csv":
                    df.to_csv(filepath, index=False)
                else:
                    df.to_parquet(filepath, index=False)
            ping_log.record(season, season_type, data_type, duration, True)
            shared.report_success()
            shared.sleep(local.next_pause())
        except _AbortError as fatal:
            print(f"    {fatal} — aborting {season} {season_type} [{data_type}]")
            return
        except Exception as exc:
            ping_log.record(season, season_type, data_type, time.time() - t0, False)
            print(f"    Error {data_type} {game_id}: {exc}")
            try:
                pause, is_new = shared.report_failure()
                if is_new and pause > 0:
                    print(f"    Backing off {pause:.1f}s ({shared.cons_failures} consecutive)")
            except _AbortError as fatal:
                print(f"    {fatal} — aborting {season} {season_type} [{data_type}]")
                return
            try:
                shared.check_pause()
            except _AbortError as fatal:
                print(f"    {fatal} — aborting {season} {season_type} [{data_type}]")
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

    missing_log = output_dir.parent / "import_missing.csv"

    for season in seasons:
        season_year = int(season[:4])

        # --- Season-level ---
        if "rosters" in season_data_types:
            filepath = output_dir / f"{season}_rosters.{output_format}"
            if not _valid_file(filepath) or overwrite:
                df = _fetch_with_retry(lambda: fetch_roster_data(season), f"{season} rosters")
                if df is not None:
                    save_data(df, output_dir, f"{season}_rosters", output_format)
                else:
                    _log_missing(missing_log, season, "all", "rosters")
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
                if _valid_file(filepath) and not overwrite:
                    print(f"  Skipping {filepath.name} (already exists)")
                    continue

                label = f"{season} {season_type} {data_type}"
                if data_type in ["players", "teams"]:
                    pt = PLAYER_OR_TEAM_MAP[data_type]
                    df = _fetch_with_retry(lambda pt=pt: fetch_league_game_log(season, season_type, pt), label)
                elif data_type == "player_bios":
                    df = _fetch_with_retry(lambda: fetch_player_bio_data(season, season_type), label)
                elif data_type == "player_season_stats":
                    df = _fetch_with_retry(lambda: fetch_player_season_stats(season, season_type), label)
                elif data_type == "team_season_stats":
                    df = _fetch_with_retry(lambda: fetch_team_season_stats(season, season_type), label)
                else:
                    print(f"Unknown data type: {data_type}")
                    continue

                if df is not None:
                    save_data(df, output_dir, f"{season}_{season_type}_{data_type}", output_format)
                else:
                    _log_missing(missing_log, season, season_type, data_type)

    # --- Game-level (parallel by season) ---
    if not game_data_types:
        return

    season_tasks = []
    for season in seasons:
        season_year = int(season[:4])
        for data_type in game_data_types:
            dt_min_year = GAME_LEVEL_DATA_TYPES[data_type][1]
            if dt_min_year is not None and season_year < dt_min_year:
                continue
            excluded_types = GAME_LEVEL_EXCLUDED_SEASON_TYPES.get(data_type, set())
            for season_type in season_types:
                if season_type in excluded_types:
                    continue
                st_min_year = SEASON_TYPE_MIN_YEAR.get(season_type)
                if st_min_year is not None and season_year < st_min_year:
                    continue
                season_tasks.append((season, season_type, data_type))

    if not season_tasks:
        return

    max_workers = int(job.get("max_workers", 4))
    max_failures = int(job.get("max_failures", 5))
    vpn_switch_threshold = int(job.get("vpn_switch_threshold", 3))
    max_vpn_switches = int(job.get("max_vpn_switches", 5))
    vpn_switcher = _VpnSwitcher(max_switches=max_vpn_switches)
    shared = _SharedRateState(
        max_failures=max_failures,
        vpn_switch_threshold=vpn_switch_threshold,
        vpn_switcher=vpn_switcher,
    )
    ping_log = _PingLog(output_dir.parent / "ping_log.csv")
    print(f"\nGame-level import: {len(season_tasks)} batches across {max_workers} workers\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_season_game_level,
                season, season_type, data_type,
                output_dir, output_format, overwrite, shared, ping_log,
            ): (season, season_type, data_type)
            for season, season_type, data_type in season_tasks
        }
        for future in as_completed(futures):
            s, st, dt = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  Worker failed for {s} {st} [{dt}]: {exc}")

    ping_log.flush()
    print(f"Ping log saved to {ping_log._path}")


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
