"""Configuration utilities for the NBA data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_OUTPUT_FORMAT = "csv"
DEFAULT_PIPELINE_STAGES = ["collect"]


def normalize_stage(stage: Any) -> str:
    """Normalize a stage name to lowercase and stripped."""
    return str(stage).strip().lower()


def load_job_request(filepath: str = "job_request.json") -> Dict[str, Any]:
    """Load configuration from a JSON job request file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Job request file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_output_directory(job: Dict[str, Any]) -> Path:
    """Get the output directory from job config."""
    return Path(job.get("output_dir", DEFAULT_OUTPUT_DIR))


def get_output_format(job: Dict[str, Any]) -> str:
    """Get the output format from job config."""
    return str(job.get("output_format", DEFAULT_OUTPUT_FORMAT)).lower()


def get_requested_stages(job: Dict[str, Any]) -> list[str]:
    """Get the requested pipeline stages from job config."""
    stages = job.get("stages", DEFAULT_PIPELINE_STAGES)
    return [normalize_stage(stage) for stage in stages]


def get_skipped_stages(job: Dict[str, Any]) -> list[str]:
    """Get the stages to skip from job config."""
    return [normalize_stage(stage) for stage in job.get("skip_stages", [])]


def generate_seasons(start_season: str, end_season: str) -> list[str]:
    """Generate an inclusive season list from start to end."""
    start_year = int(start_season.split("-")[0])
    end_year = int(end_season.split("-")[0])
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def build_season_plan(job: Dict[str, Any]) -> list[str]:
    """Build the list of seasons to process from job config, most recent first."""
    if "season_start" in job and "season_end" in job:
        return list(reversed(generate_seasons(job["season_start"], job["season_end"])))
    if "seasons" in job:
        return list(reversed(job["seasons"]))
    raise ValueError("job_request.json must define either season_start/season_end or seasons")
