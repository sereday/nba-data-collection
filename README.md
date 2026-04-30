# NBA Game Plus-Minus

A Python pipeline that collects historical NBA data from the NBA Stats API, cleans and merges it, engineers features for lineup analysis, and estimates player plus-minus using a regularized regression model.

The pipeline covers season-level data back to 1946-47 and game-level data (box scores, play-by-play, rotations) from roughly 2005 onward, depending on data type.

## Pipeline overview

The pipeline runs in sequential stages, each of which can be run independently:

| Stage | Description |
|---|---|
| `import` | Fetch data from the NBA Stats API and write CSV/Parquet files to `data/` |
| `patch` | Backfill pre-1996-97 season stats using career-aggregate endpoints |
| `import_validate` | Check that all expected output files exist and are non-empty |
| `clean` | Merge files by season and season type into unified player/team datasets |
| `impute` | Fill missing values, primarily minutes played for older seasons |
| `features` | Build the design matrix: player participation pivots, team/opponent stats |
| `gpm` | Estimate player plus-minus coefficients using H2O's GLM (requires Java 8+) |

Stages to run and stages to skip are configured in `job_request.json`. The default configuration skips everything except `gpm` so you can run only what you need.

## Project structure

```
nba-game-plus-minus/
├── run_pipeline.py       # Entry point with CLI argument parsing
├── job_request.json      # Pipeline configuration (edit this before running)
├── src/
│   ├── config.py         # Configuration loading and season planning
│   ├── collection.py     # API fetching with rate limiting and threading
│   ├── cleaning.py       # Data merging into unified player/team datasets
│   ├── impute.py         # Missing value imputation
│   ├── patch.py          # Pre-1996 data backfill
│   ├── features.py       # Feature engineering and design matrix construction
│   ├── validate.py       # File validation
│   └── gpm.py            # Plus-minus estimation via H2O GLM
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Test dependencies
├── data/                 # Generated output (gitignored, can be 2GB+ for full history)
└── tests/
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Configuration

All pipeline behavior is controlled by `job_request.json`. Edit it before running.

**Key fields:**

```jsonc
{
  "season_start": "1946-47",   // First season to process
  "season_end":   "2020-21",   // Last season to process
  "season_types": ["Regular", "Playoffs", "Preseason", "PlayIn", "IST"],

  // Season-level: players, teams, rosters, player_season_stats, team_season_stats
  // Game-level:   playbyplay, game_rotation, boxscore_hustle, boxscore_misc,
  //               boxscore_matchups, boxscore_playertrack
  "data_types": ["players", "teams", "rosters", "player_season_stats", "team_season_stats"],

  "output_format": "csv",      // "csv" or "parquet"
  "output_dir": "./data",

  "max_workers": 1,            // Parallel API threads — start at 1
  "base_pause": 6.1346,        // Seconds between requests; increase if throttled
  "max_failures": 7,           // Abort after this many consecutive failures

  "stages":      ["import", "patch", "import_validate", "clean", "impute", "features", "gpm"],
  "skip_stages": ["patch", "import_validate", "clean", "impute", "features", "gpm"]
}
```

## Running the pipeline

```bash
# Run stages as configured in job_request.json
python run_pipeline.py

# Override stages from the command line
python run_pipeline.py --stages import clean
python run_pipeline.py --skip-stages import patch

# Re-fetch files that already exist
python run_pipeline.py --overwrite
```

## Quick starts

**Collect one recent season (season-level only):**

```json
{
  "season_start": "2024-25",
  "season_end":   "2024-25",
  "season_types": ["Regular", "Playoffs"],
  "data_types":   ["players", "teams", "rosters", "player_season_stats", "team_season_stats"],
  "stages":       ["import", "import_validate", "clean"],
  "skip_stages":  []
}
```

**Collect game-level data for recent seasons (box scores + play-by-play):**

```json
{
  "season_start": "2015-16",
  "season_end":   "2020-21",
  "season_types": ["Regular", "Playoffs"],
  "data_types":   ["players", "teams", "playbyplay", "game_rotation",
                   "boxscore_misc", "boxscore_hustle"],
  "stages":       ["import", "import_validate", "clean"],
  "skip_stages":  []
}
```

## Data availability

| Data category | Availability |
|---|---|
| Season-level (players, teams, rosters, stats) | 1946-47 onward |
| Pre-1996-97 season stats | Backfilled from career endpoints via the `patch` stage |
| Play-by-play | ~2005-06 onward |
| Box scores (misc, hustle, matchups, tracking) | Varies; roughly 2015-16 onward for full coverage |
| Game rotations | ~2015-16 onward |

Running the `import_validate` stage after import will flag any missing files before you proceed to cleaning.

## Plus-minus stage (gpm)

The `gpm` stage requires H2O and Java 8+, which are not in `requirements.txt`. Install separately:

```bash
pip install h2o
# Java 8+ must also be available on your PATH
```

The stage loads the design matrix built by `features`, fits a Generalized Linear Model with L2 regularization, and outputs per-player plus-minus coefficients.

## Notes

- `data/` is gitignored. Full history (1946-47 to 2020-21, all data types) can exceed 2 GB.
- Set `max_workers: 1` for initial runs. The NBA API rate-limits aggressively; increasing concurrency without tuning `base_pause` will cause throttling and failures.
- The `overwrite` flag (in config or `--overwrite` on the CLI) controls whether existing files are re-fetched.
- `skip_stages` takes precedence over `stages` — a stage listed in both will be skipped.
