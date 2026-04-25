# NBA Game Plus Minus

A lightweight NBA data pipeline that uses `nba_api` and `pandas` to fetch league game logs, store them in CSV/Parquet, and provide a clean foundation for data cleaning, modeling, and sharing.

## Project structure

- `run_pipeline.py` — main entry point for the collection pipeline
- `job_request.json` — pipeline configuration for seasons, season types, output format, and output folder
- `src/` — project modules for configuration, collection, and cleaning logic
- `requirements.txt` — runtime dependencies
- `data/` — generated data output (ignored by Git)
- `tests/` — basic coverage for core utilities

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Install development dependencies for tests:

   ```bash
   python -m pip install -r requirements-dev.txt
   ```

## Run the pipeline

```bash
python run_pipeline.py
```

## Customize data collection

Edit `job_request.json` to change the season range, season types, data types, output format, destination folder, or pipeline stages.

A minimal stage configuration example:

```json
{
  "season_start": "2025-26",
  "season_end": "2025-26",
  "season_types": ["Playoffs"],
  "data_types": ["players", "teams", "player_bios"],
  "output_format": "csv",
  "output_dir": "./data",
  "stages": ["collect", "clean"],
  "skip_stages": []
}
```

## Run collect stage only

To run just the collect stage and verify the current extraction logic:

```bash
python run_pipeline.py --stages collect
```

To skip the collect stage (for future pipeline stages like cleaning or modeling):

```bash
python run_pipeline.py --skip-stages collect
```

## Run clean stage only

To run just the clean stage (assumes data is already collected):

```bash
python run_pipeline.py --stages clean
```

## Run tests

After installing the dev requirements:

```bash
pytest -q
```

## Notes

- The `data/` directory is ignored by Git to keep the repository clean.
- The package is intentionally organized under `src/` for a more professional Python layout.
- The code is ready for extension with new data sources, cleaning steps, ML pipelines, and sharing workflows.
