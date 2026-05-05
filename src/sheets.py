"""Push GPM results to Google Sheets after each run.

Setup (one-time):
  1. Go to https://console.cloud.google.com/ and create a project.
  2. Enable "Google Sheets API" and "Google Drive API".
  3. Create a Service Account, download the JSON key.
  4. Set env var:  GOOGLE_SHEETS_CREDS=/path/to/service_account.json
  5. The spreadsheet is created automatically and made publicly readable.
     Share the URL with whoever needs it — it stays fixed across runs.
"""
import os
from pathlib import Path

import pandas as pd

_CREDS_ENV        = "GOOGLE_SHEETS_CREDS"
_SPREADSHEET_NAME = "NBA GPM Results"


def _client():
    import gspread
    from google.oauth2.service_account import Credentials

    creds_path = os.environ.get(_CREDS_ENV)
    if not creds_path or not Path(creds_path).exists():
        raise FileNotFoundError(
            f"Set {_CREDS_ENV} to the path of your service account JSON. "
            "See: https://docs.gspread.org/en/latest/oauth2.html"
        )
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    return gspread.authorize(creds)


def _open_or_create(gc, title: str):
    import gspread
    try:
        return gc.open(title)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(title)
        sh.share(None, perm_type="anyone", role="reader")
        return sh


def _upload_tab(sh, tab_name: str, df: pd.DataFrame) -> None:
    import gspread
    safe = df.fillna("").astype(str)
    rows = [safe.columns.tolist()] + safe.values.tolist()
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=max(len(df) + 20, 100),
                              cols=max(len(df.columns) + 5, 26))
    ws.update(rows)


def push_results(gpm_results: pd.DataFrame, run_comparisons: pd.DataFrame) -> str | None:
    """Upload both tables to Google Sheets. Returns the spreadsheet URL, or None on failure."""
    try:
        gc = _client()
    except Exception as e:
        print(f"  [sheets] Skipped — {e}")
        return None

    try:
        sh = _open_or_create(gc, _SPREADSHEET_NAME)
        _upload_tab(sh, "gpm_results",     gpm_results)
        _upload_tab(sh, "run_comparisons", run_comparisons)
        print(f"  [sheets] Updated → {sh.url}")
        return sh.url
    except Exception as e:
        print(f"  [sheets] Upload failed — {e}")
        return None
