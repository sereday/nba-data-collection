import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_output_directory, get_output_format, get_target_stat

STOP_AFTER_OPTIONS = ["load", "pivots", "join", "matrix"]


def _parse_min(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _build_pivot(df, is_home: int, prefix: str, min_threshold: float, feature_type: str, target_col: str = None, debug_dir: str = None):
    min_col = "MIN_filled" if "MIN_filled" in df.columns else "MIN"
    side = df[df["is_home"] == is_home].copy()
    side["MIN_float"] = side[min_col].apply(_parse_min)

    if feature_type == "pct":
        side["MIN_pct"] = side["MIN_float"] / 48.0
        qualified = side[side["MIN_pct"] >= min_threshold]
    else:
        qualified = side[side["MIN_float"] >= min_threshold]

    qualified = qualified[["GAME_ID", "PLAYER_ID"] + (["MIN_pct"] if feature_type == "pct" else [])].drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])

    all_games   = sorted(qualified["GAME_ID"].unique())
    all_players = sorted(qualified["PLAYER_ID"].unique())
    game_idx    = {g: i for i, g in enumerate(all_games)}
    player_idx  = {p: i for i, p in enumerate(all_players)}

    row_mapped = qualified["GAME_ID"].map(game_idx)
    col_mapped = qualified["PLAYER_ID"].map(player_idx)
    if row_mapped.isna().any() or col_mapped.isna().any():
        bad_games   = qualified.loc[row_mapped.isna(), "GAME_ID"].unique().tolist()
        bad_players = qualified.loc[col_mapped.isna(), "PLAYER_ID"].unique().tolist()
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            tag = f"pivot_{prefix}_is_home{is_home}"
            pd.DataFrame({"GAME_ID": list(game_idx.keys()), "idx": list(game_idx.values())}).to_csv(
                os.path.join(debug_dir, f"{tag}_game_idx.csv"), index=False)
            pd.DataFrame({"PLAYER_ID": list(player_idx.keys()), "idx": list(player_idx.values())}).to_csv(
                os.path.join(debug_dir, f"{tag}_player_idx.csv"), index=False)
            qualified.loc[row_mapped.isna() | col_mapped.isna()].to_csv(
                os.path.join(debug_dir, f"{tag}_unmapped_rows.csv"), index=False)
            print(f"  [debug] saved index dicts and unmapped rows to {debug_dir}/{tag}_*.csv")
        raise ValueError(
            f"_build_pivot index map failed (is_home={is_home}): "
            f"{len(bad_games)} unmapped GAME_ID(s): {bad_games[:5]}, "
            f"{len(bad_players)} unmapped PLAYER_ID(s): {bad_players[:5]}"
        )

    row_idx = row_mapped.values
    col_idx = col_mapped.values

    if feature_type == "pct":
        data = np.zeros((len(all_games), len(all_players)), dtype=np.float32)
        data[row_idx, col_idx] = qualified["MIN_pct"].values
    else:
        data = np.zeros((len(all_games), len(all_players)), dtype=np.int8)
        data[row_idx, col_idx] = 1

    pivot = pd.DataFrame(data, index=all_games, columns=[f"{prefix}_{p}" for p in all_players])
    pivot.index.name = "GAME_ID"
    if target_col is not None:
        pts = side.groupby("GAME_ID")[target_col].first().rename(target_col)
        pivot = pd.concat([pivot, pts], axis=1)
    return pivot


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _save_debug(info: dict, debug_dir: str, filename: str) -> None:
    os.makedirs(debug_dir, exist_ok=True)

    json_path = os.path.join(debug_dir, filename)
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    csv_path = json_path.replace(".json", ".csv")
    flat = _flatten(info)
    pd.DataFrame(flat.items(), columns=["key", "value"]).to_csv(csv_path, index=False)

    print(f"  [debug] saved {csv_path}")


def run_features_stage(job):
    out_dir = get_output_directory(job)
    min_threshold  = float(job["min_threshold"])
    feature_type = job["feature_type"]
    target_stat    = get_target_stat(job)
    debug          = bool(job.get("debug_features", False))
    stop_after     = job.get("debug_stop_after", "matrix")
    debug_dir      = os.path.join(str(out_dir), "..", "debug")

    if stop_after not in STOP_AFTER_OPTIONS:
        raise ValueError(f"debug_stop_after must be one of {STOP_AFTER_OPTIONS}")

    # --- load ---
    imputed = os.path.join(out_dir, "imputed_player_data.csv")
    cleaned = os.path.join(out_dir, "cleaned_player_data.csv")
    src = imputed if os.path.exists(imputed) else cleaned

    needed = {"GAME_ID", "PLAYER_ID", "TEAM_ID", "GAME_DATE", "is_home", "MATCHUP",
              "season", "season_type", "MIN", "MIN_filled", target_stat}
    df = pd.read_csv(src, usecols=lambda c: c in needed)

    if "is_home" not in df.columns:
        if "MATCHUP" not in df.columns:
            raise ValueError("cleaned data has neither 'is_home' nor 'MATCHUP' — re-run the clean stage")
        print("  'is_home' missing — deriving from MATCHUP (re-run clean stage to fix permanently)")
        df["is_home"] = (~df["MATCHUP"].str.contains("@", regex=False)).astype("int8")
    df = df.drop(columns=["MATCHUP"], errors="ignore")

    if "season_type" in df.columns:
        df = df[df["season_type"] != "Preseason"]
    if "season" in df.columns:
        df = df[df["season"].str[:4].astype(int) >= 1951]

    min_games = int(job["min_games"])
    min_total_games = int(job.get("min_total_games", 0))
    if min_games > 0:
        min_col = "MIN_filled" if "MIN_filled" in df.columns else "MIN"
        min_vals = df[min_col].apply(_parse_min)
        if feature_type == "pct":
            qualified_rows = min_vals / 48.0 >= min_threshold
        else:
            qualified_rows = min_vals >= min_threshold

        qualified_counts = df[qualified_rows].groupby("PLAYER_ID")["GAME_ID"].nunique()
        total_counts = df.groupby("PLAYER_ID")["GAME_ID"].nunique()

        # Eligible if qualified games >= min_games OR total games >= min_total_games
        condition = total_counts.index.map(qualified_counts).fillna(0) >= min_games
        if min_total_games > 0:
            condition |= (total_counts >= min_total_games)

        eligible = total_counts[condition].index

        df = df[df["PLAYER_ID"].isin(eligible)]
        print(f"  min_games filter: {len(eligible)} players (qualified >= {min_games} OR total >= {min_total_games})")

    if job.get("pace_adjustment", False):
        pace_path = Path(out_dir) / "team_season_pace.csv"
        if pace_path.exists() and "season" in df.columns:
            pace_df = pd.read_csv(pace_path)
            df = df.merge(pace_df[["TEAM_ID", "season", "pace_est"]], on=["TEAM_ID", "season"], how="left")
            game_pace = df.groupby("GAME_ID")["pace_est"].mean().rename("game_pace")
            df = df.merge(game_pace, on="GAME_ID", how="left")
            valid = df["game_pace"].notna() & (df["game_pace"] > 0)
            df.loc[valid, "tm_PTS"] = df.loc[valid, "tm_PTS"] / df.loc[valid, "game_pace"]
            df = df.drop(columns=["pace_est", "game_pace"])
            print(f"  Pace adjustment applied (tm_PTS ÷ game_pace)")
        else:
            print(f"  Pace adjustment skipped: {pace_path} not found — run: python research/pace_estimate.py")

    if debug:
        _save_debug({
            "step": "load",
            "source": src,
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "missing_needed": list(needed - set(df.columns)),
            "games": int(df["GAME_ID"].nunique()),
            "players": int(df["PLAYER_ID"].nunique()),
            "is_home_counts": df["is_home"].value_counts().to_dict() if "is_home" in df.columns else "missing",
        }, debug_dir, "features_01_load.json")

    if stop_after == "load":
        print("Stopped after: load")
        return

    # --- pivots ---
    # Home row:  O_ = home qualified players,  D_ = road qualified players,  target = home pts
    # Road row:  O_ = road qualified players,  D_ = home qualified players,  target = road pts
    home_off = _build_pivot(df, is_home=1, prefix="O", target_col="tm_PTS", min_threshold=min_threshold, feature_type=feature_type, debug_dir=debug_dir)
    road_off = _build_pivot(df, is_home=0, prefix="O", target_col="tm_PTS", min_threshold=min_threshold, feature_type=feature_type, debug_dir=debug_dir)
    home_def = _build_pivot(df, is_home=0, prefix="D", min_threshold=min_threshold, feature_type=feature_type, debug_dir=debug_dir)
    road_def = _build_pivot(df, is_home=1, prefix="D", min_threshold=min_threshold, feature_type=feature_type, debug_dir=debug_dir)

    if debug:
        _save_debug({
            "step": "pivots",
            "home_off": {"shape": list(home_off.shape), "games": len(home_off), "players": sum(c.startswith("O_") for c in home_off.columns)},
            "road_off": {"shape": list(road_off.shape), "games": len(road_off), "players": sum(c.startswith("O_") for c in road_off.columns)},
            "home_def": {"shape": list(home_def.shape), "games": len(home_def), "players": sum(c.startswith("D_") for c in home_def.columns)},
            "road_def": {"shape": list(road_def.shape), "games": len(road_def), "players": sum(c.startswith("D_") for c in road_def.columns)},
            "note": "home_def uses road players (is_home=0); road_def uses home players (is_home=1)",
        }, debug_dir, "features_02_pivots.json")

    if stop_after == "pivots":
        print("Stopped after: pivots")
        return

    # --- join ---
    d_only = lambda p: [c for c in p.columns if c.startswith("D_")]

    home_rows = pd.concat([home_off, home_def], axis=1).copy()
    home_rows.rename(columns={"tm_PTS": "team_pts"}, inplace=True)
    home_rows["home"] = 1
    road_rows = pd.concat([road_off, road_def], axis=1).copy()
    road_rows.rename(columns={"tm_PTS": "team_pts"}, inplace=True)
    road_rows["home"] = 0

    if debug:
        _save_debug({
            "step": "join",
            "home_rows": {
                "shape": list(home_rows.shape),
                "null_team_pts": int(home_rows["team_pts"].isna().sum()),
                "o_cols": sum(c.startswith("O_") for c in home_rows.columns),
                "d_cols": sum(c.startswith("D_") for c in home_rows.columns),
            },
            "road_rows": {
                "shape": list(road_rows.shape),
                "null_team_pts": int(road_rows["team_pts"].isna().sum()),
                "o_cols": sum(c.startswith("O_") for c in road_rows.columns),
                "d_cols": sum(c.startswith("D_") for c in road_rows.columns),
            },
        }, debug_dir, "features_03_join.json")

    if stop_after == "join":
        print("Stopped after: join")
        return

    # --- matrix ---
    matrix = pd.concat([home_rows, road_rows]).reset_index()

    if debug:
        o_cols = [c for c in matrix.columns if c.startswith("O_")]
        d_cols = [c for c in matrix.columns if c.startswith("D_")]
        _save_debug({
            "step": "matrix",
            "shape": list(matrix.shape),
            "rows": len(matrix),
            "o_cols": len(o_cols),
            "d_cols": len(d_cols),
            "null_team_pts": int(matrix["team_pts"].isna().sum()),
            "home_row_count": int((matrix["is_home"] == 1).sum()),
            "road_row_count": int((matrix["is_home"] == 0).sum()),
            "players_only_home": len(set(o_cols) - set(d_cols)),
            "players_only_road": len(set(d_cols) - set(o_cols)),
        }, debug_dir, "features_04_matrix.json")

    out_path = os.path.join(out_dir, "design_matrix.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    matrix.to_parquet(out_path, index=False)

    print(f"Design matrix shape: {matrix.shape}")
    print(f"O cols: {sum(c.startswith('O_') for c in matrix.columns)}  D cols: {sum(c.startswith('D_') for c in matrix.columns)}")
