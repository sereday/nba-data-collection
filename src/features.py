import os
import pandas as pd
from config import get_output_directory, get_output_format, get_target_stat


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


def run_features_stage(job):
    out_dir = get_output_directory(job)
    min_threshold = job.get("min_threshold", 24.0)
    threshold_type = job.get("threshold_type", "minutes")
    target_stat = get_target_stat(job)

    imputed = os.path.join(out_dir, "imputed_player_data.csv")
    cleaned = os.path.join(out_dir, "cleaned_player_data.csv")
    src = imputed if os.path.exists(imputed) else cleaned

    needed = {"GAME_ID", "PLAYER_ID", "TEAM_ID", "MATCHUP", "GAME_DATE",
              "season", "season_type", "MIN", "MIN_filled", target_stat}
    df = pd.read_csv(src, usecols=lambda c: c in needed)
    min_col = "MIN_filled" if "MIN_filled" in df.columns else "MIN"
    df["MIN_float"] = df[min_col].apply(_parse_min)

    if threshold_type == "pct":
        df["qualified"] = df["MIN_float"] / 48.0 >= min_threshold
    else:
        df["qualified"] = df["MIN_float"] >= min_threshold

    df["is_home"] = df["MATCHUP"].str.contains(r"\bvs\.", regex=True)

    game_meta = {}
    for game_id, gdf in df.groupby("GAME_ID"):
        home_rows = gdf[gdf["is_home"] & gdf["qualified"]]
        road_rows = gdf[~gdf["is_home"] & gdf["qualified"]]

        home_all = gdf[gdf["is_home"]]
        road_all = gdf[~gdf["is_home"]]

        home_pts_series = home_all[target_stat].dropna()
        road_pts_series = road_all[target_stat].dropna()

        game_meta[game_id] = {
            "home_players": set(home_rows["PLAYER_ID"].tolist()),
            "road_players": set(road_rows["PLAYER_ID"].tolist()),
            "home_team_id": home_all["TEAM_ID"].iloc[0] if len(home_all) else None,
            "road_team_id": road_all["TEAM_ID"].iloc[0] if len(road_all) else None,
            "home_pts": home_pts_series.iloc[0] if len(home_pts_series) else None,
            "road_pts": road_pts_series.iloc[0] if len(road_pts_series) else None,
            "game_date": gdf["GAME_DATE"].iloc[0],
            "season": gdf["season"].iloc[0],
            "season_type": gdf["season_type"].iloc[0],
        }

    all_player_ids = sorted(
        set(pid for m in game_meta.values() for pid in m["home_players"] | m["road_players"])
    )
    o_cols = [f"O_{pid}" for pid in all_player_ids]
    d_cols = [f"D_{pid}" for pid in all_player_ids]
    id_cols = ["GAME_ID", "team_id", "opp_id", "home", "team_pts", "GAME_DATE", "season", "season_type"]

    rows_a = []
    rows_b = []

    for game_id, m in game_meta.items():
        base = {
            "GAME_ID": game_id,
            "GAME_DATE": m["game_date"],
            "season": m["season"],
            "season_type": m["season_type"],
        }

        row_a = {
            **base,
            "team_id": m["home_team_id"],
            "opp_id": m["road_team_id"],
            "home": 1,
            "team_pts": m["home_pts"],
        }
        for pid in all_player_ids:
            row_a[f"O_{pid}"] = 1 if pid in m["home_players"] else 0
            row_a[f"D_{pid}"] = 1 if pid in m["road_players"] else 0
        rows_a.append(row_a)

        row_b = {
            **base,
            "team_id": m["road_team_id"],
            "opp_id": m["home_team_id"],
            "home": 0,
            "team_pts": m["road_pts"],
        }
        for pid in all_player_ids:
            row_b[f"O_{pid}"] = 1 if pid in m["road_players"] else 0
            row_b[f"D_{pid}"] = 1 if pid in m["home_players"] else 0
        rows_b.append(row_b)

    col_order = id_cols + o_cols + d_cols
    matrix = pd.concat(
        [pd.DataFrame(rows_a), pd.DataFrame(rows_b)],
        ignore_index=True,
    )[col_order]

    for c in o_cols + d_cols:
        if c not in matrix.columns:
            matrix[c] = 0
    matrix[o_cols + d_cols] = matrix[o_cols + d_cols].fillna(0).astype("int8")

    out_path = os.path.join(out_dir, "design_matrix.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    matrix.to_parquet(out_path, index=False)

    print(f"Design matrix shape: {matrix.shape}")
    print(f"Unique players: {len(all_player_ids)}")
