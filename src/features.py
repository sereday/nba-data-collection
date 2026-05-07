import json
import os

import numpy as np
import pandas as pd

from config import get_output_directory, get_output_format, get_target_stat, save_dataframe, load_dataframe

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


def _build_pivot(df, is_home: int, prefix: str, min_threshold: float, feature_type: str,
                 target_col: str = None, player_universe: list = None, game_universe: list = None):
    min_col = "MIN_filled" if "MIN_filled" in df.columns else "MIN"
    side = df[df["is_home"] == is_home].copy()
    side["MIN_float"] = side[min_col].apply(_parse_min)

    if feature_type == "pct":
        side["MIN_pct"] = side["MIN_float"] / 48.0
        qualified = side[side["MIN_pct"] >= min_threshold]
    else:
        qualified = side[side["MIN_float"] >= min_threshold]

    qualified = qualified[
        ["GAME_ID", "PLAYER_ID"] + (["MIN_pct"] if feature_type == "pct" else [])
    ].drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])

    # Use caller-supplied universes (ensures identical index/columns across pivots) or derive locally
    all_games   = list(game_universe)   if game_universe   is not None else sorted(qualified["GAME_ID"].unique())
    all_players = list(player_universe) if player_universe is not None else sorted(qualified["PLAYER_ID"].unique())
    game_idx    = {g: i for i, g in enumerate(all_games)}
    player_idx  = {p: i for i, p in enumerate(all_players)}

    # Only keep rows whose game AND player are both in the respective universes
    q = qualified[qualified["GAME_ID"].isin(game_idx) & qualified["PLAYER_ID"].isin(player_idx)]
    row_idx = q["GAME_ID"].map(game_idx).values
    col_idx = q["PLAYER_ID"].map(player_idx).values

    if feature_type == "pct":
        data = np.zeros((len(all_games), len(all_players)), dtype=np.float32)
        data[row_idx, col_idx] = q["MIN_pct"].values
    else:
        data = np.zeros((len(all_games), len(all_players)), dtype=np.int8)
        data[row_idx, col_idx] = 1

    pivot = pd.DataFrame(data, index=all_games, columns=[f"{prefix}_{p}" for p in all_players])
    pivot.index.name = "GAME_ID"
    if target_col is not None:
        pts = side.groupby("GAME_ID")[target_col].first().reindex(pivot.index).rename(target_col)
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


def _build_qualified_games(home_pivot: pd.DataFrame, road_pivot: pd.DataFrame, prefix: str) -> pd.DataFrame:
    def _counts(pivot):
        cols = [c for c in pivot.columns if c.startswith(f"{prefix}_")]
        counts = (pivot[cols] != 0).sum()
        counts.index = [c[len(prefix) + 1:] for c in counts.index]
        return counts

    home_q = _counts(home_pivot).rename("qualified_home")
    road_q = _counts(road_pivot).rename("qualified_road")
    qg = pd.concat([home_q, road_q], axis=1).fillna(0).astype(int).reset_index()
    qg.columns = ["PLAYER_ID", "qualified_home", "qualified_road"]
    qg["qualified_total"] = qg["qualified_home"] + qg["qualified_road"]
    return qg


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
    min_threshold = float(job.get("min_threshold", 24.0))
    feature_type  = job.get("feature_type", "minutes")
    min_games      = int(job.get("min_games", 0))
    target_stat    = get_target_stat(job)
    off_def_split  = bool(job.get("off_def_split", True))
    debug          = bool(job.get("debug_features", False))
    stop_after     = job.get("debug_stop_after", "matrix")
    debug_dir      = os.path.join(str(out_dir), "..", "debug")

    if stop_after not in STOP_AFTER_OPTIONS:
        raise ValueError(f"debug_stop_after must be one of {STOP_AFTER_OPTIONS}")

    # --- load ---
    needed = {"GAME_ID", "PLAYER_ID", "TEAM_ID", "GAME_DATE", "is_home", "MATCHUP",
              "season", "season_type", "MIN", "MIN_filled", target_stat}
    df = load_dataframe(os.path.join(out_dir, "imputed_player_data"), columns=list(needed))
    if df is None:
        df = load_dataframe(os.path.join(out_dir, "cleaned_player_data"), columns=list(needed))
    if df is None:
        raise FileNotFoundError(f"No imputed or cleaned player data found in {out_dir}")
    src = str(out_dir)

    if "is_home" not in df.columns:
        if "MATCHUP" not in df.columns:
            raise ValueError("cleaned data has neither 'is_home' nor 'MATCHUP' — re-run the clean stage")
        print("  'is_home' missing — deriving from MATCHUP (re-run clean stage to fix permanently)")
        df["is_home"] = (~df["MATCHUP"].str.contains("@", regex=False)).astype("int8")
    df = df.drop(columns=["MATCHUP"], errors="ignore")

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

    if not job.get("preseason", True) and "season_type" in df.columns:
        before = df["GAME_ID"].nunique()
        df = df[df["season_type"] != "Preseason"]
        print(f"  preseason filter: {df['GAME_ID'].nunique()}/{before} games kept")

    training_season_start = job.get("training_season_start")
    if training_season_start and "season" in df.columns:
        before = df["GAME_ID"].nunique()
        df = df[df["season"] >= training_season_start]
        print(f"  training_season_start filter: {df['GAME_ID'].nunique()}/{before} games kept (>= {training_season_start})")

    if job.get("pace_adjustment", False):
        raise NotImplementedError(
            "pace_adjustment is not yet ready — blocked on team_season_stats data quality issues. "
            "Implementation plan: compute per-game pace from team_season_stats via research/pace_estimate.py, "
            "then divide tm_PTS by game-average pace_est before building pivots. "
            "See pace_adjustment__note in job_request.json."
        )

    if stop_after == "load":
        print("Stopped after: load")
        return

    # --- games played ---
    games_played = (
        df.groupby("PLAYER_ID")["GAME_ID"].nunique()
        .rename("games_played")
        .reset_index()
    )
    save_dataframe(games_played, os.path.join(out_dir, "player_games"))

    if min_games > 0:
        eligible = games_played.loc[games_played["games_played"] >= min_games, "PLAYER_ID"]
        before = df["PLAYER_ID"].nunique()
        df = df[df["PLAYER_ID"].isin(eligible)]
        print(f"  min_games filter: {len(eligible)}/{before} players kept (>= {min_games} games)")

    if off_def_split:
        # --- pivots (split) ---
        # Home row:  O_ = home qualified players,  D_ = road qualified players,  target = home pts
        # Road row:  O_ = road qualified players,  D_ = home qualified players,  target = road pts
        home_off = _build_pivot(df, is_home=1, prefix="O", target_col="tm_PTS", min_threshold=min_threshold, feature_type=feature_type)
        road_off = _build_pivot(df, is_home=0, prefix="O", target_col="tm_PTS", min_threshold=min_threshold, feature_type=feature_type)
        home_def = _build_pivot(df, is_home=0, prefix="D", min_threshold=min_threshold, feature_type=feature_type)
        road_def = _build_pivot(df, is_home=1, prefix="D", min_threshold=min_threshold, feature_type=feature_type)

        if debug:
            _save_debug({
                "step": "pivots",
                "mode": "off_def_split",
                "home_off": {"shape": list(home_off.shape), "games": len(home_off), "players": sum(c.startswith("O_") for c in home_off.columns)},
                "road_off": {"shape": list(road_off.shape), "games": len(road_off), "players": sum(c.startswith("O_") for c in road_off.columns)},
                "home_def": {"shape": list(home_def.shape), "games": len(home_def), "players": sum(c.startswith("D_") for c in home_def.columns)},
                "road_def": {"shape": list(road_def.shape), "games": len(road_def), "players": sum(c.startswith("D_") for c in road_def.columns)},
                "note": "home_def uses road players (is_home=0); road_def uses home players (is_home=1)",
            }, debug_dir, "features_02_pivots.json")

        if stop_after == "pivots":
            print("Stopped after: pivots")
            return

        # --- join (split) ---
        home_rows = pd.concat([home_off, home_def], axis=1).copy()
        home_rows.rename(columns={"tm_PTS": "team_pts"}, inplace=True)
        home_rows["home"] = 1
        road_rows = pd.concat([road_off, road_def], axis=1).copy()
        road_rows.rename(columns={"tm_PTS": "team_pts"}, inplace=True)
        road_rows["home"] = 0

        if debug:
            _save_debug({
                "step": "join",
                "mode": "off_def_split",
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

        # --- matrix (split) ---
        matrix = pd.concat([home_rows, road_rows]).reset_index()
        matrix["team_pts"] = pd.to_numeric(matrix["team_pts"], errors="coerce")
        null_pts = matrix["team_pts"].isna().sum()
        if null_pts:
            print(f"  WARNING: dropping {null_pts} rows with null team_pts")
            matrix = matrix[matrix["team_pts"].notna()]

        if debug:
            o_cols = [c for c in matrix.columns if c.startswith("O_")]
            d_cols = [c for c in matrix.columns if c.startswith("D_")]
            _save_debug({
                "step": "matrix",
                "mode": "off_def_split",
                "shape": list(matrix.shape),
                "rows": len(matrix),
                "o_cols": len(o_cols),
                "d_cols": len(d_cols),
                "null_team_pts": int(matrix["team_pts"].isna().sum()),
                "home_row_count": int((matrix.get("home", pd.Series()) == 1).sum()),
                "road_row_count": int((matrix.get("home", pd.Series()) == 0).sum()),
                "players_only_home": len(set(o_cols) - set(d_cols)),
                "players_only_road": len(set(d_cols) - set(o_cols)),
            }, debug_dir, "features_04_matrix.json")

        save_dataframe(_build_qualified_games(home_off, road_off, "O"),
                       os.path.join(out_dir, "player_qualified_games"))
        print(f"Design matrix shape: {matrix.shape}")
        print(f"O cols: {sum(c.startswith('O_') for c in matrix.columns)}  D cols: {sum(c.startswith('D_') for c in matrix.columns)}")

    else:
        ptdiff_type     = job.get("ptdiff_type", "diff")
        ptdiff_exponent = float(job.get("ptdiff_exponent", 1.0))

        # --- game-level target for ALL games (before pivots, so no NaN from missing side) ---
        home_pts_all = df[df["is_home"] == 1].groupby("GAME_ID")[target_stat].first()
        road_pts_all = df[df["is_home"] == 0].groupby("GAME_ID")[target_stat].first()

        if ptdiff_type == "ratio":
            # Scale by 100 so coefficients are in the same ballpark as a diff-based model
            game_target = ((home_pts_all / road_pts_all.replace(0, np.nan)) ** ptdiff_exponent) * 100
        else:
            game_target = home_pts_all - road_pts_all

        # --- shared player + game universes ---
        def _qualifying_ids(is_home):
            min_col = "MIN_filled" if "MIN_filled" in df.columns else "MIN"
            side = df[df["is_home"] == is_home].copy()
            side["MIN_float"] = side[min_col].apply(_parse_min)
            if feature_type == "pct":
                return set(side[side["MIN_float"] / 48.0 >= min_threshold]["PLAYER_ID"].unique()), \
                       set(side[side["MIN_float"] / 48.0 >= min_threshold]["GAME_ID"].unique())
            return set(side[side["MIN_float"] >= min_threshold]["PLAYER_ID"].unique()), \
                   set(side[side["MIN_float"] >= min_threshold]["GAME_ID"].unique())

        home_players, home_games = _qualifying_ids(1)
        road_players, road_games = _qualifying_ids(0)

        all_players = sorted(home_players | road_players)
        all_game_ids = sorted(home_games | road_games)
        print(f"  Combined player universe: {len(all_players)} players, {len(all_game_ids)} games")

        # --- pivots (combined, one row per game) ---
        # Both pivots share identical index (all_game_ids) and columns (all_players).
        # home_lineup[game, player] = 1 if player was a qualifying home player in that game.
        # road_lineup[game, player] = 1 if player was a qualifying road player in that game.
        home_lineup = _build_pivot(df, is_home=1, prefix="P", target_col=None,
                                   min_threshold=min_threshold, feature_type=feature_type,
                                   player_universe=all_players, game_universe=all_game_ids)
        road_lineup = _build_pivot(df, is_home=0, prefix="P", target_col=None,
                                   min_threshold=min_threshold, feature_type=feature_type,
                                   player_universe=all_players, game_universe=all_game_ids)

        assert home_lineup.shape == road_lineup.shape, (
            f"Pivot shape mismatch: home={home_lineup.shape}, road={road_lineup.shape}"
        )
        assert list(home_lineup.columns) == list(road_lineup.columns), "Pivot column mismatch"
        assert list(home_lineup.index) == list(road_lineup.index), "Pivot index mismatch"

        if debug:
            _save_debug({
                "step": "pivots",
                "mode": "combined",
                "home_lineup": {"shape": list(home_lineup.shape), "games": len(home_lineup), "players": len(all_players)},
                "road_lineup": {"shape": list(road_lineup.shape), "games": len(road_lineup), "players": len(all_players)},
                "ptdiff_exponent": ptdiff_exponent,
                "note": "home players +1, road players -1; shared game+player universe, guaranteed same shape",
            }, debug_dir, "features_02_pivots.json")

        if stop_after == "pivots":
            print("Stopped after: pivots")
            return

        # --- join (combined) ---
        # home_lineup and road_lineup have identical index and columns — subtract is element-wise.
        # Result: +1 where player was on home team, -1 where on road team, 0 otherwise.
        signed = home_lineup.subtract(road_lineup)
        signed["team_pts"] = game_target.reindex(home_lineup.index)

        if debug:
            _save_debug({
                "step": "join",
                "mode": "combined",
                "signed_shape": list(signed.shape),
                "null_team_pts": int(signed["team_pts"].isna().sum()),
                "p_cols": sum(c.startswith("P_") for c in signed.columns),
            }, debug_dir, "features_03_join.json")

        if stop_after == "join":
            print("Stopped after: join")
            return

        # --- matrix (combined) ---
        matrix = signed.reset_index()
        matrix["team_pts"] = pd.to_numeric(matrix["team_pts"], errors="coerce")
        null_pts = matrix["team_pts"].isna().sum()
        if null_pts:
            print(f"  WARNING: dropping {null_pts} rows with null team_pts")
            matrix = matrix[matrix["team_pts"].notna()]

        if debug:
            p_cols = [c for c in matrix.columns if c.startswith("P_")]
            _save_debug({
                "step": "matrix",
                "mode": "combined",
                "shape": list(matrix.shape),
                "rows": len(matrix),
                "p_cols": len(p_cols),
                "null_team_pts": int(matrix["team_pts"].isna().sum()),
            }, debug_dir, "features_04_matrix.json")

        save_dataframe(_build_qualified_games(home_lineup, road_lineup, "P"),
                       os.path.join(out_dir, "player_qualified_games"))
        print(f"Design matrix shape: {matrix.shape}")
        print(f"P cols (signed home−road): {sum(c.startswith('P_') for c in matrix.columns)}")

    save_dataframe(matrix, os.path.join(out_dir, "design_matrix"))
    _build_career_summary(job, out_dir)


def _build_career_summary(job: dict, out_dir: str) -> None:
    """Aggregate player_season_stats across all seasons into a career summary."""
    data_dir = get_output_directory(job)
    season_types = job.get("season_types", ["Regular"])
    # Only use Regular season for career MPG
    use_type = "Regular" if "Regular" in season_types else season_types[0]

    files = sorted(data_dir.glob(f"*_{use_type}_player_season_stats.csv"))
    if not files:
        print(f"  career_summary: no player_season_stats files found, skipping")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["PLAYER_ID", "PLAYER_NAME", "GP", "MIN"],
                             low_memory=False)
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return

    all_seasons = pd.concat(dfs, ignore_index=True)
    all_seasons["GP"] = pd.to_numeric(all_seasons["GP"], errors="coerce").fillna(0)
    all_seasons["MIN"] = all_seasons["MIN"].apply(_parse_min)

    career = (
        all_seasons.groupby("PLAYER_ID", as_index=False)
        .agg(
            career_gp=("GP", "sum"),
            career_min=("MIN", "sum"),
            player_name=("PLAYER_NAME", "last"),
        )
    )
    career["career_mpg"] = (career["career_min"] / career["career_gp"]).round(2)
    career["career_gp"] = career["career_gp"].astype(int)
    career["career_min"] = career["career_min"].round(1)
    career["PLAYER_ID"] = career["PLAYER_ID"].astype(str)

    save_dataframe(career, os.path.join(out_dir, "career_stats"))
    print(f"  Career summary: {len(career)} players saved → {out_dir}/career_stats")
