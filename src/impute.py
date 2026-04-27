import os
import warnings
import numpy as np
import pandas as pd

from config import build_season_plan, get_output_directory, get_output_format, get_skipped_stages

IMPUTABLE_STATS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]


def _parse_min(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _load_csv(path):
    if not os.path.exists(path):
        warnings.warn(f"File not found, skipping: {path}")
        return None
    return pd.read_csv(path, low_memory=False)


def _load_parquet_or_csv(base_path):
    parquet_path = base_path + ".parquet"
    csv_path = base_path + ".csv"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, low_memory=False)
    warnings.warn(f"No file found at {base_path}(.parquet|.csv)")
    return None


def _save(df, path, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _aggregate_logs(df, group_keys):
    df = df.copy()
    df["MIN"] = df["MIN"].apply(_parse_min)

    present_stats = [s for s in IMPUTABLE_STATS if s in df.columns]

    agg = {s: "sum" for s in present_stats}
    agg["GAME_ID"] = "count"
    count_agg = {s: lambda x: x.notna().sum() for s in present_stats}

    sums = df.groupby(group_keys, as_index=False)[present_stats + ["GAME_ID"]].agg(
        {**{s: "sum" for s in present_stats}, "GAME_ID": "count"}
    )
    sums = sums.rename(columns={s: f"{s}_log_sum" for s in present_stats})
    sums = sums.rename(columns={"GAME_ID": "GP_log"})

    counts = df.groupby(group_keys, as_index=False)[present_stats].agg(
        {s: lambda x: x.notna().sum() for s in present_stats}
    )
    counts = counts.rename(columns={s: f"{s}_log_count" for s in present_stats})

    merged = sums.merge(counts, on=group_keys)
    return merged, present_stats


def _load_season_stats(season_plan, data_dir, suffix, output_format):
    frames = []
    for season, season_type in season_plan:
        base = os.path.join(data_dir, f"{season}_{season_type}_{suffix}")
        if output_format == "parquet" and os.path.exists(base + ".parquet"):
            df = pd.read_parquet(base + ".parquet")
        elif os.path.exists(base + ".csv"):
            df = pd.read_csv(base + ".csv", low_memory=False)
        else:
            continue
        df["season"] = season
        df["season_type"] = season_type
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _compute_imputed(agg_df, season_stats_df, group_keys, present_stats):
    stat_cols = [s for s in present_stats if s in season_stats_df.columns]

    season_stats_clean = season_stats_df.copy()
    if "MIN" in season_stats_clean.columns:
        season_stats_clean["MIN"] = season_stats_clean["MIN"].apply(_parse_min)

    rename_season = {s: f"{s}_season" for s in stat_cols}
    rename_season["GP"] = "GP_season"
    season_stats_clean = season_stats_clean.rename(columns=rename_season)

    season_cols = group_keys + [c for c in season_stats_clean.columns if c in list(rename_season.values())]
    season_stats_clean = season_stats_clean[season_cols]

    merged = agg_df.merge(season_stats_clean, on=group_keys, how="inner")

    for s in present_stats:
        if s not in stat_cols:
            merged[f"{s}_imp"] = 0.0
            continue
        missing_stat = np.clip(merged[f"{s}_season"] - merged[f"{s}_log_sum"], 0, None)
        missing_gp = np.clip(merged["GP_season"] - merged[f"{s}_log_count"], 0, None)
        imp = np.where(missing_gp > 0, missing_stat / missing_gp.replace(0, np.nan), 0.0)
        merged[f"{s}_imp"] = imp

    imp_cols = group_keys + [f"{s}_imp" for s in present_stats if f"{s}_imp" in merged.columns]
    return merged[imp_cols]


def _save_impute_describe(imp_df, present_stats, out_path):
    imp_cols = [f"{s}_imp" for s in present_stats if f"{s}_imp" in imp_df.columns]
    if not imp_cols:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imp_df[imp_cols].describe().to_csv(out_path)
    print(f"Saved impute describe to {out_path}")


def _apply_imputed(cleaned_df, imp_df, group_keys, present_stats):
    df = cleaned_df.merge(imp_df, on=group_keys, how="left")
    for s in present_stats:
        imp_col = f"{s}_imp"
        if imp_col not in df.columns:
            continue
        if s not in df.columns:
            df[s] = df[imp_col]
        else:
            df[s] = np.where(df[s].notna(), df[s], df[imp_col])
    drop_cols = [f"{s}_imp" for s in present_stats if f"{s}_imp" in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def _run_player_impute(season_plan, data_dir, cleaned_df, output_format):
    group_keys_agg = ["PLAYER_ID", "TEAM_ID", "season", "season_type"]

    if cleaned_df is None or cleaned_df.empty:
        warnings.warn("cleaned_player_data is empty or missing; skipping player impute.")
        return

    agg_df, present_stats = _aggregate_logs(cleaned_df, group_keys_agg)

    season_stats_df = _load_season_stats(season_plan, data_dir, "player_season_stats", output_format)
    if season_stats_df.empty:
        warnings.warn("No player season stats loaded; skipping player impute.")
        return

    has_team_id = season_stats_df["TEAM_ID"].notna() if "TEAM_ID" in season_stats_df.columns else pd.Series(False, index=season_stats_df.index)
    season_stats_df = season_stats_df[has_team_id]

    if season_stats_df.empty:
        warnings.warn("No player season stats with TEAM_ID available; skipping player impute.")
        return

    join_keys = ["PLAYER_ID", "TEAM_ID", "season", "season_type"]
    imp_df = _compute_imputed(agg_df, season_stats_df, join_keys, present_stats)

    _save_impute_describe(imp_df, present_stats, os.path.join(os.path.dirname(data_dir), "validation", "impute_describe_player.csv"))

    result = _apply_imputed(cleaned_df, imp_df, join_keys, present_stats)

    ext = "parquet" if output_format == "parquet" else "csv"
    out_path = os.path.join(data_dir, f"imputed_player_data.{ext}")
    _save(result, out_path, output_format)
    print(f"Saved imputed player data to {out_path}")


def _run_team_impute(season_plan, data_dir, output_format):
    frames = []
    for season, season_type in season_plan:
        base = os.path.join(data_dir, f"{season}_{season_type}_teams")
        if output_format == "parquet" and os.path.exists(base + ".parquet"):
            df = pd.read_parquet(base + ".parquet")
        elif os.path.exists(base + ".csv"):
            df = pd.read_csv(base + ".csv", low_memory=False)
        else:
            continue
        df["season"] = season
        df["season_type"] = season_type
        frames.append(df)

    if not frames:
        warnings.warn("No team game log files found; skipping team impute.")
        return

    team_logs = pd.concat(frames, ignore_index=True)

    group_keys_agg = ["TEAM_ID", "season", "season_type"]
    agg_df, present_stats = _aggregate_logs(team_logs, group_keys_agg)

    season_stats_df = _load_season_stats(season_plan, data_dir, "team_season_stats", output_format)
    if season_stats_df.empty:
        warnings.warn("No team season stats loaded; skipping team impute.")
        return

    join_keys = ["TEAM_ID", "season", "season_type"]
    imp_df = _compute_imputed(agg_df, season_stats_df, join_keys, present_stats)

    _save_impute_describe(imp_df, present_stats, os.path.join(os.path.dirname(data_dir), "validation", "impute_describe_team.csv"))

    result = _apply_imputed(team_logs, imp_df, join_keys, present_stats)

    ext = "parquet" if output_format == "parquet" else "csv"
    out_path = os.path.join(data_dir, f"imputed_team_data.{ext}")
    _save(result, out_path, output_format)
    print(f"Saved imputed team data to {out_path}")


def run_impute_stage(job):
    if "impute" in get_skipped_stages(job):
        print("Skipping impute stage.")
        return

    seasons = build_season_plan(job)
    season_types = list(job.get("season_types", []))
    season_plan = [(s, st) for s in seasons for st in season_types]
    data_dir = get_output_directory(job)
    output_format = get_output_format(job)

    cleaned_base = os.path.join(data_dir, "cleaned_player_data")
    cleaned_df = _load_parquet_or_csv(cleaned_base)

    _run_player_impute(season_plan, data_dir, cleaned_df, output_format)
    _run_team_impute(season_plan, data_dir, output_format)
