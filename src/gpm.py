import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from config import get_output_directory, get_output_format

_ROOT = Path(__file__).resolve().parent.parent


def _run_rapm_validate(job: dict, results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rapm_path = _ROOT / "data" / "rapm_30y.csv"
    if not rapm_path.exists():
        print(f"  RAPM validation skipped: {rapm_path} not found (run: python research/rapm_import.py)")
        return results.copy(), {}

    threshold = float(job.get("rapm_signal_threshold", 0.53))
    rapm = pd.read_csv(rapm_path)

    # Select xRAPM metrics and standard errors
    cols_to_keep = [
        "name_match_id", "offense", "offense_se", 
        "defense", "defense_se", "total", "total_se"
    ]
    rapm = rapm[rapm["name_signal"] > threshold][
        [c for c in cols_to_keep if c in rapm.columns]
    ].copy()
    
    rapm["player_id"] = rapm["name_match_id"].astype(str)
    rapm = rapm.rename(columns={
        "offense": "rapm_offense", "offense_se": "rapm_offense_se",
        "defense": "rapm_defense", "defense_se": "rapm_defense_se",
        "total": "rapm_total", "total_se": "rapm_total_se"
    })

    merged = results.copy()
    merged["player_id"] = merged["player_id"].astype(str)
    merged = merged.merge(rapm.drop(columns=["name_match_id"]), on="player_id", how="left")

    matched = merged.dropna(subset=["rapm_total"])
    metrics = {}
    if len(matched) > 0:
        for gpm_col, key in [
            ("offensive_rating", "offense"),
            ("defensive_rating", "defense"),
            ("combined_rating",  "total"),
        ]:
            diff = matched[gpm_col] - matched[f"rapm_{key}"]
            metrics[f"rmse_{key}"] = round(float((diff ** 2).mean() ** 0.5), 4)
            metrics[f"corr_{key}"] = round(float(matched[gpm_col].corr(matched[f"rapm_{key}"])), 4)
        metrics["n_players_matched"] = int(len(matched))

    print(f"  RAPM validation: {len(matched)}/{len(results)} matched (signal > {threshold})")
    return merged, metrics


SPOTLIGHT_PLAYERS = {
    "MJ":  "Michael Jordan",
    "OR":  "Oscar Robertson",
    "NJ":  "Nikola Jokic",
    "GM":  "George Mikan",
    "LJ":  "LeBron James",
    "KG":  "Kevin Garnett",
    "DR":  "David Robinson",
    "KAJ": "Kareem Abdul-Jabbar",
    "KD":  "Kevin Duckworth",
    "RM":  "Ron Mercer",
    "HW":  "Hakim Warrick",
    "PM":  "Pete Maravich",
    "JW":  "James Wiseman",
}


def _lookup_spotlight(merged: pd.DataFrame) -> dict:
    if "player_name" not in merged.columns:
        return {}
    name_col = merged["player_name"].fillna("").str.lower()
    out = {}
    for initials, full_name in SPOTLIGHT_PLAYERS.items():
        mask = name_col == full_name.lower()
        row = merged[mask]
        if not row.empty:
            out[initials] = round(float(row["combined_rating"].iloc[0]), 4)
    return out


def _build_run_summary(job: dict, merged: pd.DataFrame, metrics: dict) -> dict:
    GLM_TRACK = ["family", "lambda_", "alpha", "max_active_predictors",
                 "offset_column", "weights_column", "beta_constraints"]
    glm_kwargs = job.get("glm_kwargs", {})

    default_track = ["min_threshold", "threshold_type", "min_games", "min_total_games",
                     "season_start", "season_end", "rapm_signal_threshold"]
    extra_track = job.get("mlflow_track_params", [])
    params = {k: job.get(k) for k in dict.fromkeys(default_track + extra_track)}
    params.update({f"glm_{k}": glm_kwargs.get(k) for k in GLM_TRACK})

    has_name = "player_name" in merged.columns
    top10 = merged.nlargest(10, "combined_rating")[
        ["player_id", "offensive_rating", "defensive_rating", "combined_rating"]
        + (["player_name"] if has_name else [])
    ].copy()
    if not has_name:
        top10["player_name"] = top10["player_id"]
    else:
        top10["player_name"] = top10["player_name"].fillna(top10["player_id"])

    spotlight = _lookup_spotlight(merged)

    summary = {
        "run_timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params":         params,
        "metrics":        metrics,
        "spotlight":      spotlight,
        "top10_combined": top10[
            ["player_id", "player_name", "offensive_rating", "defensive_rating", "combined_rating"]
        ].round(4).to_dict(orient="records"),
    }

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    rows = (
        [{"key": f"params.{k}",   "value": v} for k, v in params.items()]
        + [{"key": f"metrics.{k}", "value": v} for k, v in metrics.items()]
        + [{"key": f"spotlight.{k}", "value": v} for k, v in spotlight.items()]
        + [{"key": f"top10.{i+1}.{k}", "value": v}
           for i, p in enumerate(summary["top10_combined"])
           for k, v in p.items()]
    )
    pd.DataFrame(rows).to_csv(results_dir / "run_summary.csv", index=False)
    print(f"  Run summary saved → {results_dir}/run_summary.json/.csv")
    return summary


def _log_to_mlflow(job: dict, summary: dict) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("  MLflow not installed — skipping (pip install mlflow)")
        return None

    tracking_uri = job.get("mlflow_tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(job.get("mlflow_experiment", "nba-gpm"))

    p = summary.get("params", {})
    custom_name = job.get("run_name", "").strip()
    if custom_name:
        existing = [r.info.run_name for r in mlflow.search_runs(
            experiment_names=[job.get("mlflow_experiment", "nba-gpm")],
            output_format="list",
        )] if mlflow.search_runs else []
        run_name = (
            f"{custom_name} ({datetime.now().strftime('%m-%d %H:%M')})"
            if custom_name in existing else custom_name
        )
    else:
        run_name = (
            f"{p.get('season_start', '?')}→{p.get('season_end', '?')} | "
            f"{p.get('feature_type', '?')}={p.get('min_threshold', '?')} | "
            f"min_g={p.get('min_games', '?')} | "
            f"{p.get('glm_family', '?')}"
        )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) if v is not None else "null"
                           for k, v in summary.get("params", {}).items()})

        # RAPM validation metrics (only when present)
        rapm_metrics = {k: float(v) for k, v in summary.get("metrics", {}).items()
                        if isinstance(v, (int, float))}
        if rapm_metrics:
            mlflow.log_metrics(rapm_metrics)

        # Top-5 ranked players — name as value
        for i, p in enumerate(summary.get("top10_combined", [])[:5], 1):
            mlflow.log_param(f"rank_{i}", p["player_name"])

        # Spotlight players
        spotlight = summary.get("spotlight", {})
        if spotlight:
            mlflow.log_metrics({k: float(v) for k, v in spotlight.items()})

        for rel in ["results/gpm_results.csv", "results/run_summary.json", "results/run_summary.csv"]:
            path = _ROOT / rel
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=rel.split("/")[0])

        run_id = mlflow.active_run().info.run_id

    tracking_uri_display = mlflow.get_tracking_uri()
    print(f"  MLflow run: {run_id}")
    print(f"  View: mlflow ui --backend-store-uri {tracking_uri_display}  → http://localhost:5000")
    return run_id


def run_gpm_stage(job) -> dict | None:
    output_dir = get_output_directory(job)
    for ext in ["parquet", "csv"]:
        design_matrix_path = os.path.join(output_dir, f"design_matrix.{ext}")
        if os.path.exists(design_matrix_path):
            break
    else:
        print(f"Design matrix not found in {output_dir}. Run features stage first.")
        return None

    print(f"Loading design matrix from {design_matrix_path}...")

    try:
        import h2o
        from h2o.estimators import H2OGeneralizedLinearEstimator
    except ImportError:
        print("H2O is not installed. Install with: pip install h2o (also requires Java 8+)")
        return None

    try:
        import psutil
        mem_gb = int(psutil.virtual_memory().total / 1024**3 * 0.75)
        h2o.init(max_mem_size=f"{mem_gb}g")
    except Exception as e:
        print(f"Failed to initialize H2O. Ensure Java 8+ is installed and available on PATH.\nError: {e}")
        return None

    summary = None
    try:
        print("Importing into H2O...")
        h2o_frame = h2o.import_file(design_matrix_path)

        o_cols = [c for c in h2o_frame.columns if c.startswith("O_")]
        d_cols = [c for c in h2o_frame.columns if c.startswith("D_")]
        predictors = o_cols + d_cols + ["home"]
        response = "team_pts"

        print(f"H2OFrame: {h2o_frame.shape[0]} rows, {len(predictors)} predictors")

        glm_kwargs = {k: v for k, v in job.get("glm_kwargs", {}).items() if not k.endswith("_note") and v is not None}
        glm = H2OGeneralizedLinearEstimator(**glm_kwargs)
        print("Training GLM...")
        glm.train(x=predictors, y=response, training_frame=h2o_frame)

        coef_df = glm.coef_with_p_values().as_data_frame()
        coef_df = coef_df.rename(columns={"names": "name"})

        off = coef_df[coef_df["name"].str.startswith("O_")].copy()
        off["player_id"] = off["name"].str[2:]
        off = off.rename(columns={"coefficients": "offensive_rating", "std_error": "offensive_se"})
        off = off[["player_id", "offensive_rating", "offensive_se"]]

        dff = coef_df[coef_df["name"].str.startswith("D_")].copy()
        dff["player_id"] = dff["name"].str[2:]
        dff = dff.rename(columns={"coefficients": "defensive_rating", "std_error": "defensive_se"})
        dff = dff[["player_id", "defensive_rating", "defensive_se"]]

        results = pd.merge(off, dff, on="player_id", how="outer")
        results["combined_rating"] = results["offensive_rating"] - results["defensive_rating"]

        # Join player names from source data
        for name_file in ("imputed_player_data.csv", "cleaned_player_data.csv"):
            path = output_dir / name_file
            if path.exists():
                names_df = pd.read_csv(path, usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates("PLAYER_ID")
                names_df["PLAYER_ID"] = names_df["PLAYER_ID"].astype(str)
                results = results.merge(names_df, left_on="player_id", right_on="PLAYER_ID", how="left").drop(columns=["PLAYER_ID"])
                results = results.rename(columns={"PLAYER_NAME": "player_name"})
                break

        # Enrich with RAPM metrics
        results, metrics = _run_rapm_validate(job, results)

        # Reorder columns: name first, then id, sorted descending by combined_rating
        name_col = ["player_name"] if "player_name" in results.columns else []
        other_cols = [c for c in results.columns if c not in name_col + ["player_id"]]
        results = results[name_col + ["player_id"] + other_cols].sort_values(
            "combined_rating", ascending=False
        ).reset_index(drop=True)

        intercept_row = coef_df[coef_df["name"] == "Intercept"]
        home_row      = coef_df[coef_df["name"] == "home"]
        if not intercept_row.empty:
            metrics["intercept"]        = round(float(intercept_row["coefficients"].values[0]), 4)
            print(f"\nIntercept: {metrics['intercept']:.4f}")
        if not home_row.empty:
            metrics["home_coefficient"] = round(float(home_row["coefficients"].values[0]), 4)
            print(f"Home coefficient: {metrics['home_coefficient']:.4f}")
        metrics["n_players"] = len(results)

        disp_name = name_col + ["player_id"]
        print("\nTop 10 Offensive Ratings:")
        print(results.nlargest(10, "offensive_rating")[disp_name + ["offensive_rating", "offensive_se"]].to_string(index=False))
        print("\nTop 10 Defensive Ratings (lower = better at suppressing opponent scoring):")
        print(results.nsmallest(10, "defensive_rating")[disp_name + ["defensive_rating", "defensive_se"]].to_string(index=False))
        print("\nTop 10 Combined Ratings (offensive - defensive):")
        print(results.head(10)[disp_name + ["offensive_rating", "defensive_rating", "combined_rating"]].to_string(index=False))

        results_dir = _ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        results.to_csv(results_dir / "gpm_results.csv", index=False)
        print(f"\nResults saved to {results_dir / 'gpm_results.csv'}")

        summary = _build_run_summary(job, results, metrics)
        run_id = _log_to_mlflow(job, summary)
        if run_id:
            summary["mlflow_run_id"] = run_id
            summary["mlflow_tracking_uri"] = job.get("mlflow_tracking_uri", "http://localhost:5000")

    finally:
        try:
            h2o.cluster().shutdown()
        except Exception:
            pass

    return summary
