import os
import pandas as pd
from config import get_output_directory, get_output_format


def run_gpm_stage(job):
    output_dir = get_output_directory(job)
    for ext, reader in [("parquet", pd.read_parquet), ("csv", pd.read_csv)]:
        design_matrix_path = os.path.join(output_dir, f"design_matrix.{ext}")
        if os.path.exists(design_matrix_path):
            break
    else:
        print(f"Design matrix not found in {output_dir}. Run features stage first.")
        return

    print(f"Loading design matrix from {design_matrix_path}...")
    df = reader(design_matrix_path)

    try:
        import h2o
        from h2o.estimators import H2OGeneralizedLinearEstimator
    except ImportError:
        print("H2O is not installed. Install with: pip install h2o (also requires Java 8+)")
        return

    try:
        h2o.init()
    except Exception as e:
        print(f"Failed to initialize H2O. Ensure Java 8+ is installed and available on PATH.\nError: {e}")
        return

    try:
        o_cols = [c for c in df.columns if c.startswith("O_")]
        d_cols = [c for c in df.columns if c.startswith("D_")]
        predictors = o_cols + d_cols + ["home"]
        response = "team_pts"

        print(f"Converting to H2OFrame ({len(df)} rows, {len(predictors)} predictors)...")
        h2o_frame = h2o.H2OFrame(df[predictors + [response]])

        glm = H2OGeneralizedLinearEstimator(
            family="gaussian",
            lambda_=0,
            compute_p_values=True,
            remove_collinear_columns=True,
            intercept=True,
        )
        print("Training GLM...")
        glm.train(x=predictors, y=response, training_frame=h2o_frame)

        coef_df = glm.coef_table().as_data_frame()
        # coef_df columns: name, coefficients, std_error, z_value, p_value

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

        intercept_row = coef_df[coef_df["name"] == "Intercept"]
        home_row = coef_df[coef_df["name"] == "home"]

        if not intercept_row.empty:
            print(f"\nIntercept: {intercept_row['coefficients'].values[0]:.4f}")
        if not home_row.empty:
            print(f"Home coefficient: {home_row['coefficients'].values[0]:.4f}")

        print("\nTop 10 Offensive Ratings:")
        print(results.nlargest(10, "offensive_rating")[["player_id", "offensive_rating", "offensive_se"]].to_string(index=False))

        print("\nTop 10 Defensive Ratings (lower = better at suppressing opponent scoring):")
        print(results.nsmallest(10, "defensive_rating")[["player_id", "defensive_rating", "defensive_se"]].to_string(index=False))

        print("\nTop 10 Combined Ratings (offensive - defensive):")
        print(results.nlargest(10, "combined_rating")[["player_id", "offensive_rating", "defensive_rating", "combined_rating"]].to_string(index=False))

        csv_path = os.path.join(output_dir, "gpm_results.csv")
        parquet_path = os.path.join(output_dir, "gpm_results.parquet")
        results.to_csv(csv_path, index=False)
        results.to_parquet(parquet_path, index=False)
        print(f"\nResults saved to {csv_path} and {parquet_path}")

    finally:
        try:
            h2o.shutdown(prompt=False)
        except Exception:
            pass
