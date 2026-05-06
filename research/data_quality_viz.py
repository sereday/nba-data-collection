"""
Interactive data quality explorer.

Run:
    streamlit run research/data_quality_viz.py
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CSV = os.path.join(_ROOT, "results", "data_quality_summary.csv")

STATS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
         "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

GROUPING_OPTIONS = {
    "Per player / season": [],
    "By season × stat":    ["season", "stat"],
    "By player × stat":    ["PLAYER_ID", "player_name", "stat"],
    "By stat only":        ["stat"],
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["ratio"] = (df["avg_nonNull"] / df["season_avg"]).replace([np.inf, -np.inf], np.nan)
    return df


def main():
    st.set_page_config(page_title="Data Quality Explorer", layout="wide")
    st.title("Data Quality Explorer")

    csv_path = st.sidebar.text_input("CSV path", value=_DEFAULT_CSV)
    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        st.stop()

    df = load_data(csv_path)
    all_seasons = sorted(df["season"].unique())
    all_stypes  = sorted(df["season_type"].unique())

    # ── Sidebar filters ──────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    season_start = st.sidebar.selectbox(
        "Season start", options=all_seasons, index=0
    )
    season_end = st.sidebar.selectbox(
        "Season end", options=all_seasons, index=len(all_seasons) - 1
    )

    season_types = st.sidebar.multiselect(
        "Season types", options=all_stypes, default=["Regular"]
    )

    selected_stats = st.sidebar.multiselect(
        "Stats", options=STATS, default=STATS
    )

    st.sidebar.header("Display")

    grouping_label = st.sidebar.radio(
        "Group % not null by", options=list(GROUPING_OPTIONS.keys())
    )

    x_round = st.sidebar.selectbox(
        "Round % not null to nearest", options=[1, 2, 5, 10], index=0,
        format_func=lambda v: f"{v}%"
    )

    ratio_clip = st.sidebar.slider(
        "Clip ratio y-axis to ±", min_value=0.5, max_value=5.0, value=2.0, step=0.1
    )

    # ── Filter ───────────────────────────────────────────────────────────────
    mask = (
        (df["season"] >= season_start) &
        (df["season"] <= season_end) &
        df["season_type"].isin(season_types) &
        df["stat"].isin(selected_stats)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    filtered["pct_not_null_r"] = (filtered["pct_not_null"] / x_round).round() * x_round
    filtered["ratio"] = filtered["ratio"].clip(1 / ratio_clip, ratio_clip)

    # ── Aggregate ────────────────────────────────────────────────────────────
    group_cols = GROUPING_OPTIONS[grouping_label]
    if group_cols:
        agg_cols = {
            "pct_not_null_r": "mean",
            "ratio":          "mean",
            "avg_nonNull":    "mean",
            "season_avg":     "mean",
            "null_games":     "mean",
            "inferred_avg_null": "mean",
            "PLAYER_ID":      "nunique",
        }
        agg_cols = {k: v for k, v in agg_cols.items()
                    if k in filtered.columns or k == "PLAYER_ID"}
        plot_df = filtered.groupby(group_cols, as_index=False).agg(
            {k: v for k, v in agg_cols.items() if k in filtered.columns}
        )
        if "PLAYER_ID" in plot_df.columns and "PLAYER_ID" in agg_cols:
            plot_df = plot_df.rename(columns={"PLAYER_ID": "n_players"})
    else:
        plot_df = filtered.copy()

    # ── Plot ─────────────────────────────────────────────────────────────────
    color_col = "stat" if "stat" in plot_df.columns else None
    hover_cols = [c for c in ["season", "stat", "player_name", "n_players",
                               "null_games", "avg_nonNull", "season_avg",
                               "inferred_avg_null"] if c in plot_df.columns]

    fig = px.bar(
        plot_df.sort_values("pct_not_null_r"),
        x="pct_not_null_r",
        y="ratio",
        color=color_col,
        barmode="group",
        hover_data=hover_cols,
        labels={
            "pct_not_null_r": f"% games not null (rounded to {x_round}%)",
            "ratio":          "reported avg / season avg",
            "stat":           "Stat",
        },
        title=(
            f"Data quality: {season_start} → {season_end}"
            f"  |  {', '.join(season_types)}"
            f"  |  {grouping_label}"
        ),
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="ratio = 1")
    fig.update_layout(height=600, xaxis_range=[0, 105],
                      yaxis_range=[1 / ratio_clip, ratio_clip])

    st.plotly_chart(fig, use_container_width=True)

    # ── Means table ───────────────────────────────────────────────────────────
    mean_cols = [c for c in ["stat", "season", "player_name",
                              "pct_not_null_r", "ratio", "avg_nonNull", "season_avg",
                              "null_games", "inferred_avg_null"] if c in plot_df.columns]
    group_by = [c for c in ["stat", "season"] if c in plot_df.columns]
    if group_by:
        means_df = (
            plot_df[mean_cols]
            .groupby(group_by, as_index=False)
            .mean(numeric_only=True)
            .sort_values(group_by)
        )
    else:
        means_df = plot_df[mean_cols].mean(numeric_only=True).to_frame("mean").T

    st.subheader("Means")
    st.dataframe(means_df.style.format({
        "pct_not_null_r": "{:.1f}%",
        "ratio":          "{:.4f}",
        "avg_nonNull":    "{:.2f}",
        "season_avg":     "{:.2f}",
        "null_games":     "{:.1f}",
        "inferred_avg_null": "{:.2f}",
    }, na_rep="—"), use_container_width=True)

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Show raw data"):
        st.dataframe(plot_df.sort_values("pct_not_null_r"), use_container_width=True)


if __name__ == "__main__":
    main()
