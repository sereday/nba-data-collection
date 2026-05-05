"""
Interactive data quality explorer.

Run:
    streamlit run research/data_quality_viz.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CSV = os.path.join(_ROOT, "results", "data_quality_summary.csv")

STATS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
         "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["season_year"] = df["season"].str[:4].astype(int)
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
    all_years   = sorted(df["season_year"].unique())
    all_stypes  = sorted(df["season_type"].unique())

    # ── Sidebar filters ──────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    year_range = st.sidebar.slider(
        "Season year range",
        min_value=int(all_years[0]),
        max_value=int(all_years[-1]),
        value=(int(all_years[0]), int(all_years[-1])),
    )

    season_types = st.sidebar.multiselect(
        "Season types", options=all_stypes, default=["Regular"]
    )

    selected_stats = st.sidebar.multiselect(
        "Stats", options=STATS, default=STATS
    )

    x_round = st.sidebar.selectbox(
        "Round % not null to nearest", options=[1, 2, 5, 10], index=0, format_func=lambda v: f"{v}%"
    )

    ratio_clip = st.sidebar.slider(
        "Clip ratio y-axis to ±", min_value=0.5, max_value=5.0, value=2.0, step=0.1
    )

    # ── Filter ───────────────────────────────────────────────────────────────
    mask = (
        df["season_year"].between(*year_range) &
        df["season_type"].isin(season_types) &
        df["stat"].isin(selected_stats)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    filtered["pct_not_null_r"] = (filtered["pct_not_null"] / x_round).round() * x_round
    filtered["ratio"] = filtered["ratio"].clip(1 / ratio_clip, ratio_clip)

    # ── Plot ─────────────────────────────────────────────────────────────────
    hover_cols = [c for c in ["season", "stat", "null_games", "avg_nonNull",
                               "season_avg", "inferred_avg_null", "PLAYER_ID"] if c in filtered.columns]

    fig = px.scatter(
        filtered,
        x="pct_not_null_r",
        y="ratio",
        color="stat",
        hover_data=hover_cols,
        labels={
            "pct_not_null_r": f"% games not null (±{x_round}%)",
            "ratio":          "reported avg / season avg",
            "stat":           "Stat",
        },
        title=f"Data quality: {year_range[0]}–{year_range[1]}  |  {', '.join(season_types)}",
        opacity=0.5,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="ratio = 1")
    fig.update_layout(height=600, xaxis_range=[0, 105], yaxis_range=[1 / ratio_clip, ratio_clip])

    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    with st.expander("Show data table"):
        st.dataframe(filtered.sort_values("pct_not_null_r"), use_container_width=True)


if __name__ == "__main__":
    main()
