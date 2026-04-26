import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="NBA GPM", layout="wide")

st.title("NBA Generalized Plus Minus (GPM)")

# --- Sidebar ---
st.sidebar.header("Controls")

sort_by = st.sidebar.radio(
    "Sort by",
    ["Combined (O - D)", "Offensive", "Defensive"],
)

top_n = st.sidebar.slider("Top N players", min_value=10, max_value=100, value=25)

# TODO: Add minimum games threshold filter once appearance counts are available in gpm_results.

# --- Load data ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GPM_PATH = os.path.join(DATA_DIR, "gpm_results.csv")

if not os.path.exists(GPM_PATH):
    st.error(f"GPM results file not found: {GPM_PATH}")
    st.stop()

gpm = pd.read_csv(GPM_PATH)

required_cols = {"player_id", "offensive_rating", "offensive_se", "defensive_rating", "defensive_se"}
missing = required_cols - set(gpm.columns)
if missing:
    st.error(f"gpm_results.csv is missing columns: {missing}")
    st.stop()

# --- Optionally join player names ---
name_col = None
for name_file in ("imputed_player_data.csv", "cleaned_player_data.csv"):
    path = os.path.join(DATA_DIR, name_file)
    if os.path.exists(path):
        try:
            players = pd.read_csv(path, usecols=["PLAYER_ID", "PLAYER_NAME"])
            players = players.drop_duplicates("PLAYER_ID")
            gpm = gpm.merge(
                players,
                left_on="player_id",
                right_on="PLAYER_ID",
                how="left",
            )
            name_col = "PLAYER_NAME"
        except Exception:
            pass
        break

if name_col is None or name_col not in gpm.columns:
    gpm["PLAYER_NAME"] = gpm["player_id"].astype(str)
    name_col = "PLAYER_NAME"

# --- Derived metrics ---
gpm["combined_rating"] = gpm["offensive_rating"] - gpm["defensive_rating"]
gpm["neg_defensive_rating"] = -gpm["defensive_rating"]

# --- Sort ---
sort_map = {
    "Combined (O - D)": "combined_rating",
    "Offensive": "offensive_rating",
    "Defensive": "neg_defensive_rating",
}
sort_col = sort_map[sort_by]
gpm_sorted = gpm.sort_values(sort_col, ascending=False).reset_index(drop=True)

# --- Main table ---
st.subheader("Player Ratings")

display_df = gpm_sorted[[name_col, "offensive_rating", "offensive_se", "defensive_rating", "defensive_se", "combined_rating"]].copy()
display_df.columns = ["Player", "Off Rating", "Off SE", "Def Rating", "Def SE", "Combined"]

st.dataframe(
    display_df.style.format({
        "Off Rating": "{:.2f}",
        "Off SE": "{:.2f}",
        "Def Rating": "{:.2f}",
        "Def SE": "{:.2f}",
        "Combined": "{:.2f}",
    }),
    use_container_width=True,
)

st.divider()

# --- Scatter plot ---
st.subheader("Offensive vs Defensive Rating")

fig_scatter = px.scatter(
    gpm_sorted,
    x="offensive_rating",
    y="neg_defensive_rating",
    size=gpm_sorted["combined_rating"].clip(lower=0.01),
    hover_name=name_col,
    labels={
        "offensive_rating": "Offensive Rating",
        "neg_defensive_rating": "Defensive Rating (negated, higher = better)",
    },
    title="Offensive vs Defensive Rating",
)
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# --- Side-by-side bar charts ---
col_off, col_def = st.columns(2)

top_off = gpm_sorted.nlargest(top_n, "offensive_rating")
top_def = gpm_sorted.nlargest(top_n, "neg_defensive_rating")

with col_off:
    st.subheader(f"Top {top_n} Offensive Players")
    fig_off = px.bar(
        top_off.sort_values("offensive_rating"),
        x="offensive_rating",
        y=name_col,
        orientation="h",
        labels={"offensive_rating": "Offensive Rating", name_col: "Player"},
    )
    fig_off.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(400, top_n * 18))
    st.plotly_chart(fig_off, use_container_width=True)

with col_def:
    st.subheader(f"Top {top_n} Defensive Players")
    fig_def = px.bar(
        top_def.sort_values("neg_defensive_rating"),
        x="neg_defensive_rating",
        y=name_col,
        orientation="h",
        labels={"neg_defensive_rating": "Defensive Rating (negated)", name_col: "Player"},
    )
    fig_def.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(400, top_n * 18))
    st.plotly_chart(fig_def, use_container_width=True)

# --- Model info ---
st.divider()
st.caption(
    "GPM uses H2O GLM with λ=0 (no regularization). "
    "Coefficients represent per-game points contributed on offense/allowed on defense."
)
