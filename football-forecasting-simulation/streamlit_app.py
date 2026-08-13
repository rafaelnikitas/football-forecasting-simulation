"""Interactive product layer for forecasting and league simulations."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from football_lab.data.football_data_uk import parse_csv
from football_lab.data.synthetic import (
    create_simulation_snapshot,
    generate_synthetic_league,
)
from football_lab.evaluation.backtest import benchmark_models
from football_lab.models import (
    DixonColesModel,
    IndependentPoissonModel,
    LeagueRateBaseline,
)
from football_lab.simulation.league import current_table, simulate_league

st.set_page_config(
    page_title="Football Forecasting Lab",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data
def synthetic_snapshot() -> tuple[pd.DataFrame, pd.DataFrame]:
    return create_simulation_snapshot(generate_synthetic_league())


@st.cache_resource
def fit_cached_model(
    csv_payload: bytes | None = None,
    league: str = "premier_league",
    season: str = "2024-25",
) -> tuple[DixonColesModel, pd.DataFrame, pd.DataFrame]:
    if csv_payload:
        real = parse_csv(
            io.BytesIO(csv_payload),
            league=league,
            season=season,
        )
        completed = real.loc[real["status"].eq("FINISHED")].reset_index(drop=True)
        fixtures = real.loc[real["status"].eq("SCHEDULED")].reset_index(drop=True)
    else:
        completed, fixtures = synthetic_snapshot()
    return DixonColesModel().fit(completed), completed, fixtures


st.title("Football Forecasting & Simulation Lab")
st.caption("Calibrated score forecasts, leak-free backtests, and Monte Carlo league scenarios")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Football-Data.co.uk CSV", type=["csv"])
    st.selectbox(
        "League",
        ["premier_league", "la_liga"],
        key="upload_league",
        disabled=upload is None,
    )
    st.text_input("Season", value="2024-25", key="upload_season", disabled=upload is None)
    if upload is None:
        st.info("Running the clearly labeled synthetic demo.")
    else:
        st.success("Using the uploaded provider file.")

payload = upload.getvalue() if upload is not None else None
model, completed_matches, remaining_fixtures = fit_cached_model(
    payload,
    st.session_state.upload_league,
    st.session_state.upload_season,
)
teams = sorted(set(completed_matches["home_team"]).union(completed_matches["away_team"]))

overview_tab, prediction_tab, simulation_tab, model_tab, data_tab = st.tabs(
    ["Overview", "Match Predictor", "League Simulator", "Model Lab", "Data Sources"]
)

with overview_tab:
    columns = st.columns(4)
    columns[0].metric("Completed matches", f"{len(completed_matches):,}")
    columns[1].metric("Remaining fixtures", f"{len(remaining_fixtures):,}")
    columns[2].metric("Teams", len(teams))
    columns[3].metric("Model", "Dixon–Coles")
    st.subheader("Current table")
    latest_season = completed_matches["season"].iloc[-1]
    current_season = completed_matches.loc[completed_matches["season"].eq(latest_season)]
    st.dataframe(current_table(current_season), use_container_width=True, hide_index=True)
    st.subheader("Estimated team strengths")
    st.bar_chart(model.strengths().set_index("team")[["attack_strength", "defence_resistance"]])

with prediction_tab:
    left, right = st.columns(2)
    home_team = left.selectbox("Home team", teams, index=0)
    away_options = [team for team in teams if team != home_team]
    away_team = right.selectbox("Away team", away_options, index=min(1, len(away_options) - 1))
    prediction = model.predict_match(home_team, away_team)
    probability_columns = st.columns(3)
    probability_columns[0].metric("Home win", f"{prediction['p_home']:.1%}")
    probability_columns[1].metric("Draw", f"{prediction['p_draw']:.1%}")
    probability_columns[2].metric("Away win", f"{prediction['p_away']:.1%}")
    st.write(
        f"Expected goals: **{home_team} {prediction['expected_home_goals']:.2f} – "
        f"{prediction['expected_away_goals']:.2f} {away_team}**"
    )
    matrix = model.score_matrix(home_team, away_team)
    scorelines = [
        {"score": f"{home}-{away}", "probability": matrix[home, away]}
        for home in range(matrix.shape[0])
        for away in range(matrix.shape[1])
    ]
    st.subheader("Most likely scorelines")
    st.dataframe(
        pd.DataFrame(scorelines).nlargest(10, "probability"),
        use_container_width=True,
        hide_index=True,
        column_config={"probability": st.column_config.ProgressColumn(format="percent")},
    )

with simulation_tab:
    simulations = st.slider("Monte Carlo simulations", 500, 10_000, 2_000, step=500)
    if remaining_fixtures.empty:
        st.warning("This upload contains no scheduled fixtures to simulate.")
    elif st.button("Run league simulation", type="primary"):
        with st.spinner("Sampling season outcomes..."):
            summary, positions = simulate_league(
                model,
                completed_matches,
                remaining_fixtures,
                n_simulations=simulations,
            )
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "title_probability": st.column_config.ProgressColumn(format="percent"),
                "top_four_probability": st.column_config.ProgressColumn(format="percent"),
                "relegation_probability": st.column_config.ProgressColumn(format="percent"),
            },
        )
        selected_team = st.selectbox("Position distribution", summary["team"].tolist())
        chart = positions.loc[positions["team"].eq(selected_team)].set_index("position")
        st.bar_chart(chart["probability"])

with model_tab:
    st.write(
        "The repository evaluates models with expanding-window backtesting. "
        "Every prediction is made using only matches dated strictly before it."
    )
    if upload is None and st.button("Run synthetic benchmark"):
        with st.spinner("Running walk-forward fits..."):
            complete = generate_synthetic_league()
            metrics, _ = benchmark_models(
                complete,
                {
                    "league_rate_baseline": lambda: LeagueRateBaseline(),
                    "independent_poisson": lambda: IndependentPoissonModel(),
                    "dixon_coles": lambda: DixonColesModel(),
                },
                initial_train_matches=100,
                refit_every=50,
            )
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.dataframe(model.strengths(), use_container_width=True, hide_index=True)

with data_tab:
    st.markdown(
        """
        - **Football-Data.co.uk:** historical results, match statistics, and odds.
        - **football-data.org:** current fixtures and standings with a free API token.
        - **OpenFootball:** CC0 fallback match data.
        - **StatsBomb Open Data:** future event-level tactical analytics extension.

        Provider data is not redistributed by this repository. Follow each
        provider's terms and attribution requirements.
        """
    )
