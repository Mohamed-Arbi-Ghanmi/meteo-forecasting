"""Can I out-forecast the weathermen? A small PyTorch LSTM vs Open-Meteo's
official forecast, evaluated live on Streamlit."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CITIES, HOLDOUT_DAYS, HORIZON, SEQUENCE_LENGTH
from src.evaluate import backtest, predict_live

st.set_page_config(page_title="Out-Forecasting the Weathermen", page_icon="🌡️", layout="wide")

COLOR_ACTUAL = "#8a8f98"
COLOR_OFFICIAL = "#4c78a8"
COLOR_MODEL = "#e45756"


@st.cache_data(ttl=1800, show_spinner="Fetching latest data and running the model...")
def cached_predict_live(city: str, horizon_hours: int) -> pd.DataFrame:
    return predict_live(city, CITIES[city], horizon_hours=horizon_hours)


@st.cache_data(ttl=6 * 3600, show_spinner="Backtesting against the last 45 days...")
def cached_backtest(city: str) -> pd.DataFrame:
    return backtest(city, CITIES[city])


st.title("🌡️ Can I out-forecast the weathermen?")
st.caption(
    "A 2-layer PyTorch LSTM, trained only on historical hourly temperature, "
    "going head-to-head with Open-Meteo's official forecast."
)

tab_forecast, tab_scoreboard, tab_about = st.tabs(["Forecast", "Scoreboard", "About"])

with tab_forecast:
    col1, col2 = st.columns([2, 1])
    with col1:
        city = st.selectbox("City", list(CITIES.keys()), key="forecast_city")
    with col2:
        horizon_hours = st.radio("Horizon", [24, 48], index=1, horizontal=True)

    df = cached_predict_live(city, horizon_hours)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["actual_recent"], name="Actual (recent)",
                              line=dict(color=COLOR_ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["official"], name="Official forecast",
                              line=dict(color=COLOR_OFFICIAL, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["model"], name="Our LSTM",
                              line=dict(color=COLOR_MODEL, width=2, dash="dot")))
    fig.update_layout(
        height=480,
        yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    future = df.dropna(subset=["model", "official"])
    if not future.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model avg. temp (forecast window)", f"{future['model'].mean():.1f} °C")
        c2.metric("Official avg. temp (forecast window)", f"{future['official'].mean():.1f} °C")
        c3.metric("Model vs official, avg. gap", f"{(future['model'] - future['official']).abs().mean():.2f} °C")

with tab_scoreboard:
    city_sb = st.selectbox("City", list(CITIES.keys()), key="scoreboard_city")
    bt = cached_backtest(city_sb)

    if bt.empty:
        st.warning("Not enough overlapping data to backtest this city right now.")
    else:
        wins = int(bt["model_win"].sum())
        total = len(bt)
        c1, c2, c3 = st.columns(3)
        c1.metric("Model beat official forecast", f"{wins} / {total} days")
        c2.metric("Model avg. MAE", f"{bt['model_mae'].mean():.2f} °C")
        c3.metric("Official avg. MAE", f"{bt['official_mae'].mean():.2f} °C")

        st.caption(
            f"Backtest: for each of the last {HOLDOUT_DAYS} days, the model predicts the day's "
            f"24 hourly temperatures from the prior {SEQUENCE_LENGTH} actual hours; the official "
            "baseline is Open-Meteo's reconstructed historical forecast for that same day. "
            "Both are scored against what actually happened."
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt.index, y=bt["model_mae"], name="Our LSTM (daily MAE)",
                                   line=dict(color=COLOR_MODEL, width=2)))
        fig2.add_trace(go.Scatter(x=bt.index, y=bt["official_mae"], name="Official forecast (daily MAE)",
                                   line=dict(color=COLOR_OFFICIAL, width=2)))
        fig2.update_layout(
            height=380,
            yaxis_title="Mean absolute error (°C)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Daily results"):
            st.dataframe(bt.style.format({
                "model_mae": "{:.2f}", "model_rmse": "{:.2f}",
                "official_mae": "{:.2f}", "official_rmse": "{:.2f}",
            }), use_container_width=True)

with tab_about:
    st.markdown(f"""
### What this is
A small experiment: can a lightweight neural net, trained only on a city's own
temperature history, keep up with a professional numerical weather forecast?

### Data
All data comes from [Open-Meteo](https://open-meteo.com/), a free, keyless weather API:
- **Training data** — {(pd.Timestamp.today() - pd.DateOffset(years=3)).date()} to today, hourly
  temperature, via the archive endpoint.
- **Official forecast baseline** — Open-Meteo's historical-forecast API (reconstructed past
  forecasts) for backtesting, and its live forecast API for the current outlook.

### Model
A 2-layer LSTM (`hidden_size={64}`) takes the last **{SEQUENCE_LENGTH} hours** of actual
temperature and predicts the next **{HORIZON} hours** in one shot. Longer horizons (48h) are
produced by feeding the model's own predictions back in as history. One model is trained per
city, on ~3 years of hourly data, and shipped pretrained in this repo — nothing trains at
request time.

### Scoreboard methodology
For each of the last {HOLDOUT_DAYS} days, the model's predictions (seeded from real observed
history) and the official forecast for that day are both compared against what was actually
observed, using mean absolute error.

### Honest caveat
Open-Meteo's official forecast is built on full numerical weather prediction models —
satellite data, atmospheric physics, global assimilation. This project's model sees a single
number (temperature) for one city. It isn't trying to replace meteorology; it's a demonstration
of how much short-term forecasting skill a small sequence model can recover from historical
data alone, and how close that gets on a good day.
""")
