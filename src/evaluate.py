"""Backtest: for each day in the holdout window, predict with our LSTM using the
prior SEQUENCE_LENGTH actual hours, and compare both our model and Open-Meteo's
reconstructed historical forecast against what was actually observed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import HOLDOUT_DAYS, HORIZON, MODELS_DIR, SEQUENCE_LENGTH
from src.data_fetcher import fetch_archive, fetch_historical_forecast, fetch_live_forecast
from src.model import TemperatureLSTM
from src.preprocessing import Scaler, clean_series
from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


def load_model(city: str) -> tuple[TemperatureLSTM, Scaler]:
    model = TemperatureLSTM(HIDDEN_SIZE, NUM_LAYERS, HORIZON, DROPOUT)
    model.load_state_dict(torch.load(MODELS_DIR / f"{city}.pt", map_location="cpu", weights_only=True))
    model.eval()
    with open(MODELS_DIR / f"{city}_scaler.json") as f:
        scaler = Scaler.from_dict(json.load(f))
    return model, scaler


def predict_next(model: TemperatureLSTM, scaler: Scaler, history_values: np.ndarray) -> np.ndarray:
    """history_values: last SEQUENCE_LENGTH actual temperatures (real units) -> next HORIZON hours."""
    scaled = scaler.transform(history_values).astype(np.float32)
    x = torch.from_numpy(scaled).view(1, -1, 1)
    with torch.no_grad():
        pred_scaled = model(x).numpy().ravel()
    return scaler.inverse(pred_scaled)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def backtest(city: str, info: dict, holdout_days: int = HOLDOUT_DAYS,
             lag_days: int = 2) -> pd.DataFrame:
    """Per-day model vs official-forecast accuracy over the most recent `holdout_days`.

    `lag_days` skips the freshest couple of days since archive observations
    for them may not be finalized yet.
    """
    model, scaler = load_model(city)

    lead_days = -(-SEQUENCE_LENGTH // 24) + 1  # hours of history needed, in whole days, +buffer
    end_date_actual = pd.Timestamp.today().normalize() - pd.Timedelta(days=lag_days)
    start_date_actual = end_date_actual - pd.Timedelta(days=holdout_days + lead_days)

    actual = fetch_archive(info["lat"], info["lon"], start_date_actual.date().isoformat(),
                            end_date_actual.date().isoformat(), info["timezone"])
    actual = clean_series(actual)

    holdout_start = end_date_actual - pd.Timedelta(days=holdout_days)
    official = fetch_historical_forecast(info["lat"], info["lon"], holdout_start.date().isoformat(),
                                          end_date_actual.date().isoformat(), info["timezone"])

    rows = []
    day = holdout_start
    while day + pd.Timedelta(hours=HORIZON - 1) <= end_date_actual + pd.Timedelta(hours=23):
        history_start = day - pd.Timedelta(hours=SEQUENCE_LENGTH)
        history = actual.loc[history_start: day - pd.Timedelta(hours=1), "temperature_2m"]

        day_end = day + pd.Timedelta(hours=HORIZON - 1)
        actual_day = actual.loc[day:day_end, "temperature_2m"]
        official_day = official.loc[day:day_end, "temperature_2m"] if not official.empty else pd.Series(dtype=float)

        if len(history) == SEQUENCE_LENGTH and len(actual_day) == HORIZON and len(official_day) == HORIZON:
            model_pred = predict_next(model, scaler, history.values)
            actual_vals = actual_day.values
            official_vals = official_day.values

            rows.append({
                "date": day.date(),
                "model_mae": _mae(model_pred, actual_vals),
                "model_rmse": _rmse(model_pred, actual_vals),
                "official_mae": _mae(official_vals, actual_vals),
                "official_rmse": _rmse(official_vals, actual_vals),
            })
        day += pd.Timedelta(days=1)

    df = pd.DataFrame(rows).set_index("date")
    if not df.empty:
        df["model_win"] = df["model_mae"] < df["official_mae"]
    return df


def predict_live(city: str, info: dict, horizon_hours: int = 48) -> pd.DataFrame:
    """Our model's forecast for the next `horizon_hours`, alongside the official
    live forecast and the most recent actual observations, all on one time axis.

    horizon_hours beyond HORIZON is produced by rolling the model forward,
    feeding its own predictions back in as history (autoregressive extension).
    """
    model, scaler = load_model(city)
    live = fetch_live_forecast(info["lat"], info["lon"], info["timezone"],
                                forecast_days=max(2, -(-horizon_hours // 24)), past_days=SEQUENCE_LENGTH // 24 + 1)

    now_floor = pd.Timestamp.now(tz=None).floor("h")
    history = live.loc[:now_floor - pd.Timedelta(hours=1), "temperature_2m"].tail(SEQUENCE_LENGTH)
    official_future = live.loc[now_floor: now_floor + pd.Timedelta(hours=horizon_hours - 1), "temperature_2m"]

    history_values = history.values.copy()
    model_preds = []
    remaining = horizon_hours
    while remaining > 0:
        block = predict_next(model, scaler, history_values[-SEQUENCE_LENGTH:])
        take = min(HORIZON, remaining)
        model_preds.extend(block[:take])
        history_values = np.concatenate([history_values, block[:take]])
        remaining -= take

    future_index = pd.date_range(now_floor, periods=horizon_hours, freq="h")
    model_series = pd.Series(model_preds, index=future_index)

    full_index = history.index.union(future_index)
    return pd.DataFrame({
        "actual_recent": history.reindex(full_index),
        "official": official_future.reindex(full_index),
        "model": model_series.reindex(full_index),
    })
