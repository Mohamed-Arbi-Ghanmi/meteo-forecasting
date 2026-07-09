"""Thin wrappers around Open-Meteo's keyless REST APIs.

Three endpoints, three purposes:
- archive-api:            ground-truth observed temperatures (training + "actuals")
- historical-forecast-api: reconstructed past forecasts, used as the official
                            baseline when backtesting the scoreboard
- api (live forecast):     the current official forecast, shown next to our model
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from config import ARCHIVE_URL, FORECAST_URL, GEOCODING_URL, HISTORICAL_FORECAST_URL

_TIMEOUT = 20


def _hourly_temperature_frame(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
    })
    return df.set_index("time")


def fetch_archive(lat: float, lon: float, start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    """Observed hourly temperature between start_date and end_date (inclusive, YYYY-MM-DD)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": timezone,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return _hourly_temperature_frame(resp.json())


def fetch_historical_forecast(lat: float, lon: float, start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    """What the official forecast would have said, reconstructed for past dates."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": timezone,
    }
    resp = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return _hourly_temperature_frame(resp.json())


def fetch_live_forecast(lat: float, lon: float, timezone: str, forecast_days: int = 2,
                         past_days: int = 3) -> pd.DataFrame:
    """Current official forecast, plus a few trailing days of observed data for context."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "timezone": timezone,
        "forecast_days": forecast_days,
        "past_days": past_days,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return _hourly_temperature_frame(resp.json())


def geocode_city(name: str, count: int = 5) -> list[dict]:
    """Look up lat/lon/timezone candidates for a free-text city name."""
    params = {"name": name, "count": count, "language": "en", "format": "json"}
    resp = requests.get(GEOCODING_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [
        {
            "name": r["name"],
            "country": r.get("country", ""),
            "admin1": r.get("admin1", ""),
            "lat": r["latitude"],
            "lon": r["longitude"],
            "timezone": r["timezone"],
        }
        for r in results
    ]


def date_range_years_ago(years: float, end_offset_days: int = 0) -> tuple[str, str]:
    """Helper: (start_date, end_date) strings spanning `years` back from today minus offset."""
    end = dt.date.today() - dt.timedelta(days=end_offset_days)
    start = end - dt.timedelta(days=int(years * 365.25))
    return start.isoformat(), end.isoformat()
