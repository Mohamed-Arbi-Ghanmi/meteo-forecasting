"""Turn a raw hourly temperature series into supervised sliding-window sequences.

Pipeline: clean (fill small gaps) -> train/holdout split by date -> scale
(fit on train only, to avoid leakage) -> window into (history -> next H hours).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def clean_series(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    """Reindex to a strict hourly grid and linearly interpolate small gaps."""
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index)
    df["temperature_2m"] = df["temperature_2m"].interpolate(limit=max_gap_hours, limit_direction="both")
    df = df.dropna(subset=["temperature_2m"])
    return df


def train_holdout_split(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df.index.max() - pd.Timedelta(days=holdout_days)
    return df[df.index <= cutoff], df[df.index > cutoff]


class Scaler:
    """Minimal standardizer (mean/std), fit on training data only."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, values: np.ndarray) -> "Scaler":
        return cls(mean=float(values.mean()), std=float(values.std()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(mean=d["mean"], std=d["std"])


def make_sequences(values: np.ndarray, seq_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a (seq_len -> horizon) window across a 1D array of scaled values."""
    n = len(values) - seq_len - horizon + 1
    if n <= 0:
        raise ValueError(f"Series too short ({len(values)}) for seq_len={seq_len}, horizon={horizon}")
    X = np.empty((n, seq_len), dtype=np.float32)
    y = np.empty((n, horizon), dtype=np.float32)
    for i in range(n):
        X[i] = values[i : i + seq_len]
        y[i] = values[i + seq_len : i + seq_len + horizon]
    return X[..., None], y  # X gets a trailing feature dim of 1


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
