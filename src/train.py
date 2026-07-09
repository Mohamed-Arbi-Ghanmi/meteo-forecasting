"""Train one LSTM per city on Open-Meteo archive data and ship the weights.

Usage:
    python -m src.train                # train all cities in config.CITIES
    python -m src.train --city Tunis    # train a single city
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, CITIES, DROPOUT, EPOCHS, HIDDEN_SIZE, HOLDOUT_DAYS,
    HORIZON, LEARNING_RATE, MODELS_DIR, NUM_LAYERS, SEQUENCE_LENGTH, TRAIN_YEARS,
)
from src.data_fetcher import date_range_years_ago, fetch_archive
from src.model import TemperatureLSTM
from src.preprocessing import Scaler, clean_series, make_sequences, train_holdout_split, SequenceDataset

VAL_FRACTION = 0.1  # tail slice of the training period held out for early-stopping checks


def train_one_city(city: str, info: dict) -> None:
    print(f"\n=== {city} ===")
    start, end = date_range_years_ago(TRAIN_YEARS, end_offset_days=0)
    raw = fetch_archive(info["lat"], info["lon"], start, end, info["timezone"])
    raw = clean_series(raw)

    train_df, _holdout_df = train_holdout_split(raw, HOLDOUT_DAYS)

    val_cutoff = train_df.index.max() - (train_df.index.max() - train_df.index.min()) * VAL_FRACTION
    fit_df = train_df[train_df.index <= val_cutoff]
    val_df = train_df[train_df.index > val_cutoff]

    scaler = Scaler.fit(fit_df["temperature_2m"].values)

    fit_scaled = scaler.transform(fit_df["temperature_2m"].values)
    val_scaled = scaler.transform(val_df["temperature_2m"].values)

    X_train, y_train = make_sequences(fit_scaled, SEQUENCE_LENGTH, HORIZON)
    X_val, y_val = make_sequences(val_scaled, SEQUENCE_LENGTH, HORIZON)

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = TemperatureLSTM(HIDDEN_SIZE, NUM_LAYERS, HORIZON, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:>3}/{EPOCHS}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

    print(f"trained in {time.time() - t0:.1f}s, best val_mse={best_val_loss:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / f"{city}.pt")
    with open(MODELS_DIR / f"{city}_scaler.json", "w") as f:
        json.dump(scaler.to_dict(), f)
    print(f"saved models/{city}.pt and scaler")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", help="Train a single city (default: all cities in config.CITIES)")
    args = parser.parse_args()

    targets = {args.city: CITIES[args.city]} if args.city else CITIES
    for city, info in targets.items():
        train_one_city(city, info)


if __name__ == "__main__":
    main()
