# Can I out-forecast the weathermen?

A small PyTorch LSTM, trained only on a city's historical hourly temperature,
going head-to-head with [Open-Meteo](https://open-meteo.com/)'s official
forecast — live, on Streamlit.

## What it does

- **Forecast tab** — pick a city, see our LSTM's next 24-48h prediction plotted
  against Open-Meteo's official forecast and recent actual observations.
- **Scoreboard tab** — a 45-day rolling backtest: for each day, the model's
  prediction (seeded from real history) and the official forecast are both
  scored against what actually happened, with a daily MAE chart and a
  "beat the forecast N of the last M days" tally.
- **About tab** — architecture and methodology, including the honest caveat
  that this is a single-signal sequence model, not a numerical weather
  prediction system.

## How it works

- **Data**: Open-Meteo's archive API (historical observations), historical-forecast
  API (reconstructed past forecasts, used as the backtest baseline), and live
  forecast API. No API key required.
- **Model**: a 2-layer LSTM per city, mapping the last 72 hours of temperature
  to the next 24 hours in one shot (longer horizons roll the model forward
  autoregressively). Trained on ~3 years of hourly data per city.
- **Shipped pretrained**: weights for Tunis, Paris, London, New York, and Tokyo
  live in `models/`. Nothing trains at request time — the app only runs
  inference plus live API calls.

## Project layout

```
config.py            city list, hyperparameters, API URLs
src/data_fetcher.py   Open-Meteo API wrappers
src/preprocessing.py  cleaning, scaling, sliding-window sequences
src/model.py           the LSTM
src/train.py           training script (python -m src.train [--city NAME])
src/evaluate.py        live prediction + historical backtest
app.py                 the Streamlit app
models/                 pretrained weights + scaler stats, per city
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain (e.g. to add a city — add it to `CITIES` in `config.py` first):

```bash
python -m src.train --city "Berlin"
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), straight
from this repo. No secrets required, Open-Meteo needs no API key.
