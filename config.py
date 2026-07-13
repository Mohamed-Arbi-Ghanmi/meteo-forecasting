"""Shared constants: cities, model hyperparameters, paths."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Cities shipped with pretrained weights. lat/lon/timezone feed Open-Meteo directly.
CITIES = {
    "Tunis":     {"lat": 36.8065, "lon": 10.1815, "timezone": "Africa/Tunis"},
    "Paris":     {"lat": 48.8566, "lon": 2.3522,  "timezone": "Europe/Paris"},
    "London":    {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
    "New York":  {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
    "Tokyo":     {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
    "Sfax":      {"lat": 34.7406, "lon": 10.7603, "timezone": "Africa/Tunis"},
    "Sousse":    {"lat": 35.8254, "lon": 10.6370, "timezone": "Africa/Tunis"},
    "Bizerte":   {"lat": 37.2744, "lon": 9.8739,  "timezone": "Africa/Tunis"},
    "Gabès":     {"lat": 33.8815, "lon": 10.0982, "timezone": "Africa/Tunis"},
}

# Model / windowing hyperparameters
SEQUENCE_LENGTH = 72   # hours of history fed to the model
HORIZON = 24            # hours predicted ahead
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3

TRAIN_YEARS = 3          # years of hourly history to train on
HOLDOUT_DAYS = 45        # most recent days reserved for evaluation/scoreboard
