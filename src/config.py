import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

DATA_PATH = os.path.join(DATA_DIR, "crop_production.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")
HISTORY_PATH = os.path.join(DATA_DIR, "search_history.csv")

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, NOTEBOOKS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ── ML Settings ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Time-based splitting thresholds
# Training: <= 2015 (older data)
# Validation: 2016 - 2018 (recent data)
# Testing: 2019 - 2021 (future data)
TRAIN_YEAR_MAX = 2015
VAL_YEAR_MAX = 2018

# Columns in raw data
RAW_COLS = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area",
    "Production"
]

# Feature definitions for modeling
CATEGORICAL_FEATURES = ["State_Name", "Season", "Crop"]
NUMERICAL_FEATURES = [
    "Crop_Year",
    "Area",
    "temperature",
    "rainfall",
    "humidity",
    "temp_precip_interaction",
    "humidity_scaled"
]
TARGET = "Yield"

# ── Feature Engineering Weather Defaults ───────────────────────────────────────
# Normal/typical weather constants per state to generate realistic environments
STATE_WEATHER_PROFILES = {
    "Andhra Pradesh":  {"temp": 32.0, "rain": 950.0,  "hum": 72.0},
    "Bihar":           {"temp": 27.0, "rain": 1100.0, "hum": 72.0},
    "Chhattisgarh":    {"temp": 28.0, "rain": 1300.0, "hum": 74.0},
    "Gujarat":         {"temp": 31.0, "rain": 750.0,  "hum": 60.0},
    "Haryana":         {"temp": 26.0, "rain": 650.0,  "hum": 58.0},
    "Karnataka":       {"temp": 27.0, "rain": 1000.0, "hum": 75.0},
    "Madhya Pradesh":  {"temp": 30.0, "rain": 1100.0, "hum": 68.0},
    "Maharashtra":     {"temp": 29.0, "rain": 1200.0, "hum": 70.0},
    "Odisha":          {"temp": 29.0, "rain": 1500.0, "hum": 76.0},
    "Punjab":          {"temp": 24.0, "rain": 700.0,  "hum": 60.0},
    "Rajasthan":       {"temp": 33.0, "rain": 400.0,  "hum": 42.0},
    "Tamil Nadu":      {"temp": 31.0, "rain": 1150.0, "hum": 78.0},
    "Telangana":       {"temp": 31.0, "rain": 950.0,  "hum": 68.0},
    "Uttar Pradesh":   {"temp": 28.0, "rain": 900.0,  "hum": 65.0},
    "West Bengal":     {"temp": 28.0, "rain": 1600.0, "hum": 80.0}
}

# Base yield ranges for crops (to make synthetic yield generation highly realistic)
CROP_BASE_YIELDS = {
    "Rice":        {"base_yield": 2200, "sd": 400,  "seasons": ["Kharif"]},
    "Wheat":       {"base_yield": 2800, "sd": 350,  "seasons": ["Rabi"]},
    "Maize":       {"base_yield": 2000, "sd": 300,  "seasons": ["Kharif", "Rabi"]},
    "Sugarcane":   {"base_yield": 65000,"sd": 8000, "seasons": ["Whole Year"]},
    "Cotton":      {"base_yield": 450,  "sd": 80,   "seasons": ["Kharif"]},
    "Groundnut":   {"base_yield": 1200, "sd": 200,  "seasons": ["Kharif", "Rabi"]},
    "Soyabean":    {"base_yield": 1000, "sd": 180,  "seasons": ["Kharif"]},
    "Bajra":       {"base_yield": 900,  "sd": 150,  "seasons": ["Kharif"]},
    "Jowar":       {"base_yield": 850,  "sd": 130,  "seasons": ["Kharif", "Rabi"]},
    "Potato":      {"base_yield": 19000,"sd": 3000, "seasons": ["Rabi"]},
    "Mustard":     {"base_yield": 1100, "sd": 160,  "seasons": ["Rabi"]},
    "Sunflower":   {"base_yield": 750,  "sd": 120,  "seasons": ["Kharif", "Rabi"]},
    "Turmeric":    {"base_yield": 5500, "sd": 700,  "seasons": ["Kharif"]},
    "Chickpea":    {"base_yield": 900,  "sd": 140,  "seasons": ["Rabi"]},
    "Lentil":      {"base_yield": 850,  "sd": 120,  "seasons": ["Rabi"]},
}

# Seasons mapping
SEASONS = ["Kharif", "Rabi", "Whole Year"]

# ── Model Search Grids ─────────────────────────────────────────────────────────
RF_PARAM_GRID = {
    "regressor__n_estimators": [50, 100, 150, 200],
    "regressor__max_depth": [5, 10, 15, 20, None],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
    "regressor__max_features": ["sqrt", "log2", None]
}

XGB_PARAM_GRID = {
    "regressor__n_estimators": [50, 100, 150, 200],
    "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "regressor__max_depth": [3, 5, 7, 9],
    "regressor__subsample": [0.6, 0.8, 1.0],
    "regressor__colsample_bytree": [0.6, 0.8, 1.0],
    "regressor__min_child_weight": [1, 3, 5]
}
