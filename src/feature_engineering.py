import numpy as np
import pandas as pd
from src.config import STATE_WEATHER_PROFILES, RANDOM_STATE
from src.utils import get_logger

logger = get_logger("feature_engineering")

def add_target_and_remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Yield = (Production * 1000) / Area, drops missing target values,
    and removes Production to prevent data leakage.
    
    Why: Production is only known AFTER harvesting. If we use Production to
    predict Yield, our model will have 100% accuracy in training but will be
    completely useless before harvesting when Production is unknown. This is
    a textbook example of data leakage.
    """
    df_clean = df.copy()
    
    # Drop rows where Area or Production is null or zero
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.dropna(subset=["Area", "Production"])
    df_clean = df_clean[(df_clean["Area"] > 0) & (df_clean["Production"] >= 0)]
    dropped_rows = initial_rows - df_clean.shape[0]
    
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows due to missing/zero Area or Production values.")
        
    # Calculate Yield: Production (tonnes) * 1000 (kg/tonne) / Area (hectares) = kg/hectare
    df_clean["Yield"] = (df_clean["Production"] * 1000.0) / df_clean["Area"]
    
    # Drop original Production column to strictly prevent leakage
    df_clean = df_clean.drop(columns=["Production"])
    
    return df_clean

def generate_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates realistic weather features (temperature, rainfall, humidity) 
    using climate rules based on State_Name, Crop_Year, and Season.
    
    Physics and agricultural logic behind weather generation:
    1. Temperature:
       - Uses state baseline (e.g., Rajasthan is hot, Punjab is cooler).
       - Adds a cyclic year trend representing macro-climate fluctuations (using sine wave over years).
       - Adds seasonal adjustment (Kharif is summer/monsoon-hot, Rabi is winter-cool).
       - Includes gaussian noise for micro-climate variation.
    2. Rainfall:
       - Heavily seasonal (Kharif/Monsoon gets ~80% of annual rain, Rabi gets minimal winter rain).
       - Uses state baseline (West Bengal is high rain, Rajasthan is dry).
       - Includes noise to simulate drought or heavy monsoon years.
    3. Humidity:
       - Correlated with rainfall and temperature.
       - Higher rainfall and lower temperatures increase humidity.
       - Clipped between 10% and 100%.
    """
    logger.info("Generating realistic environmental/weather features inside code...")
    df_out = df.copy()
    
    # Set seed for reproducibility during dataset generation
    np.random.seed(RANDOM_STATE)
    
    temperatures = []
    rainfalls = []
    humidities = []
    
    for idx, row in df_out.iterrows():
        state = row["State_Name"]
        season = row["Season"]
        year = row["Crop_Year"]
        
        # Get baseline weather for this state. Fallback to average values if state is unknown.
        base = STATE_WEATHER_PROFILES.get(state, {"temp": 28.0, "rain": 1000.0, "hum": 65.0})
        
        # 1. Temperature: 25 + sin(year trend) + season adjustment + noise
        # Year trend is cyclic. We subtract 2000 to center it.
        year_trend = np.sin((year - 2000) / 3.0) 
        
        # Season adjustment: Kharif (hot/rainy), Rabi (winter/cool), Whole Year (moderate)
        season_temp_adj = 2.0 if season == "Kharif" else (-4.0 if season == "Rabi" else 0.0)
        
        # Micro-climate noise
        temp_noise = np.random.normal(0, 1.5)
        
        temp = base["temp"] + year_trend + season_temp_adj + temp_noise
        temperatures.append(round(temp, 1))
        
        # 2. Rainfall: Season-based baseline * state multiplier + noise
        # Kharif is monsoon (heaviest), Rabi is winter (dry), Whole Year is moderate
        season_rain_mult = 1.2 if season == "Kharif" else (0.2 if season == "Rabi" else 0.8)
        
        # Base rainfall modified by state characteristics
        expected_rain = base["rain"] * season_rain_mult
        rain_noise = np.random.normal(0, 100.0)
        
        rain = max(50.0, expected_rain + rain_noise)
        rainfalls.append(round(rain, 1))
        
        # 3. Humidity: Positively correlated with rainfall, capped at 100%
        # High rain -> high humidity, baseline humidity per state + season noise
        season_hum_adj = 10.0 if season == "Kharif" else (-5.0 if season == "Rabi" else 0.0)
        expected_hum = base["hum"] + (rain - base["rain"]) * 0.02 + season_hum_adj
        hum_noise = np.random.normal(0, 5.0)
        
        hum = np.clip(expected_hum + hum_noise, 15.0, 100.0)
        humidities.append(round(hum, 1))
        
    df_out["temperature"] = temperatures
    df_out["rainfall"] = rainfalls
    df_out["humidity"] = humidities
    
    return df_out

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived/engineered features:
    1. temp_precip_interaction = temperature * rainfall
       - Why: High temperature combined with high rainfall leads to humid tropical conditions,
              which increases yields for crops like rice and sugarcane but can cause pests.
              Conversely, high temp with low rain leads to drought. This captures the non-linear interaction.
    2. humidity_scaled = humidity / 100
       - Why: Streamlines humidity into a decimal ratio (0.0 to 1.0) which is standard for relative humidity modeling
              and scales it appropriately for gradient boosters and distance-based scalers.
    """
    df_out = df.copy()
    
    # 1. Temperature-Precipitation Interaction
    df_out["temp_precip_interaction"] = df_out["temperature"] * df_out["rainfall"]
    
    # 2. Humidity Scaled
    df_out["humidity_scaled"] = df_out["humidity"] / 100.0
    
    return df_out

def engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Runs the full feature engineering pipeline."""
    df_out = df.copy()
    
    if is_training:
        # Calculate yield and remove production (leakage prevention)
        df_out = add_target_and_remove_leakage(df_out)
        # Generate weather features
        df_out = generate_weather_features(df_out)
        
    # Derived features
    df_out = add_derived_features(df_out)
    
    return df_out
