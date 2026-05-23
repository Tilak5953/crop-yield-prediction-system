import os
import pandas as pd
import numpy as np
from src.config import (
    DATA_PATH,
    STATE_WEATHER_PROFILES,
    CROP_BASE_YIELDS,
    TRAIN_YEAR_MAX,
    VAL_YEAR_MAX,
    RAW_COLS,
    RANDOM_STATE
)
from src.utils import get_logger

logger = get_logger("data_loader")

def generate_crop_production_dataset(n_records: int = 15000) -> pd.DataFrame:
    """
    Generates a realistic crop production dataset representing the Government of India Open Data.
    Columns: State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
    """
    logger.info(f"Generating synthetic Crop Production dataset with {n_records} records...")
    np.random.seed(RANDOM_STATE)
    
    states = list(STATE_WEATHER_PROFILES.keys())
    crops = list(CROP_BASE_YIELDS.keys())
    years = list(range(2001, 2022)) # 2001 to 2021
    
    rows = []
    for i in range(n_records):
        state = np.random.choice(states)
        # Simple district generation based on state
        district = f"{state}_District_{np.random.randint(1, 6)}"
        crop = np.random.choice(crops)
        crop_cfg = CROP_BASE_YIELDS[crop]
        season = np.random.choice(crop_cfg["seasons"])
        year = np.random.choice(years)
        
        # Area in hectares
        area = np.random.uniform(50.0, 5000.0)
        
        # Determine crop yield based on crop type, year trend (technology advancement), and some noise
        # This will later be reconstructed as Yield = Production / Area
        base_yield = crop_cfg["base_yield"]
        sd = crop_cfg["sd"]
        
        # Mild linear year trend (yields increase slightly over years due to tech)
        trend = 1.0 + 0.005 * (year - 2001)
        
        # State variation multiplier
        state_weather = STATE_WEATHER_PROFILES[state]
        # Better rainfall/temp increases yield slightly
        state_mult = 0.95 + 0.05 * (state_weather["rain"] / 1000.0)
        
        # Compute realistic yield with normal distribution
        expected_yield = base_yield * trend * state_mult
        generated_yield = max(10.0, np.random.normal(expected_yield, sd))
        
        # Production in tonnes: (Yield in kg/ha * Area in ha) / 1000
        # Yield = Production (tonnes) * 1000 / Area
        # But wait! If standard Yield is kg/ha, let's keep units matching:
        # Production (in tonnes) = (Yield (kg/ha) * Area (ha)) / 1000
        # Let's verify: Production = Yield * Area / 1000
        # So Yield = Production * 1000 / Area.
        # Let's generate Production (tonnes) based on this:
        production = (generated_yield * area) / 1000.0
        
        # Introduce a few missing production values (about 1%) to simulate real missing data
        if np.random.rand() < 0.01:
            production = np.nan
            
        rows.append({
            "State_Name": state,
            "District_Name": district,
            "Crop_Year": year,
            "Season": season,
            "Crop": crop,
            "Area": round(area, 2),
            "Production": round(production, 2) if not pd.isna(production) else np.nan
        })
        
    df = pd.DataFrame(rows)
    
    # Save the file
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Dataset generated and saved successfully to: {DATA_PATH}")
    return df

def load_raw_data() -> pd.DataFrame:
    """Loads raw crop production dataset, generating it first if missing."""
    if not os.path.exists(DATA_PATH):
        logger.warning(f"Dataset not found at {DATA_PATH}. Triggering generation...")
        df = generate_crop_production_dataset()
    else:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset loaded successfully from {DATA_PATH}. Shape: {df.shape}")
        
    # Verify columns
    for col in RAW_COLS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in the dataset.")
            
    return df

def split_data(df: pd.DataFrame):
    """
    Performs time-based split of the dataset.
    - Training: Crop_Year <= TRAIN_YEAR_MAX (e.g., 2015)
    - Validation: TRAIN_YEAR_MAX < Crop_Year <= VAL_YEAR_MAX (e.g., 2016-2018)
    - Testing: Crop_Year > VAL_YEAR_MAX (e.g., 2019-2021)
    
    Explanation of time-based split vs random split:
    A random split causes temporal leakage because weather patterns, climate trends,
    and technological progress are correlated in time. If we randomly select records,
    the model will 'peek' into the future to predict the past or vice-versa, leading
    to overoptimistic test scores and poor generalization when deployed in the future.
    A time-based split tests the model's true forecasting ability on unseen future years.
    """
    logger.info("Performing chronological time-based train-validation-test split...")
    
    train_df = df[df["Crop_Year"] <= TRAIN_YEAR_MAX].copy()
    val_df = df[(df["Crop_Year"] > TRAIN_YEAR_MAX) & (df["Crop_Year"] <= VAL_YEAR_MAX)].copy()
    test_df = df[df["Crop_Year"] > VAL_YEAR_MAX].copy()
    
    logger.info(f"Train split: {train_df.shape[0]} rows (Years <= {TRAIN_YEAR_MAX})")
    logger.info(f"Validation split: {val_df.shape[0]} rows (Years {TRAIN_YEAR_MAX+1} - {VAL_YEAR_MAX})")
    logger.info(f"Test split: {test_df.shape[0]} rows (Years > {VAL_YEAR_MAX})")
    
    return train_df, val_df, test_df
