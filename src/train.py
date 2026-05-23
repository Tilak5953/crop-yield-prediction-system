import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import (
    MODEL_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    RF_PARAM_GRID,
    XGB_PARAM_GRID,
    RANDOM_STATE
)
from src.utils import get_logger, save_pickle
from src.data_loader import load_raw_data, split_data
from src.feature_engineering import engineer_features
from src.preprocessing import get_preprocessing_pipeline

logger = get_logger("train")

def calculate_rmse(y_true, y_pred):
    """Utility to calculate RMSE."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def run_training():
    """Executes the full model training and hyperparameter tuning pipeline."""
    logger.info("Starting the training process...")
    
    # 1. Load Data
    df_raw = load_raw_data()
    
    # 2. Feature Engineering (Generates Yield and weather features)
    df_engineered = engineer_features(df_raw, is_training=True)
    
    # 3. Chronological Time-based Split
    train_df, val_df, test_df = split_data(df_engineered)
    
    # Separate features and target
    features_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    
    X_train = train_df[features_cols]
    y_train = train_df[TARGET]
    
    X_val = val_df[features_cols]
    y_val = val_df[TARGET]
    
    X_test = test_df[features_cols]
    y_test = test_df[TARGET]
    
    logger.info(f"Target variable summary (Yield - Train): Mean={y_train.mean():.2f}, Std={y_train.std():.2f}")
    
    # 4. Get Preprocessing Pipeline
    preprocessor = get_preprocessing_pipeline()
    
    # We will concatenate train and validation sets for RandomizedSearchCV
    # and use PredefinedSplit to ensure training is on train, and validation is on val.
    X_train_val = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val]).reset_index(drop=True)
    
    # -1 indicates training indices, 0 indicates validation index
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)
    
    # 5. Define pipelines
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=RANDOM_STATE))
    ])
    
    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"))
    ])
    
    # 6. Hyperparameter Tuning
    logger.info("Tuning Random Forest Regressor using RandomizedSearchCV...")
    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=RF_PARAM_GRID,
        n_iter=5,
        cv=pds,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_search.fit(X_train_val, y_train_val)
    # The score returned is negative RMSE, convert to positive
    rf_val_rmse = -rf_search.best_score_
    logger.info(f"Best Random Forest params: {rf_search.best_params_}")
    logger.info(f"Random Forest Validation RMSE: {rf_val_rmse:.2f}")
    
    logger.info("Tuning XGBoost Regressor using RandomizedSearchCV...")
    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=XGB_PARAM_GRID,
        n_iter=5,
        cv=pds,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_search.fit(X_train_val, y_train_val)
    xgb_val_rmse = -xgb_search.best_score_
    logger.info(f"Best XGBoost params: {xgb_search.best_params_}")
    logger.info(f"XGBoost Validation RMSE: {xgb_val_rmse:.2f}")
    
    # 7. Model Selection
    logger.info("Selecting the best model based on validation RMSE...")
    if rf_val_rmse < xgb_val_rmse:
        logger.info(f"Random Forest selected as Best Model (RMSE: {rf_val_rmse:.2f} < XGBoost: {xgb_val_rmse:.2f})")
        best_search = rf_search
        best_model_name = "RandomForest"
    else:
        logger.info(f"XGBoost selected as Best Model (RMSE: {xgb_val_rmse:.2f} < RF: {rf_val_rmse:.2f})")
        best_search = xgb_search
        best_model_name = "XGBoost"
        
    best_pipeline = best_search.best_estimator_
    
    # Retrain on the complete train+val set using best hyperparameters to capture all data
    # (Optional but recommended; here we just use the fitted estimator from best_search,
    # which has already been fitted on the combined set because RandomizedSearchCV fits the
    # final best estimator on the entire dataset passed to it, i.e. X_train_val!)
    logger.info("Best estimator is automatically fitted on combined train and validation sets.")
    
    # Save the selected best pipeline
    save_pickle(best_pipeline, MODEL_PATH)
    logger.info(f"Saved the best pipeline ({best_model_name}) to: {MODEL_PATH}")
    
    # Also return test sets for evaluation
    return X_train, y_train, X_val, y_val, X_test, y_test, best_pipeline, best_model_name

if __name__ == "__main__":
    run_training()
