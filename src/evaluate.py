import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.config import (
    METRICS_PATH,
    PLOTS_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET
)
from src.utils import get_logger, save_json, set_plot_style
from src.data_loader import load_raw_data, split_data
from src.feature_engineering import engineer_features

logger = get_logger("evaluate")

# ── Explanation of ML Metrics (Viva-Ready) ──────────────────────────────────
# 
# 1. Root Mean Squared Error (RMSE):
#    - Formula: sqrt( mean( (y_true - y_pred) ^ 2 ) )
#    - Meaning: It measures the average magnitude of the prediction error. By squaring the
#      errors before averaging, RMSE penalizes larger errors more heavily. It is expressed
#      in the same units as the target variable (kg/hectare in our case). Lower is better.
# 
# 2. Mean Absolute Error (MAE):
#    - Formula: mean( abs(y_true - y_pred) )
#    - Meaning: It measures the average absolute difference between predicted and actual values.
#      Unlike RMSE, it weights all errors linearly. It is robust to outliers and is also
#      in the target units (kg/hectare). Lower is better.
# 
# 3. R² Score (Coefficient of Determination):
#    - Formula: 1 - ( Sum of Squared Residuals / Total Sum of Squares )
#    - Meaning: It represents the proportion of variance in the target variable that is
#      explainable by the model features. R² ranges from -inf to 1.
#      An R² of 1 means perfect predictions. An R² of 0 means the model performs no better
#      than predicting the historical mean of the target. Higher is better.
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_names(pipeline):
    """
    Extracts the feature names from the column transformer preprocessor.
    This accounts for the one-hot encoded categories.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # 1. Numerical features pass through or scale, order is preserved
    num_features = NUMERICAL_FEATURES
    
    # 2. Categorical features are one-hot encoded
    try:
        onehot_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = list(onehot_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception as e:
        logger.warning(f"Could not extract categorical feature names: {e}. Using raw names.")
        cat_features = CATEGORICAL_FEATURES
        
    return num_features + cat_features

def plot_feature_importance(pipeline, model_name, output_dir):
    """Generates and saves the feature importance chart for the model."""
    logger.info("Generating feature importance plot...")
    set_plot_style()
    
    regressor = pipeline.named_steps["regressor"]
    feature_names = get_feature_names(pipeline)
    
    # Extract importances
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        logger.warning(f"Model {model_name} does not support feature importances.")
        return
        
    # Sort and take top 15
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices[:15]]
    sorted_importances = importances[indices[:15]]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title(f"Top 15 Feature Importances ({model_name})", fontsize=14, color="#E0E0E0")
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300, facecolor=plt.gcf().get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"Feature importance plot saved to: {plot_path}")

def plot_predictions(y_true, y_pred, model_name, output_dir):
    """Generates and saves a scatter plot of actual vs predicted yields."""
    logger.info("Generating actual vs predicted scatter plot...")
    set_plot_style()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color="#4CAF50", edgecolors="#2E7D32")
    
    # 45-degree diagonal reference line
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(limits, limits, color="#F44336", linestyle="--", lw=2, label="Perfect Fit")
    
    plt.title(f"Actual vs Predicted Crop Yield ({model_name})", fontsize=14, color="#E0E0E0")
    plt.xlabel("Actual Yield (kg/hectare)", fontsize=12)
    plt.ylabel("Predicted Yield (kg/hectare)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300, facecolor=plt.gcf().get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"Predictions scatter plot saved to: {plot_path}")

def run_evaluation(X_train, y_train, X_val, y_val, X_test, y_test, pipeline, model_name):
    """Evaluates the model on train, validation, and test sets and saves results."""
    logger.info("Running model evaluation on all splits...")
    
    # Make predictions
    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "train": {
            "rmse": float(root_mean_squared_error(y_train, train_pred)),
            "mae": float(mean_absolute_error(y_train, train_pred)),
            "r2": float(r2_score(y_train, train_pred))
        },
        "validation": {
            "rmse": float(root_mean_squared_error(y_val, val_pred)),
            "mae": float(mean_absolute_error(y_val, val_pred)),
            "r2": float(r2_score(y_val, val_pred))
        },
        "test": {
            "rmse": float(root_mean_squared_error(y_test, test_pred)),
            "mae": float(mean_absolute_error(y_test, test_pred)),
            "r2": float(r2_score(y_test, test_pred))
        }
    }
    
    logger.info(f"--- TEST SET RESULTS ({model_name}) ---")
    logger.info(f"RMSE: {metrics['test']['rmse']:.2f} kg/ha")
    logger.info(f"MAE:  {metrics['test']['mae']:.2f} kg/ha")
    logger.info(f"R²:   {metrics['test']['r2']:.4f}")
    
    # Save metrics to JSON
    save_json(metrics, METRICS_PATH)
    logger.info(f"Saved evaluation metrics to: {METRICS_PATH}")
    
    # Generate Plots
    plot_feature_importance(pipeline, model_name, PLOTS_DIR)
    plot_predictions(y_test, test_pred, model_name, PLOTS_DIR)
    
    # Generate metric comparison chart
    generate_metrics_chart(metrics, PLOTS_DIR)
    
    return metrics

def generate_metrics_chart(metrics, output_dir):
    """Saves a comparison chart of the train vs validation vs test scores."""
    set_plot_style()
    
    categories = ["Train", "Validation", "Test"]
    rmses = [metrics["train"]["rmse"], metrics["validation"]["rmse"], metrics["test"]["rmse"]]
    maes = [metrics["train"]["mae"], metrics["validation"]["mae"], metrics["test"]["mae"]]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, rmses, width, label="RMSE", color="#2196F3")
    rects2 = ax1.bar(x + width/2, maes, width, label="MAE", color="#FF9800")
    
    ax1.set_ylabel("Error Metric Value (kg/ha)", fontsize=12)
    ax1.set_title("Model Error Metrics Across Splitting Stages", fontsize=14, color="#E0E0E0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", color="#B0B0B0", fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(plot_path, dpi=300, facecolor=plt.gcf().get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"Metrics comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    # If run directly, run evaluation on current saved model
    from src.utils import load_pickle
    from src.config import MODEL_PATH
    
    # Load model
    if os.path.exists(MODEL_PATH):
        pipeline = load_pickle(MODEL_PATH)
        df_raw = load_raw_data()
        df_engineered = engineer_features(df_raw, is_training=True)
        train_df, val_df, test_df = split_data(df_engineered)
        
        features_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        run_evaluation(
            train_df[features_cols], train_df[TARGET],
            val_df[features_cols], val_df[TARGET],
            test_df[features_cols], test_df[TARGET],
            pipeline,
            pipeline.named_steps["regressor"].__class__.__name__
        )
    else:
        logger.error(f"Saved model not found at {MODEL_PATH}. Run train.py first.")
