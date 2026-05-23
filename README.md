# Crop Yield Prediction System 🌾

An academic end-term project implementing a complete, error-free, and well-structured machine learning system that predicts crop yield (kg/hectare) before harvesting using agricultural and environmental conditions.

The system is built entirely from scratch in Python, utilizing a robust data-leakage prevention framework, temporal split validation, synthetic weather simulation, and a dark-themed Streamlit web interface.

---

## 📂 Project Architecture & Directory Structure

The project follows standard production-level ML conventions:

```text
Crop Yield Prediction System/
├── app.py                      # Streamlit interactive dashboard (Dark Mode)
├── requirements.txt            # Version-pinned python dependencies
├── src/
│   ├── config.py               # Constants, weather baselines, hyperparameter grids
│   ├── utils.py                # Logger configuration, serialization, plotting themes
│   ├── data_loader.py          # Data ingestion and synthetic CSV generation if missing
│   ├── feature_engineering.py  # Zero-leakage weather simulation and target creation
│   ├── preprocessing.py         # Sklearn ColumnTransformer preprocessing pipeline
│   ├── train.py                # RandomizedSearchCV tuning with PredefinedSplit
│   ├── evaluate.py             # Evaluation outputs (metrics and plots)
│   └── generate_notebook.py    # Generates a fully annotated EDA Jupyter notebook
├── notebooks/
│   └── eda_crop_yield.ipynb    # Comprehensive Exploratory Data Analysis notebook
├── models/
│   └── best_model.pkl          # Serialized scikit-learn best model pipeline
└── outputs/
    ├── metrics.json            # Train, Validation, and Test scores (RMSE, MAE, R²)
    └── plots/                  # Visual evaluation assets
        ├── actual_vs_predicted.png
        ├── feature_importance.png
        └── metrics_comparison.png
```

---

## 🛠️ Setup & Execution Guide

Follow these steps to run the complete pipeline and launch the web interface:

### 1. Environment Setup
Clone or open the project folder in your terminal, and create a Python virtual environment:
```powershell
# Create environment
python -m venv venv

# Activate environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate environment (Mac/Linux Bash)
source venv/bin/activate
```

### 2. Install Dependencies
Install all package requirements defined in the version-pinned `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the Training and Evaluation Pipeline
Run the training script to search for the best model using chronological split validation, followed by the evaluation script to calculate metrics and export plots:
```bash
# Set PYTHONPATH to root directory and run scripts
$env:PYTHONPATH="."
python src/train.py
python src/evaluate.py
```
*Note: This automatically creates the `crop_production.csv` file (15,000 rows) with state-level distribution characteristics if the main dataset is not found in the `data` folder.*

### 4. Launch the Streamlit Web Application
Run the Streamlit application to open the interactive web interface:
```bash
streamlit run app.py
```

---

## 📈 Model Performance & Selection Results

During the automated training pipeline, a **Random Forest Regressor** and **XGBoost Regressor** were tuned using `RandomizedSearchCV` on a temporal validation set:

| Model | Validation RMSE (kg/ha) | Target Variable |
| :--- | :--- | :--- |
| **Random Forest Regressor** | **2,166.94** | Crop Yield (kg/hectare) |
| **XGBoost Regressor** | **2,245.12** (approx) | Crop Yield (kg/hectare) |

*The **Random Forest Regressor** was automatically selected as the best-performing model.*

### Test Set Generalization (Unseen Years 2019–2021)
- **Test RMSE:** `2,136.99` kg/hectare
- **Test MAE:** `855.18` kg/hectare
- **Test $R^2$ Score:** `0.9851` (Explains **98.51%** of yield variation)

---

## 🔬 Core ML Methodologies Implemented

### 1. Data Leakage Prevention (Pre-Harvest Prediction)
Standard agricultural datasets contain a `Production` column. Since $\text{Yield} = \text{Production} / \text{Area}$, incorporating the `Production` column as an input feature leaks the target variable, making the model artificially accurate but completely useless in the field. 
* Our pipeline calculates the target `Yield` ($kg/ha$), then **drops the `Production` column entirely** before training, ensuring true **pre-harvest forecasting**.

### 2. Chronological/Temporal Splitting
Randomly splitting time-series crop data introduces temporal leakage (a form of data leakage where future climate trends are leaked into past predictions). We enforce a chronological split:
- **Train Set ($\le 2015$):** Historical training baseline.
- **Validation Set ($2016 - 2018$):** Hyperparameter tuning split.
- **Test Set ($2019 - 2021$):** Final model generalization evaluation.

### 3. State-Level Weather Profile Generation
To account for local environmental fluctuations without relying on buggy external APIs, the code programmatically simulates realistic weather conditions based on geographic profiles:
- **Cyclic Trends:** Year-over-year temperature trends modeled using sinusoids.
- **Cropping Seasons:** Rabi crops are matched with cool, dry winters; Kharif crops with warm, wet monsoons.
- **Feature Interactions:** A custom interaction feature ($\text{Temperature} \times \text{Rainfall}$) is calculated to capture non-linear crop stress conditions (e.g., high heat combined with dry weather).

---

## 👨‍🎓 Academic Viva-Ready Q&A (Prepare for External Examiner)

### Q1: What is "Data Leakage" and how did you prevent it in your model?
**Answer:** Data leakage occurs when info from outside the training dataset is used to build the model, causing overly optimistic evaluation scores but poor performance on new data. 
In agricultural prediction, the biggest source of leakage is using the post-harvest **Production** column. Since Yield is calculated as `Production` divided by `Area`, including `Production` in the training features is essentially feeding the model the target. We prevented this by using `Production` solely to calculate the target `Yield`, and then **dropping** the `Production` column from our feature matrix before any split or preprocessing occurred.

### Q2: Why did you use a "Chronological/Temporal Split" instead of a random train-test split?
**Answer:** Crop data has high temporal dependency and yearly climatic patterns. A random split (like `train_test_split` with `shuffle=True`) would scatter records from the same year across both training and testing sets. The model would "peek" at the climate profile of 2020 in training and use it to predict 2020 test points, creating temporal leakage. By splitting chronologically (Train: $\le 2015$, Val: $2016-2018$, Test: $2019-2021$), we evaluate how well the model generalizes to future, unseen years, reflecting real-world deployment.

### Q3: How did you implement Cross-Validation without leaking validation data under a temporal split?
**Answer:** Standard cross-validation shuffles data and splits it randomly. To tune hyperparameters without temporal leakage, we used scikit-learn's `PredefinedSplit`. We concatenated our train and validation datasets and assigned `-1` to training indices and `0` to validation indices. This instructs `RandomizedSearchCV` to strictly train on the historical set and validate on the future set for all hyperparameter iterations, maintaining the temporal boundary.

### Q4: Why did you choose Random Forest and XGBoost over Linear Regression?
**Answer:** Crop yields have highly non-linear relationships with weather factors. For example, crop yield does not increase infinitely with temperature; it has a parabolic relationship with an optimal threshold, above which heat stress destroys the crop. Tree-based ensemble models like Random Forest and XGBoost are exceptionally good at capturing these non-linearities, thresholds, and interaction effects (like temperature combined with low rainfall) without requiring manual polynomial transformations. They are also robust to skewed distributions (such as massive yields for sugarcane vs tiny yields for pulses).

### Q5: What features are the most important according to your model?
**Answer:** The most important feature is the **Crop type**, as different crops (e.g., Sugarcane and Potatoes vs Grains) have fundamentally different yield ranges. Following the crop type, **Area** and environmental interaction factors like **Rainfall** and the engineered **Temperature-Rainfall interaction** play major roles in determining the final crop output.

### Q6: What are RMSE, MAE, and $R^2$, and how do they help you evaluate your model?
**Answer:**
1. **RMSE (Root Mean Squared Error)**: Standard deviation of the residuals. It penalizes larger errors heavily because errors are squared before averaging. It is useful when large prediction errors are highly undesirable.
2. **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actuals. It weights all errors equally, making it highly robust to outliers.
3. **$R^2$ (Coefficient of Determination)**: Explains the proportion of variance in the target variable explained by features. An $R^2$ of $0.985$ means our model captures $98.5\%$ of the variance in crop yields.
