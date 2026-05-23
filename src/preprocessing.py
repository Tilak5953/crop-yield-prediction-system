from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils import get_logger

logger = get_logger("preprocessing")

def get_preprocessing_pipeline() -> ColumnTransformer:
    """
    Creates and returns a reusable scikit-learn ColumnTransformer preprocessing pipeline.
    
    Why:
    1. Numerical Scaling: Neural nets, tree-based models (to a lesser degree), and linear models 
       benefit from centered data. StandardScaler scales numerical columns to mean=0, std=1.
    2. OneHotEncoder: Model algorithms require numeric inputs. Encoders transform high-cardinality 
       categorical fields (State_Name, Crop, Season) into binary indicator matrices.
    3. SimpleImputer: Handles unexpected missing values during inference or test sets without crashing,
       replacing numerical nulls with column medians and categorical nulls with 'Unknown'.
    """
    logger.info("Building scikit-learn preprocessing ColumnTransformer...")
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop" # Drop any other columns (e.g. State_Name / District_Name if not needed)
    )
    
    return preprocessor
