"""HAVM Framework - Configuration Constants.

Centralised configuration for paths, feature definitions, thresholds,
and model hyper-parameters used across every module in the framework.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
MODEL_DIR: str = os.path.join(BASE_DIR, "models")

# Dataset file paths
TRAIN_DATA_PATH: str = os.path.join(DATA_DIR, "train_data.csv")
INCOMING_DATA_PATH: str = os.path.join(DATA_DIR, "incoming_data.csv")
MODEL_PATH: str = os.path.join(MODEL_DIR, "rf_model.joblib")

# ---------------------------------------------------------------------------
# Feature schema (UCI Heart Disease dataset)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN: str = "target"
SENSITIVE_ATTRIBUTE: str = "sex"  # For fairness monitoring

# ---------------------------------------------------------------------------
# Monitoring thresholds
# ---------------------------------------------------------------------------
DRIFT_P_VALUE_THRESHOLD: float = 0.05        # KS-test significance level
OOD_CONTAMINATION: float = 0.1               # Isolation Forest contamination
UNCERTAINTY_CONFIDENCE_THRESHOLD: float = 0.60  # Flag predictions below 60 %
FAIRNESS_DISPARITY_THRESHOLD: float = 0.10   # 10 % disparity threshold

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_ESTIMATORS: int = 100
TEST_SIZE: float = 0.2
