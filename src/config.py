from pathlib import Path

# Project Paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "logistic_regression.csv"

MODEL_DIR = BASE_DIR / "models"

ARTIFACT_DIR = BASE_DIR / "artifacts"

LOG_DIR = BASE_DIR / "logs"


# Dataset

TARGET_COLUMN = "target"


# Train-Test Split

TEST_SIZE = 0.30

RANDOM_STATE = 42


# Logistic Regression

MAX_ITER = 1000

THRESHOLD = 0.60


# Saved Files

MODEL_NAME = "loan_default_model.pkl"

SCALER_NAME = "scaler.pkl"

FEATURE_NAME = "feature_columns.pkl"