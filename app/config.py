"""Application configuration."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

APP_NAME = "Heart Disease Risk Predictor"
APP_VERSION = "1.0.0"

# Probability thresholds that map a model score to a human risk band.
RISK_THRESHOLDS = {"low": 0.33, "moderate": 0.66}
