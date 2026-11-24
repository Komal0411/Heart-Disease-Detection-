"""Loads the trained pipeline and its metadata once, at startup."""
import json

from joblib import load

from .config import METADATA_PATH, METRICS_PATH, MODEL_PATH

_model = None
_metadata = None
_metrics = None


def load_artifacts():
    """Load model + metadata into module-level singletons.

    Called once from the FastAPI startup hook so that every request reuses the
    already-deserialised objects instead of paying the load cost per call.
    """
    global _model, _metadata, _metrics

    if _model is None:
        _model = load(MODEL_PATH)

    if _metadata is None:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)

    if _metrics is None:
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)

    return _model, _metadata, _metrics


def get_model():
    if _model is None:
        load_artifacts()
    return _model


def get_metadata():
    if _metadata is None:
        load_artifacts()
    return _metadata


def get_metrics():
    if _metrics is None:
        load_artifacts()
    return _metrics
