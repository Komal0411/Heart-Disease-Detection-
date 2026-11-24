"""API tests for the 8-feature model. Run with:  pytest -q"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

LOW_RISK = {
    "age": 45, "sex": "Female", "cp": "non-anginal",
    "trestbps": 120, "chol": 180, "thalch": 170,
    "exang": False, "oldpeak": 0.0,
}

HIGH_RISK = {
    "age": 63, "sex": "Male", "cp": "asymptomatic",
    "trestbps": 160, "chol": 290, "thalch": 105,
    "exang": True, "oldpeak": 2.8,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_metadata_exposes_categories(client):
    res = client.get("/api/metadata")
    assert res.status_code == 200
    body = res.json()
    assert "model" in body
    assert len(body["features"]) == 8
    assert set(body["features"]) == {
        "age", "sex", "cp", "trestbps", "chol", "thalch", "exang", "oldpeak"}
    assert set(body["categories"]["cp"]) == {
        "typical angina", "atypical angina", "non-anginal", "asymptomatic"}


def test_predict_returns_valid_shape(client):
    res = client.post("/api/predict", json=LOW_RISK)
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Moderate", "High"}
    assert isinstance(body["recommendations"], list) and body["recommendations"]


def test_low_risk_scores_below_high_risk(client):
    low = client.post("/api/predict", json=LOW_RISK).json()["probability"]
    high = client.post("/api/predict", json=HIGH_RISK).json()["probability"]
    assert low < high, f"expected low ({low}) < high ({high})"


def test_high_risk_patient_is_flagged(client):
    body = client.post("/api/predict", json=HIGH_RISK).json()
    assert body["risk_level"] in {"Moderate", "High"}
    labels = {f["label"] for f in body["risk_factors"]}
    assert "Exercise-induced angina" in labels


def test_invalid_age_rejected(client):
    bad = dict(LOW_RISK, age=5)
    assert client.post("/api/predict", json=bad).status_code == 422


def test_invalid_category_rejected(client):
    bad = dict(LOW_RISK, cp="made up")
    assert client.post("/api/predict", json=bad).status_code == 422


def test_removed_field_is_ignored(client):
    """A field that is no longer part of the model must not break the request."""
    extra = dict(LOW_RISK, fbs=True, thal="normal", ca=2, dataset="Cleveland")
    assert client.post("/api/predict", json=extra).status_code == 200
