"""Pydantic request/response models with clinical-range validation."""
from typing import Literal

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """A single patient record, limited to the eight core predictors.

    These are the fields a person can realistically supply from a basic
    check-up. Specialist measurements (fluoroscopy vessel count, thallium
    scan) and administrative fields were dropped from the model: they add
    little predictive value here and make the form impractical to fill.
    """

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal["Male", "Female"] = Field(..., description="Biological sex")

    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"] = Field(
        ..., description="Chest pain type")
    trestbps: float = Field(..., ge=80, le=220, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    thalch: float = Field(..., ge=60, le=220, description="Maximum heart rate achieved (bpm)")
    exang: bool = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=-3.0, le=7.0, description="ST depression induced by exercise")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 57, "sex": "Male", "cp": "asymptomatic",
                "trestbps": 150, "chol": 255, "thalch": 120,
                "exang": True, "oldpeak": 2.4,
            }
        }
    }


class RiskFactor(BaseModel):
    label: str
    value: str
    detail: str


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Model probability of heart disease (0-1)")
    risk_level: Literal["Low", "Moderate", "High"]
    label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., description="Confidence in the predicted class (0-1)")
    risk_factors: list[RiskFactor]
    top_drivers: list[str]
    recommendations: list[str]
    model_name: str
