"""Prediction logic: model inference, risk banding, and explanation.

The explanation layer is deliberately rule-based and transparent: it reports
the patient's own values against standard clinical thresholds, and separately
reports which features the model relies on most globally. This avoids inventing
per-patient attributions that the model cannot actually justify.
"""
import pandas as pd

from .config import RISK_THRESHOLDS
from .model_loader import get_metadata, get_model
from .schemas import PatientInput, PredictionResponse, RiskFactor

# Clinical reference thresholds used for the explanatory flags.
HIGH_BP = 140
HIGH_CHOL = 240
LOW_MAX_HR = 120
SIGNIFICANT_ST_DEPRESSION = 2.0


def _to_frame(patient: PatientInput, feature_order: list[str]) -> pd.DataFrame:
    """Build a one-row DataFrame in the exact feature order seen in training."""
    record = patient.model_dump()

    # exang was normalised to a string ("True"/"False") during training, so the
    # encoder expects the same representation here.
    record["exang"] = str(bool(record["exang"]))

    return pd.DataFrame([record])[feature_order]


def _risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if probability < RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    return "High"


def _clinical_flags(patient: PatientInput) -> list[RiskFactor]:
    """Transparent, rule-based observations about this patient's values."""
    flags: list[RiskFactor] = []

    if patient.trestbps >= HIGH_BP:
        flags.append(RiskFactor(label="Resting blood pressure",
                                value=f"{patient.trestbps:.0f} mm Hg",
                                detail=f"At or above the {HIGH_BP} mm Hg hypertension threshold"))
    if patient.chol >= HIGH_CHOL:
        flags.append(RiskFactor(label="Serum cholesterol",
                                value=f"{patient.chol:.0f} mg/dl",
                                detail=f"Above the {HIGH_CHOL} mg/dl high-cholesterol threshold"))
    if patient.thalch < LOW_MAX_HR:
        flags.append(RiskFactor(label="Maximum heart rate",
                                value=f"{patient.thalch:.0f} bpm",
                                detail="Low exercise capacity, often associated with coronary disease"))
    if patient.oldpeak >= SIGNIFICANT_ST_DEPRESSION:
        flags.append(RiskFactor(label="ST depression (oldpeak)",
                                value=f"{patient.oldpeak:.1f}",
                                detail="Indicates myocardial ischaemia during exercise"))
    if patient.exang:
        flags.append(RiskFactor(label="Exercise-induced angina",
                                value="Present",
                                detail="Chest pain triggered by exertion"))
    if patient.cp == "asymptomatic":
        flags.append(RiskFactor(label="Chest pain type",
                                value="Asymptomatic",
                                detail="Silent ischaemia is common in confirmed cases in this cohort"))
    return flags


def _global_importance(model) -> dict[str, float]:
    """Aggregate transformed-feature importance back onto original columns."""
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    names = pre.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        raw = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        raw = abs(clf.coef_[0])
    else:
        return {}

    agg: dict[str, float] = {}
    for name, value in zip(names, raw):
        body = name.split("__", 1)[1]
        key = body if name.startswith("num__") else body.split("_")[0]
        agg[key] = agg.get(key, 0.0) + float(value)

    total = sum(agg.values()) or 1.0
    return {k: v / total for k, v in agg.items()}


def _recommendations(level: str, flags: list[RiskFactor]) -> list[str]:
    if level == "Low":
        recs = [
            "Maintain your current activity level and a balanced diet.",
            "Recheck blood pressure, cholesterol and glucose annually.",
            "Keep a consistent sleep schedule and manage stress actively.",
        ]
    elif level == "Moderate":
        recs = [
            "Book a consultation with a physician to review these results.",
            "Target 150 minutes of moderate aerobic activity per week.",
            "Reduce saturated fat and salt; monitor blood pressure at home.",
            "Recheck lipids and fasting glucose within three months.",
        ]
    else:
        recs = [
            "Arrange a cardiology consultation promptly.",
            "Do not ignore chest pain, breathlessness, or unusual fatigue — seek urgent care if they occur.",
            "Discuss diagnostic follow-up such as an ECG or stress test with your doctor.",
            "Begin a medically supervised plan before starting any new exercise programme.",
        ]

    if any(f.label == "Serum cholesterol" for f in flags):
        recs.append("Discuss lipid management with your physician.")
    if any(f.label == "Resting blood pressure" for f in flags):
        recs.append("Track blood pressure twice daily and share the log with your doctor.")
    return recs


def predict(patient: PatientInput) -> PredictionResponse:
    model = get_model()
    metadata = get_metadata()

    frame = _to_frame(patient, metadata["features"])
    probability = float(model.predict_proba(frame)[0][1])

    level = _risk_level(probability)
    predicted_class = 1 if probability >= 0.5 else 0
    confidence = probability if predicted_class == 1 else 1 - probability

    flags = _clinical_flags(patient)
    importance = _global_importance(model)
    top_drivers = [k.replace("_", " ").title()
                   for k, _ in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:5]]

    return PredictionResponse(
        probability=round(probability, 4),
        risk_level=level,
        label="Heart disease likely" if predicted_class == 1 else "No heart disease indicated",
        confidence=round(confidence, 4),
        risk_factors=flags,
        top_drivers=top_drivers,
        recommendations=_recommendations(level, flags),
        model_name=metadata["selected_model"],
    )
