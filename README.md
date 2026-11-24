# Heart Disease Risk Predictor

A full-stack machine learning application that estimates a patient's likelihood of
heart disease from **8 core clinical attributes** and explains the result.

Built on the **UCI Heart Disease** dataset (920 records from four international
cohorts: Cleveland, Hungary, Switzerland, VA Long Beach).

## What it does

1. The user enters **8 values** — age, sex, chest-pain type, resting blood pressure,
   cholesterol, maximum heart rate, exercise-induced angina, and ST depression.
2. The browser sends the record to `POST /api/predict`.
3. The trained model returns a **probability of heart disease**, mapped to a
   **Low / Moderate / High** risk band.
4. The response also includes:
   - **Observed risk factors** — the patient's own values against standard clinical thresholds.
   - **The model's most influential features** — global feature importance.
   - **Recommended next steps**, tailored to the risk band.

> **Not a medical device.** Educational screening tool; it does not replace clinical judgement.

---

## Why only 8 features?

The original dataset has 14 usable predictors. Requiring all of them made the form
impractical — several need specialist procedures (fluoroscopy vessel count, thallium
stress scan) that a person cannot self-report.

Feature importance analysis showed the predictive signal is concentrated in a small
subset, so the model was retrained on **8 fields** and the rest were dropped:

| Feature | Global importance |
|---|---|
| `cp` — chest pain type | 20.0% |
| `exang` — exercise-induced angina | 19.5% |
| `dataset` — recording cohort | 10.2% |
| `oldpeak` — ST depression | 8.9% |
| `thalch` — maximum heart rate | 7.7% |
| `sex` | 6.4% |
| `chol` — cholesterol | 6.4% |
| `age` | 6.4% |
| `thal` — thallium scan | 5.3% |
| `trestbps` — blood pressure | 2.5% |
| `slope`, `ca`, `restecg`, `fbs` | 1.0–2.1% each |

Measured cost of the reduction (5-fold CV ROC-AUC):

| Feature set | Fields | CV ROC-AUC |
|---|---|---|
| All features | 14 | 0.8984 |
| **Selected set** | **8** | **0.8820** |
| Clinical minimal (no ECG) | 6 | 0.8512 |
| Top-importance only | 6 | 0.8590 |

Eight fields costs ~1.6 ROC-AUC points and removes every specialist-only
measurement plus the administrative `dataset` field.

---

## Architecture

```
┌─────────────────────────────┐
│  Browser (HTML/CSS/JS)      │  frontend/
│  form → fetch → render      │
└──────────────┬──────────────┘
               │  JSON over HTTP
┌──────────────▼──────────────┐
│  FastAPI backend            │  app/
│  /api/health                │
│  /api/metadata              │
│  /api/predict               │
│  /docs (Swagger UI)         │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  sklearn Pipeline           │  model/heart_model.joblib
│  imputer → scaler/one-hot   │
│  → classifier               │
└─────────────────────────────┘
```

Preprocessing and classifier are saved as **one Pipeline**, so inference applies
exactly the same imputation, scaling and encoding as training.

---

## Project structure

```
heart_disease_app/
├── app/
│   ├── main.py           # FastAPI routes, static mounting, lifespan
│   ├── schemas.py        # Pydantic request/response models + validation
│   ├── model_loader.py   # loads the pipeline and metadata once at startup
│   ├── predictor.py      # inference, risk banding, explanation, recommendations
│   └── config.py         # paths and risk thresholds
├── frontend/
│   ├── index.html        # single-page UI (8 fields)
│   ├── style.css         # dark theme, responsive
│   └── app.js            # form handling, API calls, result rendering
├── model/
│   ├── heart_model.joblib
│   ├── metrics.json      # metrics for every candidate model
│   └── metadata.json     # feature list, categories, selected model
├── data/
│   └── heart_disease_uci.csv
├── tests/
│   └── test_api.py       # 8 API tests
├── train_model.py        # reproducible training script
├── requirements.txt
├── Dockerfile
├── run.sh
└── README.md
```

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000 | Web UI |
| http://127.0.0.1:8000/docs | Interactive API documentation |
| http://127.0.0.1:8000/api/health | Health check |

### Docker

```bash
docker build -t heart-risk .
docker run -p 8000:8000 heart-risk
```

---

## Retraining

```bash
python train_model.py
```

Loads `data/heart_disease_uci.csv`, binarises the target (`num > 0 → 1`), builds the
preprocessing `ColumnTransformer`, runs `GridSearchCV` over five classifiers with
5-fold stratified CV (scoring = ROC-AUC), selects the best by test ROC-AUC, and
writes the model, `metrics.json` and `metadata.json`.

---

## Model results

70/30 stratified split — 644 training, 276 test records. 8 features.

| Model | Accuracy | F1 | ROC-AUC | Brier | CV ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** (selected) | 0.8043 | 0.8269 | **0.8988** | 0.1302 | 0.8757 |
| Support Vector Machine | 0.8116 | 0.8323 | 0.8905 | 0.1299 | 0.8759 |
| XGBoost | 0.7971 | 0.8170 | 0.8868 | 0.1378 | 0.8648 |
| Gradient Boosting | 0.7971 | 0.8182 | 0.8853 | 0.1375 | 0.8644 |
| Logistic Regression | **0.8225** | **0.8414** | 0.8847 | 0.1365 | **0.8790** |

> Hyperparameter grids are deliberately small so a full retrain finishes in about
> a minute. Widening them recovers a little more AUC.

---

## API reference

### `POST /api/predict`

```json
{
  "age": 63,
  "sex": "Male",
  "cp": "asymptomatic",
  "trestbps": 160,
  "chol": 290,
  "thalch": 105,
  "exang": true,
  "oldpeak": 2.8
}
```

Response:

```json
{
  "probability": 0.9332,
  "risk_level": "High",
  "label": "Heart disease likely",
  "confidence": 0.9332,
  "risk_factors": [
    {
      "label": "Resting blood pressure",
      "value": "160 mm Hg",
      "detail": "At or above the 140 mm Hg hypertension threshold"
    }
  ],
  "top_drivers": ["Cp", "Exang", "Oldpeak", "Thalch", "Sex"],
  "recommendations": ["Arrange a cardiology consultation promptly.", "..."],
  "model_name": "Random Forest"
}
```

### Field reference

| Field | Type | Range / values |
|---|---|---|
| `age` | int | 18–100 |
| `sex` | string | `Male`, `Female` |
| `cp` | string | `typical angina`, `atypical angina`, `non-anginal`, `asymptomatic` |
| `trestbps` | float | 80–220 (mm Hg) |
| `chol` | float | 100–600 (mg/dl) |
| `thalch` | float | 60–220 (bpm) |
| `exang` | bool | exercise-induced angina |
| `oldpeak` | float | −3.0 to 7.0 |

Unknown fields are ignored; out-of-range values are rejected with HTTP 422.

---

## Tests

```bash
pytest -q
```

```
8 passed
```

Covered: health endpoint, metadata (8 features + categories), response schema,
low-risk scoring below high-risk, risk-factor detection, rejection of out-of-range
values and invalid categories, and tolerance of removed fields.

---

## Design notes

**Why a Pipeline instead of separate preprocessor + model files?**
The original research scripts saved them separately. Serving from two artefacts
invites drift — one forgotten `transform()` and the model silently receives
unscaled data. A single Pipeline makes that impossible.

**Why rule-based risk factors instead of per-patient SHAP?**
The explanation layer reports two honest things: the patient's values against
standard clinical thresholds, and the model's *global* feature importance. It does
not invent per-patient attributions the model cannot justify.

**Why risk bands rather than a bare probability?**
A probability alone is hard to act on. Low/Moderate/High with matching
recommendations makes the output usable while keeping the number visible.

---

## Limitations

- **Dataset age and size.** 920 records, four cohorts; no longitudinal follow-up,
  medication data or imaging biomarkers.
- **Cross-sectional.** Predicts presence of disease from one snapshot, not progression.
- **Reduced feature set.** Dropping six predictors costs ~1.6 ROC-AUC points.
- **Not clinically validated.** No prospective validation performed.

## Future work

- Re-validate on MIMIC-IV or UK Biobank with temporal holdout.
- Add per-patient SHAP explanations and calibration curves.
- Multi-class severity prediction (original `num` values 1–4).
- Authentication, audit logging, and a database instead of stateless requests.

---

## Author

**Komal Bansal**

## License

MIT — free to use for learning and academic work.
