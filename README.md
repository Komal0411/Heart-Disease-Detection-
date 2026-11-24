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

## License

MIT — free to use for learning and academic work.
