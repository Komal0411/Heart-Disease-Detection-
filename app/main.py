"""FastAPI application: REST API + static frontend."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import APP_NAME, APP_VERSION, FRONTEND_DIR
from .model_loader import get_metadata, get_metrics, load_artifacts
from .predictor import predict
from .schemas import PatientInput, PredictionResponse

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once when the server boots, not on every request."""
    load_artifacts()
    yield


app = FastAPI(
    lifespan=lifespan,
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Predicts the likelihood of heart disease from 14 clinical attributes "
        "using a model trained on the UCI Heart Disease dataset (920 records)."
    ),
)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/metadata", tags=["system"])
def metadata():
    """Model information plus the category options the UI needs to build its form."""
    meta = get_metadata()
    metrics = get_metrics()
    return {
        "model": meta["selected_model"],
        "dataset": meta["dataset"],
        "n_samples": meta["n_samples"],
        "n_features": len(meta["features"]),
        "features": meta["features"],
        "categories": meta["categories"],
        "metrics": metrics,
        "selected_metrics": meta["metrics"],
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_endpoint(patient: PatientInput):
    """Return a probability, a risk band, explanatory flags and recommendations."""
    try:
        return predict(patient)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


# ---- static frontend -------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
