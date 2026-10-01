"""FastAPI app that serves trip-duration predictions.

Run from the project root:
    uv run uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import Prediction, Trip
from taxi_duration.config import MODEL_URI
from taxi_duration.features import features_from_records
from taxi_duration.model import load_model

state: dict = {"model": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup. If MLflow is unreachable, the API still
    # starts so /health can report the problem instead of crashing.
    try:
        state["model"] = load_model(MODEL_URI)
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["model"] = None
        state["error"] = str(exc)
        print(f"WARNING: could not load model {MODEL_URI}: {exc}")
    yield


app = FastAPI(title="NYC Taxi Trip Duration API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if state["model"] is not None else "model_not_loaded",
        "model_uri": MODEL_URI,
        "error": state["error"],
    }


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    X = features_from_records([trip.model_dump()])
    duration = float(model.predict(X)[0])
    return Prediction(duration_minutes=round(duration, 2), model_uri=MODEL_URI)