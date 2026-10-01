"""Train the BASELINE RandomForestRegressor and track it with MLflow.

Run from the project root:
    uv run python -m taxi_duration.train
"""
import json
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from taxi_duration.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MONTH,
    REGISTERED_MODEL_NAME,
    SEED,
    YEAR,
)
from taxi_duration.data import load_train_val

ARTIFACTS_DIR = Path("artifacts")

# MLflow 3.x saves sklearn models in the safe "skops" format, which refuses to
# load/save types it doesn't know. The forest's tree storage must be allowed
# explicitly. This is safe here because we only load models we trained ourselves.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Baseline hyperparameters (chosen by hand, not tuned)
PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": SEED,
}


def train_model(X: pd.DataFrame, y: pd.Series, params: dict = PARAMS) -> RandomForestRegressor:
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    return model


def evaluate(model, X_val: pd.DataFrame, y_val: pd.Series, y_train_mean: float) -> dict:
    """RMSE and MAE for the model, plus a predict-the-mean baseline."""
    preds = model.predict(X_val)
    return {
        "rmse": root_mean_squared_error(y_val, preds),
        "mae": mean_absolute_error(y_val, preds),
        "mean_baseline_rmse": root_mean_squared_error(y_val, [y_train_mean] * len(y_val)),
    }


def model_size_mb(model) -> float:
    return len(pickle.dumps(model)) / 1e6


def log_and_register(model, X_val: pd.DataFrame, register: bool = True):
    """Log the model to MLflow and (optionally) register a new version."""
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(X_val, model.predict(X_val)),
        input_example=X_val.head(3),
        registered_model_name=REGISTERED_MODEL_NAME if register else None,
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )


def save_json(name: str, data: dict) -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / name
    path.write_text(json.dumps(data, indent=2))
    return path


def main():
    X_train, X_val, y_train, y_val = load_train_val()
    print(f"Training on {len(X_train):,} rows, validating on {len(X_val):,} rows")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline") as run:
        mlflow.set_tag("model_type", "baseline")
        mlflow.log_params({**PARAMS, "year": YEAR, "month": MONTH, "train_rows": len(X_train)})

        start = time.perf_counter()
        model = train_model(X_train, y_train)
        train_seconds = time.perf_counter() - start

        metrics = evaluate(model, X_val, y_val, y_train.mean())
        metrics.update(train_seconds=train_seconds, model_size_mb=model_size_mb(model))
        mlflow.log_metrics(metrics)
        log_and_register(model, X_val)

    print(
        f"RMSE: {metrics['rmse']:.2f} min | MAE: {metrics['mae']:.2f} min | "
        f"predict-the-mean RMSE: {metrics['mean_baseline_rmse']:.2f} min | "
        f"train time: {train_seconds:.1f}s"
    )
    save_json("metrics.json", {"run_id": run.info.run_id, **metrics, "params": PARAMS})
    print(f"Done. Run {run.info.run_id} at {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()