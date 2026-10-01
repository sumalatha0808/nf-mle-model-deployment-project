"""Settings shared by training, tuning and the API."""
import os
 
# ---- Data ----
YEAR = 2025
MONTH = 1
# The full month has ~3.5M rows; a sample keeps training fast. Override with SAMPLE_SIZE=...
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200000"))
SEED = 42
 
# ---- Features ----
# Features the model uses, in the order it expects them.
FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"
 
# Only trips in this range (minutes) are used for training.
MIN_DURATION = 1
MAX_DURATION = 60
 
# ---- MLflow ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "nyc-taxi-duration"
REGISTERED_MODEL_NAME = "random-forest-regressor"
 
# The API loads this model. Override with e.g. "models:/random-forest-regressor/3"
# or "runs:/<run_id>/model" to pin a specific version.
MODEL_URI = os.getenv("MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}/latest")