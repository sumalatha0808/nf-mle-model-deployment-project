"""Load the trained model from MLflow."""
from functools import lru_cache

import mlflow
import mlflow.pyfunc

from taxi_duration.config import MLFLOW_TRACKING_URI, MODEL_URI


@lru_cache(maxsize=4)
def load_model(model_uri: str = MODEL_URI, tracking_uri: str = MLFLOW_TRACKING_URI):
    """Load a model once and cache it.

    Uses the pyfunc flavor so MLflow checks inputs against the logged signature.
    """
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)