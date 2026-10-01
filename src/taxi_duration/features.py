"""Feature preparation used by BOTH training and the API.

Keeping this in one place guarantees the model sees the same columns,
in the same order, with the same types in production as it did in training.
"""
from collections.abc import Iterable, Mapping

import pandas as pd

from taxi_duration.config import FEATURES, MAX_DURATION, MIN_DURATION, TARGET

# Fixed dtypes so the MLflow signature matches at training and serving time.
# (The raw parquet stores location IDs as int32; JSON requests arrive as int64.)
FEATURE_DTYPES = {
    "PULocationID": "int64",
    "DOLocationID": "int64",
    "trip_distance": "float64",
}


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add the target column: trip duration in minutes."""
    df = df.copy()
    df[TARGET] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    return df


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Keep realistic trips only: MIN..MAX minutes and positive distance."""
    mask = (
        (df[TARGET] >= MIN_DURATION)
        & (df[TARGET] <= MAX_DURATION)
        & (df["trip_distance"] > 0)
    )
    return df.loc[mask, FEATURES + [TARGET]].dropna()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features in the right order with consistent dtypes."""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return df[FEATURES].astype(FEATURE_DTYPES)


def features_from_records(records: Iterable[Mapping]) -> pd.DataFrame:
    """Build a feature DataFrame from request payloads (list of dicts)."""
    return prepare_features(pd.DataFrame(list(records)))