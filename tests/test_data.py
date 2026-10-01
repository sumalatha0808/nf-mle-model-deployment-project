"""Data and feature tests: cleaning rules and train/serve feature parity."""
import pandas as pd
import pytest

from taxi_duration.config import FEATURES, TARGET
from taxi_duration.features import (
    add_duration,
    clean_trips,
    features_from_records,
    prepare_features,
)


def raw_trips():
    pickup = pd.Timestamp("2025-01-01 10:00")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [pickup] * 4,
            "tpep_dropoff_datetime": [
                pickup + pd.Timedelta(minutes=10),   # kept
                pickup + pd.Timedelta(seconds=30),   # too short
                pickup + pd.Timedelta(minutes=90),   # too long
                pickup + pd.Timedelta(minutes=5),    # zero distance
            ],
            "PULocationID": pd.array([1, 2, 3, 4], dtype="int32"),
            "DOLocationID": pd.array([5, 6, 7, 8], dtype="int32"),
            "trip_distance": [1.5, 0.2, 20.0, 0.0],
        }
    )


def test_add_duration_in_minutes():
    df = add_duration(raw_trips())
    assert df[TARGET].iloc[0] == pytest.approx(10.0)


def test_clean_trips_filters_unrealistic_rows():
    df = clean_trips(add_duration(raw_trips()))
    assert len(df) == 1
    assert list(df.columns) == FEATURES + [TARGET]


def test_training_and_api_features_match():
    train_X = prepare_features(clean_trips(add_duration(raw_trips())))
    api_X = features_from_records([{"PULocationID": 1, "DOLocationID": 5, "trip_distance": 1.5}])
    assert list(train_X.columns) == list(api_X.columns) == FEATURES
    assert train_X.dtypes.equals(api_X.dtypes)


def test_missing_feature_raises():
    with pytest.raises(ValueError):
        features_from_records([{"PULocationID": 1}])