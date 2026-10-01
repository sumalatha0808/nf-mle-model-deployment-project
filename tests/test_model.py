"""Model tests on small synthetic data (no download or MLflow server needed)."""
import pandas as pd

from taxi_duration.config import FEATURES
from taxi_duration.features import prepare_features
from taxi_duration.train import evaluate, train_model

SMALL_PARAMS = {"n_estimators": 10, "max_depth": 5, "random_state": 0}


def trip(distance):
    return prepare_features(
        pd.DataFrame([{"PULocationID": 100, "DOLocationID": 100, "trip_distance": distance}])
    )


def test_model_predicts_one_value_per_row(synthetic):
    X, y = synthetic
    model = train_model(X, y, SMALL_PARAMS)
    assert model.predict(X.head(7)).shape == (7,)
    assert list(model.feature_names_in_) == FEATURES


def test_model_beats_mean_baseline(synthetic):
    X, y = synthetic
    model = train_model(X[:400], y[:400], SMALL_PARAMS)
    metrics = evaluate(model, X[400:], y[400:], y[:400].mean())
    assert metrics["rmse"] < metrics["mean_baseline_rmse"]


def test_longer_trips_take_longer(synthetic):
    X, y = synthetic
    model = train_model(X, y, SMALL_PARAMS)
    assert model.predict(trip(12.0))[0] > model.predict(trip(1.0))[0]