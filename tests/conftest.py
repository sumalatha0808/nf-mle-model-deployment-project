import numpy as np
import pandas as pd
import pytest

from taxi_duration.features import prepare_features


@pytest.fixture
def synthetic():
    """Small fake dataset where duration is roughly 4 min per mile."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "PULocationID": rng.integers(1, 266, n),
            "DOLocationID": rng.integers(1, 266, n),
            "trip_distance": rng.uniform(0.1, 15, n),
        }
    )
    y = pd.Series(4 * df["trip_distance"] + rng.normal(0, 1, n), name="duration")
    return prepare_features(df), y