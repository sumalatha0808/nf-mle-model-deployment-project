"""Download, load and split the NYC taxi data."""
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from taxi_duration.config import MONTH, SAMPLE_SIZE, SEED, TARGET, YEAR
from taxi_duration.features import add_duration, clean_trips, prepare_features

DATA_DIR = Path("data")
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_data(year: int = YEAR, month: int = MONTH, data_dir: Path = DATA_DIR) -> Path:
    """Download the monthly parquet file once and reuse it on later runs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    path = data_dir / filename
    if not path.exists():
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_data(path: Path, sample_size: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """Read raw trips, add the target, clean, and optionally sample."""
    df = clean_trips(add_duration(pd.read_parquet(path)))
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return df


def load_train_val(sample_size: int = SAMPLE_SIZE, test_size: float = 0.2, seed: int = SEED):
    """Return X_train, X_val, y_train, y_val.

    Training and tuning both call this, so they are compared on the exact same split.
    """
    df = load_data(download_data(), sample_size=sample_size, seed=seed)
    X, y = prepare_features(df), df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)