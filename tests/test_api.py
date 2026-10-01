"""API tests with a fake model, so no MLflow server is needed."""
import pytest
from fastapi.testclient import TestClient

import app.main as main


class FakeModel:
    def predict(self, X):
        return [X["trip_distance"].iloc[0] * 4.0]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda uri: FakeModel())
    with TestClient(main.app) as c:
        yield c


def test_health_ok(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_duration(client):
    resp = client.post("/predict", json={"PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.5})
    assert resp.status_code == 200
    assert resp.json()["duration_minutes"] == 10.0


@pytest.mark.parametrize(
    "payload",
    [
        {"PULocationID": 999, "DOLocationID": 236, "trip_distance": 2.5},  # bad zone
        {"PULocationID": 161, "DOLocationID": 236, "trip_distance": -1},   # negative distance
        {"PULocationID": 161, "DOLocationID": 236},                        # missing field
    ],
)
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_503_when_model_missing(monkeypatch):
    def fail(uri):
        raise RuntimeError("MLflow unreachable")

    monkeypatch.setattr(main, "load_model", fail)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "model_not_loaded"
        assert c.post("/predict", json={"PULocationID": 1, "DOLocationID": 2, "trip_distance": 1.0}).status_code == 503