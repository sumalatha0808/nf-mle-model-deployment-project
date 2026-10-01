# NYC Taxi Trip Duration

Predicts NYC yellow taxi trip duration (minutes) from pickup zone, drop-off zone and distance, using January 2025 data. Built with scikit-learn, MLflow, FastAPI and Docker.

**Final model:** baseline random forest, **validation RMSE 4.75 min**.

## How to reproduce

Requires [uv](https://docs.astral.sh/uv/) and Docker Desktop. Run from the project root.

```powershell
uv sync                                              # install
uv run pytest                                        # 14 tests
docker compose -f docker/compose.yaml up -d          # Postgres + MLflow + API (wait until mlflow is healthy)
uv run python -m taxi_duration.train                 # train + register baseline
uv run python -m taxi_duration.tune                  # grid search + comparison
docker compose -f docker/compose.yaml restart api    # load the latest model
uv run python app/sample_request.py                  # test the API
docker compose -f docker/compose.yaml down           # stop
```

MLflow UI: http://127.0.0.1:5001 · API docs: http://127.0.0.1:8000/docs. The whole pipeline can also be run from `notebooks/02_run_pipeline.ipynb`.

## Layout

```
notebooks/          full pipeline run (01)
src/taxi_duration/ config, data loading, features, training, tuning, model loading
app/               FastAPI app (main.py), schemas, sample client
tests/             data, model, tuning and API tests
docker/            compose.yaml (postgres, mlflow, api), init-db.sql
artifacts/         metrics.json, comparison.json (git-ignored)
```

Feature preparation lives in `src/taxi_duration/features.py` and is shared by training and the API, so both prepare inputs identically.

## Data

- **Source:** NYC TLC Yellow Taxi, January 2025 (~3.5M trips)
- **Target:** drop-off time − pickup time, in minutes
- **Features:** `PULocationID`, `DOLocationID`, `trip_distance`
- **Cleaning:** trips of 1–60 minutes with distance > 0
- **Sample/split:** 200,000 trips, 80/20 train/validation (seed 42)

## Model comparison

| Model | RMSE (min) | MAE (min) | Train (s) | Size (MB) |
|---|---|---|---|---|
| Predict the mean | 9.78 | – | – | – |
| **Baseline RF** (depth 15, min leaf 5) | **4.748** | 3.255 | 13.1 | 88.3 |
| Tuned RF (depth 20, min leaf 20, max_features 0.67) | 4.746 | 3.252 | 21.0 | 51.9 |

Tuning: `GridSearchCV`, 18 settings × 3 folds on 50k rows, 156 s.

**Was it worth it? No.** Tuning improved RMSE by only 0.03% (about 0.1 s per trip), while training took 60% longer. The tuned model is 41% smaller, which could matter on a small server, but not here. Because the gain was below 1%, the tuned model wasn't registered, and the API serves the baseline. The model is limited by its features, not its hyperparameters.

## API

`POST /predict` with `PULocationID` and `DOLocationID` (1–265) and `trip_distance` (> 0):

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.5}'
```

```json
{"duration_minutes": 16.17, "model_uri": "models:/random-forest-regressor/latest"}
```

Invalid input returns 422. `GET /health` shows whether the model is loaded (`/predict` returns 503 if not).

## Answers to the assignment questions

**1. What is the RMSE of your final model?**
**4.75 minutes** on the validation set (MAE 3.26), about half the error of predicting the average trip (9.78).

**2. What would you do differently with more time?**
- Add pickup hour, day of week and airport features, which are likely the biggest gain.
- Train on the full month and validate on February 2025 instead of a random split.
- Try LightGBM and Optuna tuning.
- Build a dedicated Docker image, add CI, and monitor predictions for drift.

## Notes

- Tested on **Windows 10** with PowerShell and Docker Desktop.
- MLflow runs on host port **5001** (port 5000 inside the container), to avoid clashing with other local services that use port 5000.
- The database login is passed via `PGUSER`/`PGPASSWORD`, and the URL uses `postgresql+psycopg2://`, because SQLAlchemy 2.1 defaults to a different driver.
