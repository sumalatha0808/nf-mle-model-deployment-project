"""Stretch task: tune the RandomForest with GridSearchCV and compare it to the baseline.

Run from the project root (after training the baseline):
    uv run python -m taxi_duration.tune

What it does:
  1. Loads the SAME train/validation split as the baseline.
  2. Retrains the baseline on it (so both models are compared fairly).
  3. Runs GridSearchCV (3-fold CV) on a subsample of the training set to keep it fast.
  4. Retrains the best settings on the full training set and scores it on validation.
  5. Logs everything to MLflow and registers the tuned model ONLY if it beats the baseline.
"""
import time

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from taxi_duration.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, SEED
from taxi_duration.data import load_train_val
from taxi_duration.train import (
    ARTIFACTS_DIR,
    PARAMS,
    evaluate,
    log_and_register,
    model_size_mb,
    save_json,
    train_model,
)

# 3 x 3 x 2 = 18 combinations x 3 folds = 54 fits
PARAM_GRID = {
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [1, 5, 20],
    "max_features": [1.0, 0.67],  # all 3 features, or 2 of 3 per split
}
TUNE_SAMPLE_SIZE = 50_000
CV_FOLDS = 3
# Only switch to the tuned model if it improves RMSE by at least this much
MIN_IMPROVEMENT_PCT = 1.0


def run_grid_search(X, y, param_grid=PARAM_GRID, cv=CV_FOLDS, n_estimators=100) -> GridSearchCV:
    # Parallelise across CV fits (n_jobs=-1 on the search), not inside each forest,
    # to avoid starting too many processes at once.
    base = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=SEED)
    search = GridSearchCV(
        base,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    return search


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def main():
    X_train, X_val, y_train, y_val = load_train_val()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="grid-search") as run:
        mlflow.set_tag("model_type", "tuned")

        # 1. Baseline on the same split
        print("Training baseline for comparison ...")
        baseline, baseline_secs = timed(train_model, X_train, y_train, PARAMS)
        base_m = evaluate(baseline, X_val, y_val, y_train.mean())

        # 2. Grid search on a subsample
        n = min(TUNE_SAMPLE_SIZE, len(X_train))
        X_sub = X_train.sample(n=n, random_state=SEED)
        y_sub = y_train.loc[X_sub.index]
        print(f"Grid search on {n:,} rows ...")
        search, search_secs = timed(run_grid_search, X_sub, y_sub)

        cv_results = pd.DataFrame(search.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")
        cv_results["mean_cv_rmse"] = -cv_results.pop("mean_test_score")
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        cv_path = ARTIFACTS_DIR / "grid_search_results.csv"
        cv_results.to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path))

        # 3. Refit best settings on the full training set
        tuned_params = {**PARAMS, **search.best_params_}
        print(f"Best params: {search.best_params_}")
        tuned, tuned_secs = timed(train_model, X_train, y_train, tuned_params)
        tuned_m = evaluate(tuned, X_val, y_val, y_train.mean())

        improvement = (base_m["rmse"] - tuned_m["rmse"]) / base_m["rmse"] * 100
        worth_it = improvement >= MIN_IMPROVEMENT_PCT

        comparison = {
            "baseline": {
                "params": PARAMS,
                "rmse": base_m["rmse"],
                "mae": base_m["mae"],
                "train_seconds": baseline_secs,
                "model_size_mb": model_size_mb(baseline),
            },
            "tuned": {
                "params": tuned_params,
                "rmse": tuned_m["rmse"],
                "mae": tuned_m["mae"],
                "train_seconds": tuned_secs,
                "model_size_mb": model_size_mb(tuned),
                "best_cv_rmse": -search.best_score_,
                "search_seconds": search_secs,
            },
            "mean_baseline_rmse": base_m["mean_baseline_rmse"],
            "rmse_improvement_pct": improvement,
            "registered_tuned_model": worth_it,
        }

        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_params({"tune_rows": n, "cv_folds": CV_FOLDS, "grid_size": len(cv_results)})
        mlflow.log_metrics(
            {
                "baseline_rmse": base_m["rmse"],
                "rmse": tuned_m["rmse"],
                "mae": tuned_m["mae"],
                "rmse_improvement_pct": improvement,
                "search_seconds": search_secs,
                "tuned_train_seconds": tuned_secs,
                "baseline_train_seconds": baseline_secs,
                "tuned_model_size_mb": comparison["tuned"]["model_size_mb"],
                "baseline_model_size_mb": comparison["baseline"]["model_size_mb"],
            }
        )
        mlflow.log_dict(comparison, "comparison.json")
        log_and_register(tuned, X_val, register=worth_it)

    save_json("comparison.json", {"run_id": run.info.run_id, **comparison})

    print("\n=== Baseline vs. tuned (validation set) ===")
    print(f"{'':12}{'RMSE':>8}{'MAE':>8}{'train s':>10}{'size MB':>10}")
    for name in ("baseline", "tuned"):
        m = comparison[name]
        print(f"{name:12}{m['rmse']:8.2f}{m['mae']:8.2f}{m['train_seconds']:10.1f}{m['model_size_mb']:10.1f}")
    print(f"Grid search took {search_secs:.0f}s; RMSE improvement: {improvement:.2f}%")
    if worth_it:
        print("Tuned model registered as a new version. Restart the API to serve it.")
    else:
        print(f"Improvement < {MIN_IMPROVEMENT_PCT}%: kept the baseline as the served model.")


if __name__ == "__main__":
    main()