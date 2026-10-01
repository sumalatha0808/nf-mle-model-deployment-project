"""Tuning test with a tiny grid so it runs in seconds."""
from taxi_duration.tune import run_grid_search


def test_grid_search_picks_from_grid(synthetic):
    X, y = synthetic
    grid = {"max_depth": [2, 8], "min_samples_leaf": [1, 10]}
    search = run_grid_search(X, y, param_grid=grid, cv=2, n_estimators=10)
    assert search.best_params_["max_depth"] in grid["max_depth"]
    assert len(search.cv_results_["params"]) == 4
    # Deeper trees should fit this clearly non-trivial pattern better
    assert search.best_params_["max_depth"] == 8