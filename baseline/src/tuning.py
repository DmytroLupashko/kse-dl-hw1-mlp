"""Early-stopped LightGBM trials for tuning lgbm_building (exp03 to exp05): one MLflow run per fold."""
from time import perf_counter

import lightgbm as lgb
import mlflow
import numpy as np
from mlflow.entities import Metric

from .callbacks import progress


def run_trial(name, params, folds, *, learning_rate, max_trees, patience, curve_every, logged, tags):
    """Trains `params` on every fold, early-stopped, and logs one MLflow run per fold to the active experiment:
    parameters plus `logged`, the fold's best tree count, validation RMSLE there (all and clean rows), training time
    and every `curve_every`-th tree of both validation curves. Returns the mean best tree count and mean RMSLE."""

    # Train every fold, recording both validation curves
    curves, seconds = {}, {}
    for fold, d in folds.items():
        model = lgb.LGBMRegressor(
            n_estimators=max_trees, learning_rate=learning_rate, metric="rmse", random_state=0, verbose=-1, **params
        )
        history = {}

        start = perf_counter()
        model.fit(
            d["X_train"],
            d["y_train"],
            eval_X=(d["X_val"], d["X_val"][d["clean"]]),
            eval_y=(d["y_val"], d["y_val"][d["clean"]]),
            eval_names=["all", "clean"],
            callbacks=[lgb.record_evaluation(history), progress(f"{name} | {fold}", max_trees, patience)],
        )
        seconds[fold] = perf_counter() - start

        # LightGBM's rmse on log1p values: RMSLE except that negative predictions aren't clipped at 0
        curves[fold] = {rows: np.array(history[rows]["rmse"]) for rows in ["all", "clean"]}

    # Each fold's best tree (an index), and the trial's tree count: their mean
    best = {fold: int(c["all"].argmin()) for fold, c in curves.items()}
    trees = round(np.mean([b + 1 for b in best.values()]))

    # One MLflow run per fold
    client = mlflow.MlflowClient()
    for fold, c in curves.items():
        b = best[fold]
        run_params = {"learning_rate": learning_rate, **params, "max_trees": max_trees, "patience": patience, **logged}
        run_metrics = {
            "best_trees": b + 1,
            "trial_trees": trees,
            "val_rmsle": c["all"][b],
            "val_rmsle_clean": c["clean"][b],
            "train_seconds": seconds[fold],
        }

        # Every `curve_every`-th point of both curves, logged as a metric with the tree count as its step
        steps = range(curve_every - 1, len(c["all"]), curve_every)
        curve_points = [
            Metric(f"val_rmsle_curve_{rows}", float(c[rows][i]), 0, i + 1)
            for rows in ["all", "clean"]
            for i in steps
        ]

        with mlflow.start_run(run_name=f"{name} | {fold}", tags={**tags, "fold": fold, "trial": name}) as run:
            mlflow.log_params(run_params)
            mlflow.log_metrics(run_metrics)
            client.log_batch(run.info.run_id, metrics=curve_points)

    score = float(np.mean([c["all"][best[fold]] for fold, c in curves.items()]))
    print(f"{name}: {trees} trees (fold mean), fold-average RMSLE {score:.4f}\n")
    return trees, score
