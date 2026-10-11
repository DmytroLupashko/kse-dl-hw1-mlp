"""Early-stopped LightGBM trials for tuning lgbm_building (exp03 to exp05): fold data, a progress bar, one MLflow run
per fold."""
from time import perf_counter

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Metric
from tqdm import tqdm

from .data import read_processed, split
from .features import build_features


def make_folds(folds, n_train, n_val):
    """Per fold: features plus `building_id` as a categorical, `log1p` targets, the clean-rows mask and the meter."""
    df = read_processed()
    building = pd.CategoricalDtype(range(int(df.building_id.max()) + 1))  # all ids, so every fold shares categories

    def with_building(X, rows):
        return X.assign(building_id=rows.building_id.astype(int).astype(building).to_numpy())

    data = {}
    for fold, months in tqdm(folds.items(), desc="fold data", unit="fold"):
        train_df, val_df = split(df, months, n_train, n_val)
        year_median = train_df.year_built.median()
        data[fold] = {
            "X_train": with_building(build_features(train_df, year_median), train_df),
            "y_train": np.log1p(train_df.meter_reading.to_numpy(dtype="float32")),
            "X_val": with_building(build_features(val_df, year_median), val_df),
            "y_val": np.log1p(val_df.meter_reading.to_numpy(dtype="float32")),
            "clean": ~val_df.anomaly.to_numpy(),
            "meter": val_df.meter.to_numpy(),
        }
    return data


def progress(desc, total, patience=None):
    """LightGBM callback: a progress bar with validation RMSLE, and early stopping on the "all" validation set."""
    bar = tqdm(total=total, desc=desc, unit="tree", dynamic_ncols=True)
    best = {"score": np.inf, "tree": 0}

    def callback(env):
        trees = env.iteration + 1
        score = {name: value for name, _, value, _ in env.evaluation_result_list}["all"]
        if score < best["score"]:
            best.update(score=score, tree=trees)
        bar.update(1)
        if trees % 10 == 0:
            bar.set_postfix_str(f"val {score:.4f}, best {best['score']:.4f} at {best['tree']}")
        stop = patience is not None and trees - best["tree"] >= patience
        if stop or trees == total:
            bar.set_postfix_str(f"best {best['score']:.4f} at {best['tree']}" + (" (early stop)" if stop else ""))
            bar.total = trees
            bar.close()
        if stop:
            raise lgb.callback.EarlyStopException(best["tree"] - 1, env.evaluation_result_list)

    callback.order = 30  # after record_evaluation, so the curve includes the last tree
    return callback


def run_trial(name, params, folds, *, learning_rate, max_trees, patience, curve_every, logged, tags):
    """Trains `params` on every fold, early-stopped, and logs one MLflow run per fold to the active experiment:
    parameters plus `logged`, the fold's best tree count, validation RMSLE there (all and clean rows), training time
    and every `curve_every`-th tree of both validation curves. Returns the mean best tree count and mean RMSLE."""
    curves, seconds = {}, {}
    for fold, d in folds.items():
        history = {}
        start = perf_counter()
        lgb.LGBMRegressor(
            n_estimators=max_trees, learning_rate=learning_rate, metric="rmse", random_state=0, verbose=-1, **params
        ).fit(
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
    best = {fold: int(c["all"].argmin()) for fold, c in curves.items()}  # index of each fold's best tree
    trees = round(np.mean([b + 1 for b in best.values()]))
    client = mlflow.MlflowClient()
    for fold, c in curves.items():
        with mlflow.start_run(run_name=f"{name} | {fold}", tags={**tags, "fold": fold, "trial": name}) as run:
            mlflow.log_params(
                {"learning_rate": learning_rate, **params, "max_trees": max_trees, "patience": patience, **logged}
            )
            mlflow.log_metrics(
                {
                    "best_trees": best[fold] + 1,
                    "trial_trees": trees,
                    "val_rmsle": c["all"][best[fold]],
                    "val_rmsle_clean": c["clean"][best[fold]],
                    "train_seconds": seconds[fold],
                }
            )
            steps = range(curve_every - 1, len(c["all"]), curve_every)
            client.log_batch(
                run.info.run_id,
                metrics=[
                    Metric(f"val_rmsle_curve_{rows}", float(c[rows][i]), 0, i + 1)
                    for rows in ["all", "clean"]
                    for i in steps
                ],
            )
    score = float(np.mean([c["all"][best[fold]] for fold, c in curves.items()]))
    print(f"{name}: {trees} trees (fold mean), fold-average RMSLE {score:.4f}\n")
    return trees, score
