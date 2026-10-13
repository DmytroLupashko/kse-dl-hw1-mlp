"""Scores of finished MLP runs read back from MLflow, and the fold table that shows them."""
import numpy as np
import pandas as pd


def scores_at_best(experiment, run_id):
    """A run's scores at its best validation epoch and at its last one, plus its per-epoch curves; a diverged
    run scores inf."""
    val = np.array(experiment.history(run_id, "val_rmsle"))
    train = np.array(experiment.history(run_id, "train_rmsle"))
    if not np.isfinite(val).all():
        return {"val": np.inf, "val_curve": val, "train_curve": train}

    clean = experiment.history(run_id, "val_rmsle_clean")
    seconds = experiment.history(run_id, "epoch_seconds")
    epoch_seconds = np.mean(seconds) if seconds else np.nan

    b = int(val.argmin())  # index of the best epoch
    return {
        "best_epoch": b + 1,
        "val": val[b],
        "val_clean": clean[b] if clean else np.nan,
        "train": train[b],
        "val_last": val[-1],
        "val_clean_last": clean[-1] if clean else np.nan,
        "train_last": train[-1],
        "epoch_seconds": epoch_seconds,
        "minutes_to_best": epoch_seconds * (b + 1) / 60,
        "val_curve": val,
        "train_curve": train,
    }


def finished_keys(results, columns):
    """The `columns` values of each run in `results`, as tuples, for `fit_folds(done=...)`."""
    return set(results[columns].itertuples(index=False, name=None))


def session_scores(experiment, keys, folds, references=(), at_best=None, own=None):
    """One row per finished run of `experiment`'s session and of the `references` frames (filtered already) in
    `folds`: run_id, the `keys` columns cast with their types ({tag: type}, so they match `fit_folds` keys), fold,
    and `at_best(run_id)` (default `scores_at_best`). `own` sets columns this session's runs don't tag."""
    runs = experiment.runs().rename(columns=lambda c: c.removeprefix("tags."))
    runs = runs[runs.status == "FINISHED"].assign(**(own or {}))  # an interrupted run is retrained

    # This session's runs plus the reference runs, in the requested folds
    runs = pd.concat([runs, *references])[["run_id", *keys, "fold"]]
    runs = runs[runs.fold.isin(folds)].astype(keys).reset_index(drop=True)

    at_best = at_best or (lambda run_id: scores_at_best(experiment, run_id))
    scored = pd.DataFrame([at_best(r) for r in runs.run_id], index=runs.index)
    return pd.concat([runs, scored], axis=1)


def fold_table(results, by, columns, folds, title="RMSLE at the best epoch, all rows unless noted", **pivot_kw):
    """Validation RMSLE of each setup in `by` per fold and averaged, then the fold averages of `columns`
    ({column: label}). Returns the table and the per-setup means of `val` and `columns`, unrenamed."""
    means = results.groupby(by)[["val", *columns]].mean()

    table = results.pivot_table(index=by, columns="fold", values="val", **pivot_kw)[folds]
    table["average"] = means.val
    table = table.join(means[list(columns)].rename(columns=columns))

    return table.rename_axis(columns=title), means
