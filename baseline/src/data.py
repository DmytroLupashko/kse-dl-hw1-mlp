"""Fold data of the LightGBM experiments."""
import pandas as pd
from tqdm import tqdm

from ashrae.data import log_target, read_processed, split
from ashrae.features import fold_features

from .features import with_building


def make_folds(folds, n_train, n_val):
    """Per fold: features plus `building_id` as a categorical, `log1p` targets, the clean-rows mask and the meter."""
    df = read_processed()
    building = pd.CategoricalDtype(range(int(df.building_id.max()) + 1))  # all ids, so every fold shares categories

    data = {}
    for fold, months in tqdm(folds.items(), desc="fold data", unit="fold"):
        train_df, val_df = split(df, months, n_train, n_val)
        X_train, X_val = fold_features(train_df, val_df)
        data[fold] = {
            "X_train": with_building(X_train, train_df, building),
            "y_train": log_target(train_df),
            "X_val": with_building(X_val, val_df, building),
            "y_val": log_target(val_df),
            "clean": ~val_df.anomaly.to_numpy(),
            "meter": val_df.meter.to_numpy(),
        }
    return data
