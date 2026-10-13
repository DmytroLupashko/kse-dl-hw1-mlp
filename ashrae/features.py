"""Model inputs, built the same way for training, validation and test rows."""
import numpy as np
import pandas as pd

from .config import METERS


def _cyclic(values, period):
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_features(df, year_median):
    """`year_median` comes from the training rows, so validation and test rows are filled the same way."""
    X = pd.DataFrame(index=df.index)

    # Building
    X["log_square_feet"] = np.log1p(df.square_feet)
    X["year_built_missing"] = df.year_built.isna()
    X["year_built"] = df.year_built.fillna(year_median)

    # Weather
    for col in ["air_temperature", "dew_temperature", "sea_level_pressure", "wind_speed"]:
        X[col] = df[col]
    X["wind_dir_sin"], X["wind_dir_cos"] = _cyclic(df.wind_direction, 360)

    # Local time, as points on a circle so that 23:00 is next to 00:00
    X["hour_sin"], X["hour_cos"] = _cyclic(df.local_timestamp.dt.hour, 24)
    X["dow_sin"], X["dow_cos"] = _cyclic(df.local_timestamp.dt.dayofweek, 7)

    # One-hot meter type (all four columns, even if `df` lacks a meter) and primary use
    meter = pd.get_dummies(pd.Categorical(df.meter, categories=list(METERS)), prefix="meter")
    use = pd.get_dummies(df.primary_use, prefix="use")
    meter.index = use.index = df.index

    return pd.concat([X, meter, use], axis=1).astype("float32")


def fold_features(train_df, val_df):
    """Features of a fold's training and validation rows, both filled with the training rows' median year."""
    year_median = train_df.year_built.median()
    return build_features(train_df, year_median), build_features(val_df, year_median)
