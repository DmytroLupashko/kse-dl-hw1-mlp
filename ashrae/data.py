"""Loading the processed data and train/validation splits (raw data: eda/src/raw_data.py)."""
import numpy as np
import pandas as pd

from .config import PROCESSED


def read_processed(name="train_merged.parquet", columns=None):
    """Output of eda/notebooks/data_aggregation.ipynb."""
    return pd.read_parquet(PROCESSED / name, columns=columns)


def split(df, val_months, n_train, n_val):
    """Sampled training rows outside `val_months`, without anomalies, and validation rows inside, all kept
    (they are scored in full)."""
    is_val = df.timestamp.dt.month.isin(val_months)

    train_df = df[~is_val & ~df.anomaly].sample(n_train, random_state=0)
    val_df = df[is_val].sample(n_val, random_state=0)
    return train_df, val_df


def log_target(df):
    """log1p meter readings of `df`, the target every model is trained and scored on."""
    return np.log1p(df.meter_reading.to_numpy(dtype="float32"))
