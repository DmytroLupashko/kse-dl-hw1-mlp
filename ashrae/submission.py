"""Turning log1p predictions into a Kaggle submission."""
import numpy as np
import pandas as pd

from .config import DATA, KBTU_TO_KWH


def to_readings(log_pred, rows):
    """Meter readings from log1p predictions for `rows`, with site 0 electricity back in kBTU, the unit Kaggle
    scores."""
    reading = np.expm1(log_pred)
    site0_electricity = ((rows.site_id == 0) & (rows.meter == 0)).to_numpy()
    reading[site0_electricity] /= KBTU_TO_KWH
    return reading


def build_submission(rows, reading):
    """The two columns Kaggle expects, in `row_id` order."""
    return pd.DataFrame({"row_id": rows.row_id.to_numpy(), "meter_reading": reading}).sort_values("row_id")


def check_submission(submission):
    """Raises if the row ids differ from sample_submission.csv or a reading is negative or not finite."""
    sample_ids = pd.read_csv(DATA / "sample_submission.csv", usecols=["row_id"]).row_id
    if not np.array_equal(submission.row_id, sample_ids):
        raise ValueError("row ids don't match sample_submission.csv")
    reading = submission.meter_reading
    if not (np.isfinite(reading).all() and (reading >= 0).all()):
        raise ValueError("meter readings must be finite and non-negative")
