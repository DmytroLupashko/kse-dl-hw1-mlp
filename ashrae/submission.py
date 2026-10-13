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
    submission = pd.DataFrame({"row_id": rows.row_id.to_numpy(), "meter_reading": reading})
    return submission.sort_values("row_id")


def check_submission(submission):
    """Raises if the row ids differ from sample_submission.csv or a reading is negative or not finite."""
    sample_ids = pd.read_csv(DATA / "sample_submission.csv", usecols=["row_id"]).row_id
    if not np.array_equal(submission.row_id, sample_ids):
        raise ValueError("row ids don't match sample_submission.csv")

    reading = submission.meter_reading
    if not (np.isfinite(reading).all() and (reading >= 0).all()):
        raise ValueError("meter readings must be finite and non-negative")


def predict_chunked(predict, rows, chunk_size):
    """`predict` applied to `rows` `chunk_size` rows at a time, concatenated."""
    chunks = [rows.iloc[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]
    return np.concatenate([predict(chunk) for chunk in chunks])


def save_submission(log_pred, rows, path):
    """Builds the submission from log1p predictions for `rows`, checks and saves it to `path` (if not None), and
    returns a summary of the readings."""
    submission = build_submission(rows, to_readings(log_pred, rows))

    if path:
        check_submission(submission)
        submission.to_csv(path, index=False, float_format="%.4f")
        print(f"saved {path}: {len(submission):,} rows")

    return submission.meter_reading.describe().round(2).to_frame()
