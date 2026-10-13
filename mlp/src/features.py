"""The building × meter series mean, an MLP input tried instead of the building embedding."""
import numpy as np
import pandas as pd


def add_series_mean(X, df, train_df):
    """Adds the mean log1p reading of each building × meter series in `train_df`; series with no training rows
    get their meter type's mean."""
    y = np.log1p(train_df.meter_reading)
    series_mean = y.groupby([train_df.building_id, train_df.meter]).mean()
    meter_mean = y.groupby(train_df.meter).mean()

    series_of_rows = pd.MultiIndex.from_arrays([df.building_id, df.meter])
    values = series_mean.reindex(series_of_rows).to_numpy()

    unseen = np.isnan(values)
    values[unseen] = df.meter.map(meter_mean).to_numpy()[unseen]
    return X.assign(series_mean=values.astype("float32"))
