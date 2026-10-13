"""Raw Kaggle CSVs, time zones, weather cleaning and anomaly flags, used to build data/processed/."""
import numpy as np
import pandas as pd

from ashrae.config import DATA


# Standard-time offset of each site, local time = UTC + offset (eda/experiments/exp02_time_zones.md)
UTC_OFFSET = {0: -5, 1: 0, 2: -7, 3: -5, 4: -8, 5: 0, 6: -5, 7: -5,
              8: -5, 9: -6, 10: -7, 11: -5, 12: 0, 13: -6, 14: -5, 15: -5}
# Weather is shifted to the meters' clock; site 14's meters are in UTC too, so it isn't shifted
WEATHER_SHIFT_HOURS = {**UTC_OFFSET, 14: 0}

WEATHER_COLS = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr",
                "sea_level_pressure", "wind_direction", "wind_speed"]


def read_building():
    dtype = {"site_id": "int8", "building_id": "int16", "primary_use": "category", "square_feet": "int32",
             "year_built": "float32", "floor_count": "float32"}
    return pd.read_csv(DATA / "building_metadata.csv", dtype=dtype)


def read_readings(name):
    """`train.csv` or `test.csv`."""
    dtype = {"row_id": "int32", "building_id": "int16", "meter": "int8", "meter_reading": "float32"}
    return pd.read_csv(DATA / name, parse_dates=["timestamp"], dtype=dtype)


def read_weather(name, columns=WEATHER_COLS):
    """`weather_train.csv` or `weather_test.csv`, with only the given weather columns."""
    dtype = {"site_id": "int8", **{c: "float32" for c in columns}}
    usecols = ["site_id", "timestamp", *columns]
    return pd.read_csv(DATA / name, usecols=usecols, parse_dates=["timestamp"], dtype=dtype)


def clean_weather(w):
    """Each site on a full hourly grid, gaps filled by linear interpolation within the site; a column a site never
    reports gets the other sites' mean at that hour. Wind direction is interpolated as sin/cos."""
    columns = [c for c in w.columns if c not in ("site_id", "timestamp")]

    # Every site gets a row for every hour; missing hours are NaN for now
    hours = pd.date_range(w.timestamp.min(), w.timestamp.max(), freq="h")
    grid = pd.MultiIndex.from_product([w.site_id.unique(), hours], names=["site_id", "timestamp"])
    w = w.set_index(["site_id", "timestamp"]).reindex(grid).reset_index()

    # An angle can't be interpolated directly (359° and 1° would average to 180°), its sin and cos can
    rad = np.deg2rad(w.pop("wind_direction"))
    w["wd_sin"], w["wd_cos"] = np.sin(rad), np.cos(rad)

    # Fill gaps within each site, then columns a site lacks entirely from the other sites
    cols = w.columns.drop(["site_id", "timestamp"])
    w[cols] = w.groupby("site_id")[cols].transform(lambda s: s.interpolate(limit_direction="both"))
    w[cols] = w[cols].fillna(w.groupby("timestamp")[cols].transform("mean"))

    w["wind_direction"] = np.rad2deg(np.arctan2(w.pop("wd_sin"), w.pop("wd_cos"))) % 360
    return w.astype({c: "float32" for c in columns})


def anomaly_flags(df):
    """The two anomaly rules of eda/experiments/exp04_anomalies.md, per row of `df` (needs building_id, meter,
    timestamp, meter_reading, air_temperature). Also returns the length in hours of the zero run each zero
    reading belongs to."""
    s = df.sort_values(["building_id", "meter", "timestamp"])
    series = s.building_id.astype(int) * 4 + s.meter.astype(int)  # one id per building × meter series

    # Runs of consecutive zero readings within one series: a new run starts where the value or the series changes
    zero = s.meter_reading.eq(0)
    run_starts = zero.ne(zero.shift()) | series.ne(series.shift())
    zero_run = run_starts.cumsum()

    # Per row: the length of its zero run (0 for non-zero readings) and the run's mean air temperature
    run_hours = zero.groupby(zero_run).transform("size").where(zero, 0)
    run_temp = s.air_temperature.groupby(zero_run).transform("mean")

    # Off when needed: steam or hot water at 0 for a week in the cold, chilled water at 0 for a week in the heat
    week_long = run_hours >= 168
    heating_off_in_cold = s.meter.isin([2, 3]) & (run_temp < 10)
    cooling_off_in_heat = (s.meter == 1) & (run_temp > 20)

    flags = pd.DataFrame({
        "zero_electricity": zero & (s.meter == 0),
        "off_when_needed": week_long & (heating_off_in_cold | cooling_off_in_heat),
        "zero_run_hours": run_hours,
    })
    return flags.reindex(df.index)
