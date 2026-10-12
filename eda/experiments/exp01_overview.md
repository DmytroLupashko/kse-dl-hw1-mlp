# Experiment 01: Data overview

## Request

Not recorded; converted from `eda/notebooks/EDA.ipynb` sections 1 to 5 and 7.

## Question

Which raw columns are too sparse or too weakly related to `log1p(meter_reading)` to keep as model inputs?

## Plan

Profile the three raw tables: size, keys, missing values per column and per site, duplicates and missing hours, building metadata, the target's distribution and zeros. Then join a sample of readings with building and weather data and measure each numeric column's Spearman correlation with `log1p(meter_reading)`, per meter type.

## Setup

- **Notebook:** [exp01_overview.ipynb](../notebooks/experiments/exp01_overview.ipynb)
- **Tracking:** none
- **What changes:** column, meter type
- **What stays fixed:** raw 2016 files (`train.csv`, `building_metadata.csv`, `weather_train.csv`), weather as given (UTC), 3M sampled readings for the correlations, seed 0
- **Metric:** share of missing values; Spearman correlation with `log1p(meter_reading)` per meter type

## Results

![Spearman correlation with log1p(reading), per meter type](../figures/exp01_overview.png)

Spearman correlation with `log1p(meter_reading)` on 3M sampled training rows (`square_feet` as `log1p`).

| Column | Missing |
| --- | --- |
| floor_count | 75.5% of buildings |
| year_built | 53.4% of buildings |
| cloud_coverage | 49.5% of weather rows |
| precip_depth_1_hr | 36.0% of weather rows |
| sea_level_pressure | 7.6% of weather rows (site 5 never reports it) |
| wind_direction | 4.5% of weather rows |

## AI Summary

Precipitation (|ρ| at most 0.06) and cloud coverage (weak, 36% and 49% missing) earn their drop, and floor_count is missing for 76% of buildings. Floor area is the strongest building feature (ρ 0.68 for electricity), and temperature works in opposite directions for cooling (0.43) and heating (about -0.45), so the model must combine meter type with weather.

## Conclusion

[human]
