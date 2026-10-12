# Experiment 04: Anomalies

## Request

Not recorded; converted from `eda/notebooks/EDA.ipynb` section 9.

## Question

Does dropping anomaly rows from training lower the fold-average validation RMSLE of LightGBM, on clean rows and on all rows?

## Plan

Flag two kinds of rows in each building x meter series: every zero electricity reading, and steam or hot water at 0 for a week or more while the week is below 10 °C (chilled water: above 20 °C), "off when needed". Train untuned LightGBM on the 4 seasonal folds with different training rows dropped, and keep all validation rows. Two more rules are tested on top: stuck meters (the same non-zero value for 48+ hours) and spikes (above 10x the series' median that day). One more variant keeps weather in UTC, as before the fix of [exp02_time_zones](exp02_time_zones.md).

## Setup

- **Notebook:** [exp04_anomalies.ipynb](../notebooks/experiments/exp04_anomalies.ipynb)
- **Tracking:** none
- **What changes:** training rows dropped: none; zero electricity in runs of at least 720, 168, 24 or 1 h; off when needed; both rules; both plus stuck meters; both plus spikes; and none with weather in UTC
- **What stays fixed:** LightGBM (500 trees, learning rate 0.1, 63 leaves) on `log1p(meter_reading)`; features `meter`, `primary_use`, `square_feet`, `year_built`, 5 weather columns, local hour and day of week; 2M training / 500K validation rows per fold; seed 0
- **Metric:** validation RMSLE per fold and averaged, on all rows and on clean rows (neither rule)

## Results

| Training rows dropped | All rows | Clean rows | Share dropped |
| --- | --- | --- | --- |
| none | 1.286 | 1.140 | 0.0% |
| zero electricity, runs of 720+ h | 1.286 | 1.095 | 1.6% |
| zero electricity, runs of 168+ h | 1.281 | 1.075 | 2.0% |
| zero electricity, runs of 24+ h | 1.281 | 1.067 | 2.4% |
| every zero electricity reading | 1.280 | 1.061 | 2.6% |
| off when needed | 1.280 | 1.122 | 0.3% |
| both rules | 1.281 | 1.048 | 3.0% |
| both rules + stuck meters | 1.281 | 1.048 | 4.5% |
| both rules + spikes | 1.278 | 1.045 | 3.8% |
| none, weather in UTC | 1.290 | 1.147 | 0.0% |

Fold-average validation RMSLE of LightGBM, 500K validation rows per fold; clean rows are 93.9% (Jan–Mar) to 98.9% (Oct–Dec) of them.

## AI Summary

On clean rows dropping anomalies helps in every fold, 1.140 to 1.048 with both rules, but on all rows the average barely moves (1.286 to 1.281): Jul–Dec improve while Jan–Jun get worse (1.363 to 1.485 in Jan–Mar), because there the broken meters stay broken into the validation months. Stuck meters and spikes add nothing on top.

## Conclusion

[human]
