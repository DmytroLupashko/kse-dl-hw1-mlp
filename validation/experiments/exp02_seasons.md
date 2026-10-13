# Experiment 02: Seasonal validation of mlp_4out_mean_bn

## Request

Not recorded; converted from `validation/notebooks/season_validation.ipynb` (runs of 2026-09-27).

## Question

How much of the spread between seasonal folds in validation RMSLE of mlp_4out_mean_bn comes from anomaly rows, measured as all-rows minus clean-rows RMSLE per fold?

## Plan

Anomalies are far more common in Jan–May than in Nov–Dec. Anomalies are dropped from training only, so train mlp_4out_mean_bn on each fold and score its validation predictions on all rows and on clean rows; the difference is what anomalies cost.

## Setup

- **Notebook:** [exp02_seasons.ipynb](../notebooks/experiments/exp02_seasons.ipynb)
- **Tracking:** MLflow `ashrae-validation-seasons`, session `2026-09-27 12:43:17`, 4 runs named `mlp_4out_mean_bn | <fold>`, tag `notebook = exp02_seasons`
- **What changes:** held-out fold
- **What stays fixed:** mlp_4out_mean_bn (mlp_4out_mean from [mlp exp02_building](../../mlp/experiments/exp02_building.md) with BatchNorm): 2 hidden layers of 64, BatchNorm, ReLU, one output per meter type, series mean input; Adam, lr 1e-3, batch 4096, 10 epochs; 2M training / 500K validation rows per fold; seed 0
- **Metric:** validation RMSLE on all rows and on clean rows (no anomaly), per fold; training RMSLE at the last epoch

## Results

![Training, validation and clean-row validation RMSLE per held-out season](../figures/exp02_seasons.png)

| Fold | Anomaly rows in validation | Anomaly cost (all minus clean) |
| --- | --- | --- |
| Jan–Mar | 6.1% | 0.480 |
| Apr–Jun | 3.6% | 0.317 |
| Jul–Sep | 1.3% | 0.122 |
| Oct–Dec | 1.1% | 0.087 |

RMSLE on 500K validation rows per fold; clean rows exclude both anomaly rules ([eda exp04_anomalies](../../eda/experiments/exp04_anomalies.md)).

## AI Summary

Most of it: on all rows the folds range from 1.12 to 1.46 in the order of their anomaly share, on clean rows only 0.93 to 1.04, and anomalies cost 0.48 in Jan–Mar against 0.09 in Oct–Dec. Of the 0.36 fold-average gap between training (0.88) and validation (1.24), about 0.25 is anomaly rows, so regularization can act on about 0.1 at most.

## Conclusion

[human]
