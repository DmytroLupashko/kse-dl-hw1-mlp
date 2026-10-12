# Experiment 02: Training rows

## Request

Not recorded; card written before the Request field existed.

## Question

Does lgbm_building get better with more training rows?

## Plan

Train lgbm_building from [exp01](exp01_building.md) on 0.5M, 1M, 2M, 4M, 8M and 14M rows per fold, sampled from the ~14.5M clean rows outside each validation block, and score every size on the same validation rows.

## Setup

- **Notebook:** [exp02_training_rows.ipynb](../notebooks/experiments/exp02_training_rows.ipynb)
- **Tracking:** MLflow `ashrae-baseline-training_rows`, session `2026-10-11 11:28:52`, 24 runs named `lgbm_building | <rows> | <fold>`
- **What changes:** training rows per fold, 0.5M to 14M
- **What stays fixed:** lgbm_building (500 trees, learning rate 0.1, 63 leaves, `building_id` as a categorical), 4 seasonal folds of 2016, the same 500K validation rows per fold for every size, anomalies dropped from training only, seed 0
- **Metric:** validation RMSLE per fold and averaged, all rows, clean rows and per meter; training RMSLE; seconds per fit

## Results

![Validation RMSLE by training rows, per fold and averaged, all rows and clean rows](../figures/exp02_training_rows.png)

| Training rows | Average, all rows | Average, clean rows | Training | Seconds per fit |
| --- | --- | --- | --- | --- |
| 0.5M | 1.223 | 0.967 | 0.801 | 8 |
| 2M | 1.217 | 0.959 | 0.815 | 10 |
| 14M | 1.213 | 0.954 | 0.819 | 35 |

Validation RMSLE, fold average (500K rows per fold).

## AI Summary

Barely: 0.5M to 14M rows lowers RMSLE by 0.010 (clean rows 0.013), about the noise level, and no fold moves by more than 0.013. Training time grows 3.5 times from 2M to 14M for 0.004.

## Conclusion

No, not meaningfully. 0.5M → 14M gains 0.010 (clean rows 0.013), about the noise level, and 2M → 14M only 0.004, for 3.5× the training time. Data size isn't what limits lgbm_building. Keep 2M rows for the next baseline experiments; the final model can use all rows, since one fit at 14M takes 35 s.
