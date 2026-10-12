# Experiment 01: lgbm vs lgbm_building

## Request

Not recorded; card written before the Request field existed.

## Question

Does LightGBM improve when it knows which building a row comes from, and by how much?

## Plan

Two untuned LightGBM models on the same rows of each fold; the only difference is `building_id` as a categorical feature, which LightGBM splits on directly.

## Setup

- **Notebook:** [exp01_building.ipynb](../notebooks/experiments/exp01_building.ipynb)
- **Tracking:** MLflow `ashrae-baseline-building`, session `2026-10-11 11:16:15`, 8 runs named `<model> | <fold>`
- **What changes:** lgbm (the 33 shared features from [ashrae/features.py](../../ashrae/features.py)) against lgbm_building (the same plus `building_id`)
- **What stays fixed:** `LGBMRegressor`, 500 trees, learning rate 0.1, 63 leaves, everything else default; 4 seasonal folds of 2016, 2M training / 500K validation rows per fold, anomalies dropped from training only; seed 0
- **Metric:** validation RMSLE per fold and averaged, all rows, clean rows and per meter; training RMSLE

## Results

![Validation RMSLE per fold, lgbm vs lgbm_building, all rows and clean rows](../figures/exp01_building.png)

| Fold average | All rows | Clean rows | Training | Electricity | Chilled water | Steam | Hot water |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm | 1.280 | 1.045 | 0.916 | 1.019 | 1.501 | 1.647 | 1.625 |
| lgbm_building | 1.217 | 0.959 | 0.815 | 0.975 | 1.392 | 1.560 | 1.583 |

Validation RMSLE (500K rows per fold); meters on all rows.

## AI Summary

The building lowers fold-average RMSLE by 0.063 on all rows (0.086 on clean rows), in every fold, most on chilled water and steam.

## Conclusion

Yes. The building is worth 0.063 on all rows (0.086 on clean rows), in every fold, most on chilled water and steam. lgbm_building is the baseline from here on.
