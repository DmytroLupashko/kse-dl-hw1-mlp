# Experiment 03: Tuning lgbm_building

## Question

How much does tuning improve lgbm_building over its untuned settings (500 trees, learning rate 0.1, 63 leaves; 1.217 in [exp01](exp01_building.md))?

## Setup

- **Notebook:** [exp03_tuning.ipynb](../notebooks/exp03_tuning.ipynb)
- **MLflow experiment:** `ashrae-baseline-tuning`
- **Model and parameters:** lgbm_building, `LGBMRegressor`; tuned parameters below, everything else default
- **Features:** the 33 shared features from [ashrae/features.py](../../ashrae/features.py) plus `building_id` as a categorical
- **Data:** 4 seasonal folds of 2016, 2M training / 500K validation rows per fold ([exp02](exp02_training_rows.md): more rows don't help), anomalies dropped from training only
- **Seed:** 0, one per run

## Description

Learning rate stays at 0.1. Every trial trains on all 4 folds with up to 5,000 trees; each fold stops once validation RMSLE hasn't improved for 200 trees. The trial's tree count is the mean of the folds' best tree counts.

- **Stage A, tree size:** 31, 63, 127, 255 leaves.
- **Stage B, regularization:** on top of the best Stage A trial, one change at a time: `min_child_samples` 100 and 500, `colsample_bytree` 0.8, smoother `building_id` categories (`cat_smooth` 50, `min_data_per_group` 500), finer ones (`max_cat_threshold` 128).
- **Final:** the best trial retrained with its tree count and scored like exp01.

The choice is made on the same folds it is scored on, so the tuned score is slightly optimistic.

A first attempt (learning rate 0.05, a fixed 1,500 trees) was stopped and its runs deleted: every validation curve was still falling at 1,500 trees, so the comparison favoured the faster learning rate.

## Parameter tuning log

## Results

## Conclusion
