# Experiment 04: Regularization

## Request

Experiment 03 looks too big for me. Suggest how to split it, without losing MLflow results

## Question

Does any of five regularization changes on 127 leaves lower fold-average validation RMSLE by more than 0.01 (about seed noise) against plain 127 leaves?

## Plan

Base is 127 leaves from [exp03](exp03_tree_size.md). Five regularizations are tried one at a time, with the same early stopping as exp03:

- `min_child_samples` 100 (default 20): every leaf needs at least 100 training rows
- `min_child_samples` 500: every leaf needs at least 500 training rows
- `colsample_bytree` 0.8 (default 1.0): each tree is built on a random 80% of the features
- `cat_smooth` 50 and `min_data_per_group` 500 (defaults 10 and 100): when ordering buildings for a `building_id` split, a building's mean residual is pulled toward zero as if it had 50 extra rows with zero residual instead of 10, so buildings with few rows count less; and each side of the split needs at least 500 rows instead of 100
- `max_cat_threshold` 128 (default 32): one `building_id` split can send up to 128 buildings to one side instead of 32

This stage ran before exp03 was split into exp03 to exp05 (20 runs), so the notebook reads that pinned MLflow session by default and the base curve from exp03's 127 leaves runs; nothing is retrained.

## Setup

- **Notebook:** [exp04_regularization.ipynb](../notebooks/experiments/exp04_regularization.ipynb)
- **Tracking:** MLflow `ashrae-baseline-tuning`, session `2026-10-11 12:25:43`, tag `stage = B` (20 runs); base from `stage = A`, `num_leaves = 127`
- **What changes:** `min_child_samples` 100 or 500; `colsample_bytree` 0.8; smoother `building_id` categories (`cat_smooth` 50, `min_data_per_group` 500); finer ones (`max_cat_threshold` 128)
- **What stays fixed:** everything in exp03, with `num_leaves` 127
- **Metric:** validation RMSLE at each fold's best tree, all rows and clean rows, averaged over folds; tree count

## Results

![Regularization on 127 leaves](../figures/exp04_regularization.png)

Validation RMSLE, all rows (500K per fold), best so far by tree count, averaged over the 4 folds; the `cat_smooth` 50 with `min_data_per_group` 500 and `max_cat_threshold` 128 curves overlap the base exactly.

## AI Summary

No: every change stays within 0.003 of the base (1.1810), far inside seed noise. `cat_smooth` 50 with `min_data_per_group` 500 and `max_cat_threshold` 128 match the base exactly on every fold, so they changed nothing.

## Conclusion

Regularization does not help.
