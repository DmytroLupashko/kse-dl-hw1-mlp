# Baseline: LightGBM

Notebook: [submission_baseline.ipynb](notebooks/submission_baseline.ipynb).

## Setup

LightGBM on the same 33 features as the MLP, untuned: 500 trees, learning rate 0.1, 63 leaves ([ensemble.ipynb](../ensemble/notebooks/ensemble.ipynb)). Two versions:

| Model | RMSLE, all rows | RMSLE, clean rows |
| --- | --- | --- |
| lgbm | 1.280 | 1.045 |
| lgbm_building (+ `building_id` as a categorical) | 1.217 | 0.959 |

LightGBM was also the quick model for checking preprocessing in the EDA: the time zone fix took it from 1.290 to 1.286, and dropping anomalies from 1.286 to 1.281 on all rows, 1.140 to 1.048 on clean rows ([EDA report](../eda/eda_report.md#preprocessing)).

## Submission

lgbm_building alone, trained on the same 2M rows as the final blend (all of 2016, anomalies dropped, same seed), to score the baseline on the leaderboard next to the blend:

| Submission | Local validation | Public | Private |
| --- | --- | --- | --- |
| lgbm_building | 1.217 | 1.110 | 1.318 |

The blend scores 1.086 public and 1.302 private ([Leaderboard results](../ensemble/ensemble_report.md#leaderboard-results)).

## Is it adequate?

LightGBM is fine as a baseline, not as the final model. It is behind the MLP in every fold (1.280 vs 1.187). With `building_id` it gets 1.217, so its problem is not knowing the building. It is untuned, so tuning could only help ([ensemble.ipynb](../ensemble/notebooks/ensemble.ipynb)).
