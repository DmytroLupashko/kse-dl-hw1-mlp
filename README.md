# ASHRAE Great Energy Predictor III

We predict hourly energy use of ~1,450 buildings at 16 sites for 4 meter types. The score is RMSLE ([EDA and Preprocessing](eda/eda_report.md#metric)).

The final submission is a blend of an MLP with building embeddings and LightGBM. It scores 1.178 on our validation, 1.086 public and 1.302 private on Kaggle. Unless noted, every score is the average over four seasonal folds of 2016 ([Validation Strategy](validation/validation_report.md#seasonal-4-fold)).

## Sections

Each section has a report and its notebooks:

1. [EDA and Preprocessing](eda/eda_report.md): data insights, metric analysis, time zone fix, anomaly removal, features
2. [Validation](validation/validation_report.md): adversarial validation, why Handout failed, seasonal 4-fold, expected leaderboard correlation
3. [Baseline](baseline/baseline_report.md): untuned LightGBM, with and without `building_id`, and its submission
4. [MLP](mlp/mlp_report.md): MLP experiments
5. [Ensembling and Submission](ensemble/ensemble_report.md): MLP + LightGBM blend and leaderboard results

| Submission | Local validation | Public | Private |
| --- | --- | --- | --- |
| lgbm_building ([baseline](baseline/baseline_report.md)) | 1.217 | 1.110 | 1.318 |
| blend, 0.65 · mlp_4out_emb_bn + 0.35 · lgbm_building ([ensemble](ensemble/ensemble_report.md)) | 1.178 | 1.086 | 1.302 |

## Project structure

```text
├── eda/            1. EDA and preprocessing
├── validation/     2. Validation strategy
├── baseline/       3. LightGBM baseline
├── mlp/            4. MLP
├── ensemble/       5. Blend and final submission
├── ashrae/         shared code: paths, data, features, metrics, MLflow tracking, submissions
├── docs/           assignment and development guide
├── ai_usage/       AI chat transcripts
└── data/           Kaggle data (not in git)
```

Every section folder has the same layout:

```text
<section>/
├── <section>_report.md     the section's report
├── experiments/            one card per experiment
├── notebooks/
│   ├── experiments/        one notebook per card
│   └── submissions/        notebooks that write a Kaggle submission
├── figures/                figures used in the report
└── src/                    the section's own code
```

Validation and EDA have no submissions; EDA also has [data_aggregation.ipynb](eda/notebooks/data_aggregation.ipynb), which builds `data/processed/` and must run first, since every other notebook reads it. A section's code is imported as `src` by its own notebooks and as `<section>.src` by the others, so `ashrae/` keeps only what no section owns. All MLflow runs are in `mlflow.db`.

## AI usage

AI tools helped write parts of the code and the reports. Full transcripts are in [ai_usage/](ai_usage/).

## References

1. 1st Place Solution in "ASHRAE Great Energy Predictor III" by Isamu & Matt — <https://www.kaggle.com/competitions/ashrae-energy-prediction/writeups/isamu-matt-1st-place-solution-team-isamu-matt>
2. "1st place solution: Correct Data is All You Need" by Volodymyr, "BirdCLEF 2023" competition — <https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y>
3. "Sites, buildings identified by internet search" for "ASHRAE Great Energy Predictor III" by Poe Dator — <https://www.kaggle.com/c/ashrae-energy-prediction/discussion/112841>
