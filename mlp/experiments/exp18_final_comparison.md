# Experiment 18: Final model against the untuned MLP and the baseline

## Request

create experiment for comparing final model with mlp_4out_emb before tuning and lgbm baseline

Change the plan: I do not want to retrain model, use results that are already in MLFlow

## Question

Does the final model lower fold-average validation RMSLE more than 0.01 below both mlp_4out_emb before tuning (1.200) and lgbm_building (1.217), on at least 3 of 4 folds?

## Plan

Read all three models from MLflow and score each as trained, without picking an epoch: the final model at epoch 40, mlp_4out_emb before tuning at epoch 10. The figure shows validation RMSLE per fold and on average for the three models, on all rows and on clean rows.

## Setup

- **Notebook:** [exp18_final_comparison.ipynb](../notebooks/experiments/exp18_final_comparison.ipynb)
- **Tracking:** none, reads finished runs from MLflow: the final model (step schedule) from `ashrae-mlp-lr-schedule`, session `2026-10-13 10:29:04` ([exp17](exp17_lr_schedule.md)); mlp_4out_emb before tuning from `ashrae-mlp-building`, session `2026-10-12 12:03:22` ([exp02](exp02_building.md)); lgbm_building from `ashrae-baseline-building`, session `2026-10-11 11:16:15` ([baseline exp01](../../baseline/experiments/exp01_building.md))
- **What changes:** final model (mlp_4out_emb, emb_dim 12, BatchNorm, dropout 0.1 after each hidden layer, Adam lr 1e-2 divided by 10 at epochs 20 and 30, 40 epochs); mlp_4out_emb before tuning (emb_dim 16, no BatchNorm or dropout, Adam lr 1e-3, 10 epochs); lgbm_building
- **What stays fixed:** 2 hidden layers of 64, ReLU, one output per meter type and batch 4096 for both MLPs; features from [ashrae/features.py](../../ashrae/features.py), 2M training / 500K validation rows per fold, anomalies dropped from training only, seed 0
- **Metric:** validation RMSLE at the last epoch per fold and averaged, all rows and clean rows; training RMSLE at the last epoch

## Results

![Validation RMSLE at the last epoch per fold and on average for the final model, mlp_4out_emb before tuning and lgbm_building, on all rows and on clean rows](../figures/exp18_final_comparison.png)

Validation RMSLE at the last epoch on all rows (500K per fold) and on clean rows, per fold and averaged over the 4 folds.

## AI Summary

Yes: the final model scores 1.170, 0.031 below mlp_4out_emb before tuning (1.200) and 0.048 below lgbm_building (1.217), and clears the 0.01 margin against both in all 4 folds (by 0.020 to 0.046 and 0.031 to 0.080). The gain holds on clean rows too (0.907 against 0.934 and 0.959) and is largest in Jul–Sep.

## Conclusion

Final model has real improvement, which is not noise.