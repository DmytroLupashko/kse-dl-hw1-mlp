# Ensemble report

We blend the final MLP ([mlp report](../mlp/mlp_report.md)) with lgbm_building ([baseline report](../baseline/baseline_report.md)) as a weighted average of their `log1p` predictions, `w · mlp + (1 − w) · lgbm`. On each fold both models are trained on the same rows, and the weight is chosen on the other three folds, so the blend is scored on rows its weight never saw ([exp01](experiments/exp01_final_blend.md)).

![Fold-average validation RMSLE on all rows for lgbm_building, the final MLP and their blend](figures/exp01_final_blend.png)

The blend scores 1.166 against 1.169 for the MLP alone and 1.217 for lgbm_building, with an MLP weight of 0.75 to 0.8. The 0.003 gain is below our 0.01 threshold: the tuned MLP leaves little for LightGBM to add.

## Kaggle

The submitted blend ([submission_blend.ipynb](notebooks/submissions/submission_blend.ipynb)) is an earlier version, 0.65 · mlp_4out_emb_bn + 0.35 · lgbm_building with the MLP before tuning. It scores 1.086 public and 1.302 private, against 1.110 and 1.318 for lgbm_building. The model is adequate and beats the baseline slightly on both parts of the leaderboard, in line with our validation.
