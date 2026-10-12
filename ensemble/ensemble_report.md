# Ensembling and Submission

Experiments, one card each: [exp01_blend](experiments/exp01_blend.md). Submission: [submission_blend.ipynb](notebooks/submissions/submission_blend.ipynb).

## Blend

We blend the MLP and LightGBM as a weighted average of their `log1p` predictions: `w · mlp + (1 − w) · lgbm`. Both get the same 33 features ([exp01_blend](experiments/exp01_blend.md)).

It's a simple two-stage pipeline. Stage 1 trains both models on each fold and keeps their predictions on the held-out months. Stage 2 picks the blend weight on three folds and scores it on the fourth, so the ensemble is checked on rows neither stage saw. We didn't try a trained second-stage model (stacking).

![Blend RMSLE by MLP weight, per fold and averaged](figures/exp01_blend.png)

| Model | RMSLE, all rows | RMSLE, clean rows |
| --- | --- | --- |
| mlp_4out_emb_bn | 1.187 | 0.924 |
| lgbm | 1.280 | 1.045 |
| lgbm_building (+ `building_id` as a categorical) | 1.217 | 0.959 |
| blend with lgbm | 1.183 | 0.919 |
| blend with lgbm_building | 1.178 | 0.911 |

- The blend helps a little, in every fold: 1.187 → 1.178. The best MLP weight is stable, 0.6–0.7.
- The gain is small because both models miss the same rows (residual correlation 0.88), mostly anomalies and seasons.
- It helps heating and cooling, not electricity.

The final submission is 0.65 · mlp_4out_emb_bn + 0.35 · lgbm_building, both retrained on 2M rows from all of 2016 with anomalies dropped ([submission_blend.ipynb](notebooks/submissions/submission_blend.ipynb)).

## Leaderboard results

| Submission | Local validation | Public | Private |
| --- | --- | --- | --- |
| lgbm_building ([baseline](../baseline/baseline_report.md)) | 1.217 | 1.110 | 1.318 |
| blend, 0.65 · mlp_4out_emb_bn + 0.35 · lgbm_building ([submission_blend.ipynb](notebooks/submissions/submission_blend.ipynb)) | 1.178 | 1.086 | 1.302 |

- The folds rank the two models correctly: the blend wins locally and on both leaderboard parts, though by less on Kaggle.
- Public is about 0.1 better than our validation, private about 0.1 worse, for both models.
- Public and private differ by ~0.21, about as much as our seasons differ (1.03–1.40 for the blend).

Two submissions only show the ranking holds for this pair, not in general.
