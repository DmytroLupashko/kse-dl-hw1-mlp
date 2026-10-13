# Experiment 05: Optimizers

## Request

Not recorded; converted from `models/notebooks/optimizers_test.ipynb` (runs of 2026-09-27).

## Question

Which of Momentum, Nesterov and Adam gives the lowest fold-average validation RMSLE for mlp_4out_emb with emb_dim 12?

## Plan

Train mlp_4out_emb with emb_dim 12, the size picked in [exp03](exp03_emb_dim.md), with each optimizer at 3 learning rates for 40 epochs on the 4 seasonal folds, since Momentum and Nesterov converge slowly. Score each run at its best epoch, read from the validation curve after training (nothing stops early), and average over folds. Compare the optimizers each at its best learning rate; if that rate is at the edge of its grid, add one more in that direction. This is verified for batch 4096 only.

## Setup

- **Notebook:** [exp05_optimizers.ipynb](../notebooks/experiments/exp05_optimizers.ipynb)
- **Tracking:** MLflow `ashrae-mlp-optimizers`, session `2026-10-12 13:42:10`, 48 finished runs named `mlp_4out_emb | <optimizer> lr <lr> | <fold>`, tag `notebook = exp04_optimizers`
- **What changes:** Adam lr 3e-4, 1e-3, 3e-3; Momentum 0.9 lr 3e-3, 1e-2, 3e-2; Nesterov 0.9 lr 3e-3, 1e-2, 3e-2; added at the edge: Adam lr 1e-2, Momentum lr 1e-1, Nesterov lr 1e-1
- **What stays fixed:** mlp_4out_emb, emb_dim 12, 2 hidden layers of 64, ReLU, no BatchNorm, no weight decay (so AdamW equals Adam), one output per meter type, batch 4096, 40 epochs, features from [ashrae/features.py](../../ashrae/features.py), 2M training / 500K validation rows per fold, anomalies dropped from training only, seed 0
- **Metric:** validation RMSLE at the best epoch, per fold and averaged, all rows and clean rows; the best epoch; training RMSLE at the same epoch; mean change between consecutive epochs over the 5 epochs before the best one

## Results

![Validation RMSLE at the best epoch by learning rate for each optimizer, and each optimizer's curves at its best rate](../figures/exp05_optimizers.png)

Validation RMSLE at each run's best epoch, all rows (500K per fold), mean over the 4 folds; on the right, per-epoch validation (solid) and training (dashed) RMSLE.

## AI Summary

Adam gives the lowest validation RMSLE, 1.176 at lr 1e-2 against 1.197 for Momentum and 1.196 for Nesterov at 3e-2, and it is best in every fold. 1e-2 is still the highest rate tried for Adam and its curve is still falling there, so a higher rate might do slightly better.

## Conclusion

For 4096 size Adam clearly outperforms Nesterov and Momentum. For all batch sizes bigger learning rates should be considered (i.e. >= 1e-2)
