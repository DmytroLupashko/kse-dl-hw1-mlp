# Experiment 08: Batch 8192 at lr 1e-2 against the previous model

## Request

increment 08, 09, 10, add experiment for comparing batch 8192 with learning rate 0.01 vs previous model

## Question

Against the previous model (batch 4096, Adam lr 1e-3), do batch 4096 and batch 8192 at Adam lr 1e-2 lower fold-average validation RMSLE at the best epoch, on all rows and on clean rows, and how much training time to the best epoch does each save?

## Plan

Read the three setups from MLflow without training, score each at its best epoch averaged over folds, and plot validation RMSLE per epoch. Show clean-row RMSLE per fold to check the differences hold in every fold.

## Setup

- **Notebook:** [exp08_batch_8192.ipynb](../notebooks/experiments/exp08_batch_8192.ipynb)
- **Tracking:** none new; reused: batch 4096 lr 1e-3 and seconds per epoch from `ashrae-mlp-batch-size`, session `2026-10-12 15:01:13` ([exp04](exp04_batch_size.md)); batch 4096 Adam lr 1e-2 from `ashrae-mlp-optimizers`, session `2026-10-12 13:42:10` ([exp05](exp05_optimizers.md)); batch 8192 Adam lr 1e-2 from `ashrae-mlp-optimizers-large-batch`, session `2026-10-12 19:15:33` ([exp06](exp06_optimizers_large_batch.md))
- **What changes:** batch 4096 at lr 1e-3 against batch 4096 and batch 8192 at lr 1e-2
- **What stays fixed:** mlp_4out_emb, emb_dim 12, 2 hidden layers of 64, ReLU, no BatchNorm, no weight decay, one output per meter type, Adam, 40 epochs, no learning-rate schedule, warmup or early stopping, features from [ashrae/features.py](../../ashrae/features.py), 2M training / 500K validation rows per fold, anomalies dropped from training only, seed 0
- **Metric:** validation RMSLE at the best epoch, per fold and averaged, all rows and clean rows; the best epoch; training RMSLE at the same epoch; seconds per epoch (from exp04) and training time to the best epoch

## Results

![Validation RMSLE per epoch for batch 4096 at lr 1e-3, batch 4096 at lr 1e-2 and batch 8192 at lr 1e-2](../figures/exp08_batch_8192.png)

| setup | RMSLE, all rows | RMSLE, clean rows | best epoch | minutes to best epoch |
| --- | --- | --- | --- | --- |
| batch 4096, lr 1e-3 | 1.186 | 0.923 | 35.3 | 1.01 |
| batch 4096, lr 1e-2 | 1.176 | 0.913 | 29.0 | 0.83 |
| batch 8192, lr 1e-2 | 1.180 | 0.920 | 30.0 | 0.54 |

| clean rows | Jan–Mar | Apr–Jun | Jul–Sep | Oct–Dec |
| --- | --- | --- | --- | --- |
| batch 4096, lr 1e-3 | 0.920 | 0.884 | 0.877 | 1.012 |
| batch 4096, lr 1e-2 | 0.908 | 0.875 | 0.869 | 1.001 |
| batch 8192, lr 1e-2 | 0.916 | 0.886 | 0.874 | 1.004 |

Validation RMSLE at each run's best epoch, mean over the 4 folds unless split by fold, on all rows (500K per fold) and on clean rows; the figure shows the per-epoch curves averaged over folds.

## AI Summary

Both lr 1e-2 setups beat the previous model: batch 4096 by 0.010 on all and clean rows, in every fold, and 18% faster to its best epoch; batch 8192 by 0.006 on all rows and 0.003 on clean rows (slightly worse in Apr–Jun), but 47% faster (0.54 against 1.01 minutes per fold). Batch 8192 at lr 1e-2 is a clear speed gain for a small score gain, while batch 4096 at lr 1e-2 gives the best score.

## Conclusion

The difference between 4096 lr 1e-2 and 4096 lr 1e-3 is exacly 0.01, which is enough to consider difference as significant. I decided that slight imporovement in accuracy is more important for me, so decide to choose 4096.
