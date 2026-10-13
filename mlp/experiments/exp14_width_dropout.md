# Experiment 14: Dropout for widths 256 and 512

## Request

create experiment 14: dropout on 512 and 256

## Question

Does dropout lower the fold-average validation RMSLE at the best epoch of width 256 or 512 below their 1.174 without it, and does either come more than 0.01 below the 1.168 of width 64?

## Plan

Train widths 256 and 512 at dropout 0.1, 0.3 and 0.5 after each hidden layer for 40 epochs on the 4 folds; both widths without dropout and width 64 are read from MLflow. Score each run at its best epoch, per fold and averaged over folds, and extend the grid if the best dropout is at its edge. Plot mean validation and training curves per width and dropout.

## Setup

- **Notebook:** [exp14_width_dropout.ipynb](../notebooks/experiments/exp14_width_dropout.ipynb)
- **Tracking:** MLflow `ashrae-mlp-width-dropout`, runs named `mlp_4out_emb_bn | width <width> | dropout <p> | <fold>`, tag `notebook = exp14_width_dropout`; reused: widths 256 and 512 without dropout from `ashrae-mlp-width`, session `2026-10-12 22:43:11` ([exp10](exp10_width.md)), and width 64 from `ashrae-mlp-batch-norm`, session `2026-10-12 21:52:51` ([exp09](exp09_batch_norm.md))
- **What changes:** width 256 or 512; dropout 0 (reused), 0.1, 0.3 or 0.5 after each hidden layer
- **What stays fixed:** mlp_4out_emb, emb_dim 12, 2 hidden layers, BatchNorm, ReLU, one output per meter type, Adam lr 1e-2, no weight decay, batch 4096, 40 epochs, no learning-rate schedule, warmup or early stopping, features from [ashrae/features.py](../../ashrae/features.py), 2M training / 500K validation rows per fold, anomalies dropped from training only, seed 0
- **Metric:** validation RMSLE at the best epoch, per fold and averaged, all rows and clean rows; the best epoch; training RMSLE at the best epoch and at epoch 40; seconds per epoch and training time to the best epoch

## Results

![Validation and training RMSLE per epoch for widths 256 and 512 at dropout 0, 0.1, 0.3 and 0.5, with width 64 dashed](../figures/exp14_width_dropout.png)

Validation RMSLE on all rows (500K per fold) and training RMSLE, per epoch, mean over the 4 folds.

| Width | Dropout | Jan–Mar | Apr–Jun | Jul–Sep | Oct–Dec | Average | Average, clean rows | Best epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0 | 1.386 | 1.195 | 1.008 | 1.085 | 1.168 | 0.912 | 35.3 |
| 256 | 0 | 1.394 | 1.200 | 1.012 | 1.090 | 1.174 | 0.911 | 15.8 |
| 256 | 0.1 | 1.382 | 1.188 | 1.007 | 1.081 | 1.164 | 0.904 | 26.0 |
| 256 | 0.3 | 1.381 | 1.190 | 1.006 | 1.075 | 1.163 | 0.907 | 25.0 |
| 256 | 0.5 | 1.384 | 1.192 | 1.010 | 1.084 | 1.167 | 0.915 | 31.3 |
| 512 | 0 | 1.395 | 1.196 | 1.013 | 1.091 | 1.174 | 0.910 | 11.8 |
| 512 | 0.1 | 1.384 | 1.186 | 1.008 | 1.083 | 1.165 | 0.910 | 10.3 |
| 512 | 0.3 | 1.378 | 1.185 | 1.001 | 1.072 | 1.159 | 0.901 | 26.0 |
| 512 | 0.5 | 1.379 | 1.190 | 1.002 | 1.077 | 1.162 | 0.906 | 25.0 |

Validation RMSLE at the best epoch on all rows (500K per fold) unless noted, per fold and averaged; best epoch is the mean over folds.

## AI Summary

Yes, dropout lowers both widths below 1.174. The best is dropout 0.3 at 1.163 for width 256 and 1.159 for width 512, but 1.159 is only 0.009 below width 64's 1.168, so neither clears the 0.01 margin.

Dropout 0.3 is consistent across folds. It beats the same width without dropout in all 4 folds, by 0.006 to 0.015 at width 256 and 0.011 to 0.019 at width 512. It also beats width 64 in all 4 folds, but by only 0.002 to 0.010 at width 256 and 0.007 to 0.013 at width 512. Only width 512 in Oct–Dec beats width 64 by more than 0.01.

## Conclusion

Dropout helps. The next step is evaluate dropout for 128 and 64 on smaller dropouts (0.3 and 0.1) and then compare
