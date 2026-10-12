# Experiment 01: Adversarial validation

## Request

Not recorded; converted from `validation/notebooks/adversarial.ipynb` (runs of 2026-09-27).

## Question

Can LightGBM tell 2016 training rows from 2017–2018 test rows, and Jan–Sep from Oct–Dec 2016 rows (the first handout split), from the model inputs alone, measured by held-out AUC?

## Plan

For each pair of sets, sample rows from both and train a classifier to tell them apart from the model's input features (no target). AUC near 0.5 means the sets look the same to a model; feature importances show where they differ. For train vs test, repeat without weather, with weather only and with each weather feature alone.

## Setup

- **Notebook:** [exp01_adversarial.ipynb](../notebooks/experiments/exp01_adversarial.ipynb)
- **Tracking:** none
- **What changes:** pair of sets (train vs test; Jan–Sep vs Oct–Dec); feature set (all, without weather, weather only, one weather feature)
- **What stays fixed:** LightGBM classifier (200 trees, 63 leaves); 500K rows per set; AUC on a stratified held-out 20%; features `meter`, `primary_use`, `square_feet`, `year_built`, 5 weather columns, local hour and day of week; training rows without anomalies, validation and test rows whole; seed 0
- **Metric:** ROC AUC on the held-out rows; each feature's share of the gain

## Results

![Feature share of gain: train vs test, and Jan–Sep vs Oct–Dec](../figures/exp01_adversarial.png)

| Features | Train vs test | Jan–Sep vs Oct–Dec |
| --- | --- | --- |
| all | 0.751 | 0.924 |
| without weather | 0.520 | 0.521 |
| weather only | 0.711 | |
| one weather feature | 0.511 to 0.541 | |

ROC AUC on the held-out 20% of 1M sampled rows; the figure shows the all-features classifiers.

## AI Summary

Train and test differ only through weather (0.75, 0.52 without), and each weather feature alone gives 0.51 to 0.54, so together they fingerprint an hour of a year rather than shift; no feature needs dropping. Jan–Sep vs Oct–Dec reaches 0.92, all from weather, so a single Oct–Dec handout tests one season only.

## Conclusion

[human]
