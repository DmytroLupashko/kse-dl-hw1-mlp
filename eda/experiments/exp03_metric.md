# Experiment 03: Metric

## Request

Not recorded; converted from `eda/notebooks/EDA.ipynb` section 8.

## Question

How much does RMSLE charge for the same relative and absolute errors at different scales, and for a small prediction on a true zero?

## Plan

RMSLE is RMSE on `log1p` values, so training on `log1p(meter_reading)` with MSE loss optimizes it directly. Score single predictions with typical errors to see what the metric rewards and punishes.

## Setup

- **Notebook:** [exp03_metric.ipynb](../notebooks/experiments/exp03_metric.ipynb)
- **Tracking:** none
- **What changes:** true value (0, 100, 10,000) and prediction
- **What stays fixed:** RMSLE formula
- **Metric:** RMSLE of one prediction

## Results

| Case | True | Predicted | RMSLE |
| --- | --- | --- | --- |
| +10% on small value | 100 | 110 | 0.094 |
| -10% on small value | 100 | 90 | 0.104 |
| +10% on large value | 10,000 | 11,000 | 0.095 |
| +1,000 on small value | 100 | 1,100 | 2.389 |
| predict 5 when truth is 0 | 0 | 5 | 1.792 |

RMSLE of a single prediction.

## AI Summary

RMSLE measures relative error: a 10% miss costs about 0.095 at any scale, so the few huge series don't dominate. It is harsh near zero: predicting 5 for a true 0 costs 1.79, 19 times a 10% miss, so broken meters reading zero can dominate the score.

## Conclusion

[human]
