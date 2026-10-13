# AI301 Deep Learning — Assignment #1: TODO

## General

- [x] Set up a clean environment (fresh `.venv` from `pyproject.toml` / `poetry.lock`, register Jupyter kernel, check every notebook runs from scratch)

- [x] Code is refactored
## EDA and Metric Analysis

- [x] Perform Exploratory Data Analysis (EDA)
- [x] Analyze the target metric: strengths, weaknesses, edge cases
- [x] Suggest complementary metrics for other aspects of performance
- [x] Propose an alternative target metric, with justification

## Validation

- [x] Design a validation strategy expected to correlate with Private/Public LB
- [x] Explain the motivation for the chosen strategy
- [x] Discuss possible reasons for low/high correlation with the LB
- [x] Perform adversarial validation (Linear/Logistic Regression is acceptable)

## Machine Learning Baseline

- [x] Do dedicated feature engineering for the tree-based model
- [x] Train a tree-based baseline (Random Forest, Gradient Boosting, sklearn, …)
- [x] Collect results: metrics, feature importance, parameter tuning logs
- [x] Submit predictions; compare Private/Public LB with local validation
- [x] Conclude whether the model is adequate

## Deep Learning Model

- [x]  Refactor DL environment to match baseline
- [x]  Do dedicated feature engineering and preprocessing for the DL model
- [x] Build a DL model beyond a plain MLP (multi-branch inputs/outputs, auxiliary losses, categorical embeddings, …)
- [x] Implement a custom layer with `nn.Module` + `nn.Parameter`
- [x] Implement an optimizer with `torch.optim.Optimizer

- [x] Regularization: Batch/Layer Norm
- [x] Regularization: Dropout 
- [x] Regularization: Weight Decay
- [x] Regularization: LR schedulers

- [x] Submit predictions; compare Private/Public LB with local validation
- [x] Conclude whether the model is adequate
- [x] From 
Pavlo: tune parameters of optimizers, collect the results
- [x] Try different size of layers
- [x] Try different Batch Sizes
- [x] Try different depth sizes

## Ensembling

- [x] Combine the baseline and DL models into an ensemble
- [x] Build a two-stage pipeline with reliable validation
- [x] Submit predictions; compare Private/Public LB with local validation

## Final Report (LaTeX or `README.md`)

- [x] Data insights
- [x] Feature engineering
- [x] Tried models, including failed ideas/experiments and what was learned from them
- [x] Chosen hyperparameters
- [x] Metrics and optimization curves
- [x] Model insights and conclusions
