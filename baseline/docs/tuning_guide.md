# Hyperparameter tuning: common approaches and what this project uses

## How to search

### Manual tuning

Change one setting by hand, look at the result, decide the next change. It is cheap, and you understand every result, but it depends on experience and does not scale to many hyperparameters.

### Grid search

List a few values per hyperparameter and try every combination. With 4 hyperparameters and 5 values each that is 5⁴ = 625 trainings. The cost grows exponentially with the number of hyperparameters, and most of the budget goes to values of unimportant hyperparameters.

### Random search

Draw each hyperparameter at random from a range, for a fixed number of trials. Usually only a few hyperparameters matter, and random search tries many distinct values of each of them, where a grid repeats the same few. For the same budget it usually beats grid search (Bergstra and Bengio, 2012). Ranges that span orders of magnitude, like the learning rate, are sampled on a log scale.

### Bayesian optimization

Use the results of earlier trials to choose the next one. A model of "setting → score" is fitted to the trials so far, and the next trial goes where that model expects a good score or is still uncertain. Common implementations:

- **TPE** (tree-structured Parzen estimator): the default in Optuna and Hyperopt
- **Gaussian processes**: scikit-optimize, and the method studied by Snoek et al. (2012)

It needs fewer trials than random search to reach the same score. The trials depend on each other, so they are harder to run in parallel.

### Multi-fidelity search and pruning

Stop bad trials early instead of training each one to the end. **Successive halving** starts many trials with a small budget, such as few trees, epochs or rows, keeps the best half, doubles their budget, and repeats. **Hyperband** (Li et al., 2017) runs successive halving at several starting budgets. Optuna calls this pruning: a trial that is clearly worse than earlier trials at the same point is stopped.

### Population-based training

Used mostly for large neural networks (Jaderberg et al., 2017). Many models train in parallel; every so often the weak ones copy the weights of strong ones and change their hyperparameters slightly. Hyperparameters such as the learning rate can then change during training. It is not used for gradient boosting.

### Staged search, one hyperparameter group at a time

A common way to tune gradient boosting, especially in Kaggle practice:

1. Fix a moderate learning rate.
2. Tune the hyperparameters that control tree size.
3. Tune regularization on top of the best tree size.
4. Optionally, lower the learning rate and add trees at the end.

Each stage keeps the best result of the previous one. It needs few trials, and the log shows what each change was worth. Its weakness is that it misses interactions: a setting that only helps together with another tree size is never tried.

## Number of trees and early stopping

For gradient boosting, the number of trees is not usually searched like the other hyperparameters. Training is run with a large maximum, and validation is scored after every tree:

- **Early stopping** ends training when validation hasn't improved for a set number of trees, the *patience*, and keeps the best tree count.
- The tree count depends on the learning rate. Halving the learning rate roughly doubles the trees needed. A comparison of learning rates at a fixed tree count favours the faster one, because the slower one hasn't finished.

Neural networks use the same idea, with epochs instead of trees.

## Practices that apply to any search

- **Fix the validation protocol first.** Split, metric and seeds stay the same for every trial; otherwise scores aren't comparable.
- **Keep the final estimate honest.** When the setting is chosen on the same folds it is scored on, the best score is optimistic, and more so the more trials there are (Cawley and Talbot, 2010). The fixes are a held-out test set that tuning never sees, or nested cross-validation, where an outer loop scores the whole tuning procedure.
- **Know the noise level.** If retraining with another seed moves the score by 0.01, a 0.005 gain between trials means nothing.
- **Log every trial, not only the best:** settings, scores, curves, time. MLflow, Weights & Biases and Optuna's storage do this.
- **Tune the important hyperparameters first** and keep the search space small. For LightGBM:

| Hyperparameter | What it controls |
| --- | --- |
| `learning_rate` + number of trees | how fast the model learns, and how long; tuned together |
| `num_leaves` | how complex each tree can be; the main capacity setting |
| `min_child_samples` (`min_data_in_leaf`) | minimum rows per leaf, against splits on a handful of rows |
| `colsample_bytree` (`feature_fraction`), `subsample` (`bagging_fraction`) | random subsets of features or rows per tree |
| `reg_alpha`, `reg_lambda` | L1 and L2 penalties on leaf values |
| `cat_smooth`, `min_data_per_group`, `max_cat_threshold` | how categorical features such as `building_id` are split |

- **Retrain the final model** with the chosen setting on all training data. With early stopping, use the tree count found during tuning, since there is no validation set left to stop on.
- **Don't tune on the leaderboard.** Picking settings by public leaderboard score fits that small part of the test set; the private score usually drops.

## What this project uses

Cards: [exp03 tree size](../experiments/exp03_tree_size.md), [exp04 regularization](../experiments/exp04_regularization.md), [exp05 tuned model](../experiments/exp05_tuned.md).

| Practice | In this project |
| --- | --- |
| Search method | staged search: Stage A tree size, Stage B regularization on top of 127 leaves (sizes within noise; 127 leaves room under the tree limit for slower settings), one change per trial |
| Validation | the 4 seasonal folds used everywhere; 2M training / 500K validation rows per fold |
| Learning rate | fixed at 0.1 |
| Number of trees | early stopping per fold: up to 5,000 trees, patience 200, on all validation rows; the trial's tree count is the mean over the folds |
| Logging | every fold of every trial is a run in MLflow (one experiment per card, `ashrae-baseline-<slug>`): settings, best tree count, RMSLE on all and clean rows, training time, full validation curves |
| Final model | the best trial retrained with its tree count and scored like the untuned baseline in [exp01](../experiments/exp01_building.md) |
| Honest estimate | not separate: the same folds choose and score, so the tuned score is slightly optimistic |
| Seeds | one per trial, seed 0 |

Stage A tries `num_leaves` 31, 63, 127 and 255. Stage B tries, one at a time on top of the Stage A winner: `min_child_samples` 100 and 500, `colsample_bytree` 0.8, smoother `building_id` categories (`cat_smooth` 50, `min_data_per_group` 500) and finer ones (`max_cat_threshold` 128).

Why this design:

- **Few trials.** The search runs on one laptop, and each fit takes from tens of seconds to minutes. A staged search needs 9 trials; a grid over the same values would need hundreds.
- **A readable log.** Each trial differs from its reference by one change, so the card can say what each change was worth.
- **Learning rate fixed at 0.1.** A first attempt compared learning rate 0.05 and 0.1 at a fixed 1,500 trees. Every curve was still falling at 1,500 trees, so the comparison favoured 0.1 for the wrong reason. It was stopped and deleted. With 0.05, the large trees would need several thousand trees per fit; the gain from a lower learning rate is usually small, and can be checked on the final setting alone.
- **Custom early stopping.** Each fit is scored on two validation sets, all rows and clean rows. LightGBM's built-in early stopping stops when either set stops improving; the project's callback watches only all rows, the metric being tuned.

What it leaves out, and what would come next:

- **Interactions** between tree size and regularization aren't searched. An Optuna run with TPE and pruning over the same hyperparameters would check whether a combination beats the staged result.
- **No untouched test set.** The leaderboard is the only fully independent check.
- **One seed.** Differences between trials below ~0.01 should be read as noise.

## References

1. J. Bergstra, Y. Bengio. Random Search for Hyper-Parameter Optimization. JMLR, 2012.
2. J. Snoek, H. Larochelle, R. P. Adams. Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS, 2012.
3. L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, A. Talwalkar. Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. JMLR, 2017.
4. M. Jaderberg et al. Population Based Training of Neural Networks. 2017.
5. T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama. Optuna: A Next-generation Hyperparameter Optimization Framework. KDD, 2019.
6. G. C. Cawley, N. L. C. Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR, 2010.
7. LightGBM documentation, Parameters Tuning: <https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html>
