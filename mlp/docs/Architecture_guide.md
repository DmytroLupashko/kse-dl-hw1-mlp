# Choosing the Architecture of a Simple Model

On tabular data, what the network sees matters more than how big it is. Decide inputs and outputs first, size last.

## 1. Inputs and outputs

If rows belong to something (a user, a store, a device), give the network its identity. Use an embedding of 8–16 numbers when each ID has many rows, or the ID's mean target when it has few. An embedding learns how the ID reacts to inputs; a mean only gives its level.

If groups react differently to the same inputs, give each group its own output and keep the one that matches the row. The groups still share the hidden layers.

Match the loss to the metric. For example, for RMSLE predict `log1p(y)` with MSE. Use no output activation for regression, sigmoid or softmax for probabilities.

Scale continuous inputs on training rows only. Encode cyclic ones (hour, weekday) as sine and cosine, and small categoricals as one-hot.

## 2. Hidden layers

A few ReLU layers of moderate width are enough. Make them wider before making them deeper; extra depth rarely helps. BatchNorm can speed up training. Add dropout only if the model overfits.

## 3. Sizing

Look at training and validation loss per epoch. Both high: make the network bigger or improve inputs. Both still falling: train longer. Validation rising: make it smaller, then regularize.

If training is far below validation, but neither regularization nor more capacity helps, the gap comes from differences between training and validation data, not from the network. New inputs fix it, architecture doesn't.

## 4. Experiments

Add one change at a time, starting from a plain MLP: identity, then outputs, then size. Fix the rule for keeping a change before running, and treat small differences as noise until other seeds confirm them. Keep everything else the same, and compare with a gradient-boosted tree model on the same features. On a tie, keep the smaller model.
