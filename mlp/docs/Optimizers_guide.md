# Choosing the Optimizer

In the formulas, $w$ are the weights, $g_t = \nabla L(w_t)$ is the gradient at step $t$, $\eta$ is the learning rate, and all vector operations are element-wise.

## 1. Gradient descent

Gradient descent steps against the gradient.

$$
w_{t+1} = w_t - \eta \, g_t
$$

Mini-batch SGD estimates $g_t$ from a batch $B$ of rows instead of the whole dataset. Larger batches give a less noisy gradient but fewer steps per epoch, so they usually need a higher learning rate.

$$
g_t = \frac{1}{|B|} \sum_{i \in B} \nabla L_i(w_t)
$$

## 2. Momentum

Momentum steps along a velocity $v$, a running sum of past gradients. Directions that agree add up, and directions that flip sign cancel.

$$
\begin{aligned}
v_{t+1} &= \beta \, v_t + g_t \\
w_{t+1} &= w_t - \eta \, v_{t+1}
\end{aligned}
$$

## 3. Nesterov momentum

Nesterov takes the gradient at the look-ahead point instead of the current weights, so it brakes earlier before overshooting.

$$
\begin{aligned}
g_t &= \nabla L(w_t - \eta \beta \, v_t) \\
v_{t+1} &= \beta \, v_t + g_t \\
w_{t+1} &= w_t - \eta \, v_{t+1}
\end{aligned}
$$

## 4. Adam

Adam keeps a moving average of the gradient $m$ and of its square $v$, and divides one by the root of the other, so each weight gets its own step size. Dividing by $1 - \beta^t$ corrects for both averages starting at zero.

$$
\begin{aligned}
m_t &= \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t \\
v_t &= \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
w_{t+1} &= w_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\end{aligned}
$$

The defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$ and $\eta = 10^{-3}$. AdamW applies weight decay to the weights directly rather than through the gradient.

$$
w_{t+1} = w_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \, w_t \right)
$$
