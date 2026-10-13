"""Colors and curve plots of the MLP figures."""
import numpy as np
import pandas as pd

from ashrae.plots import TEXT, style_axis


# One color per model in every figure; gray marks a model dropped after its first comparison.
# A `_bn` (BatchNorm) variant shares its model's color: the two never appear in the same figure.
MODEL_COLORS = {"mlp": "#999999", "mlp_4out": "#2a78d6", "mlp_4out_bn": "#7b52c9", "mlp_4out_mean": "#eb6834",
                "mlp_4out_emb": "#1baf7a", "mlp_4out_mean_bn": "#eb6834", "mlp_4out_emb_bn": "#1baf7a"}


def draw_curves(ax, groups, key, label, title):
    """One line per `(name, runs, color, ls, alpha, zorder)` in `groups`: column `key` of `runs`, mean over folds
    (a diverged run is shorter)."""
    for name, runs, color, ls, alpha, zorder in groups:
        curve = pd.DataFrame(runs[key].tolist()).mean()  # one row per fold, so the mean over folds per epoch
        epochs = np.arange(1, len(curve) + 1)
        ax.plot(epochs, curve, lw=2, color=color, ls=ls, alpha=alpha, zorder=zorder, label=name)

    ax.set(xlabel="epoch", ylabel=label)
    ax.set_title(title, loc="left", color=TEXT, fontsize=9)
    style_axis(ax)
