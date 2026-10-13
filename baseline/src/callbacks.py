"""LightGBM callbacks."""
import lightgbm as lgb
import numpy as np
from tqdm import tqdm


def progress(desc, total, patience=None):
    """LightGBM callback: a progress bar with validation RMSLE, and early stopping on the "all" validation set."""
    bar = tqdm(total=total, desc=desc, unit="tree", dynamic_ncols=True)
    best = {"score": np.inf, "tree": 0}

    def callback(env):
        trees = env.iteration + 1
        scores = {name: value for name, _, value, _ in env.evaluation_result_list}
        score = scores["all"]
        if score < best["score"]:
            best.update(score=score, tree=trees)

        bar.update(1)
        if trees % 10 == 0:
            bar.set_postfix_str(f"val {score:.4f}, best {best['score']:.4f} at {best['tree']}")

        # Last tree, early-stopped or not: show the final best and close the bar
        stop = patience is not None and trees - best["tree"] >= patience
        if stop or trees == total:
            bar.set_postfix_str(f"best {best['score']:.4f} at {best['tree']}" + (" (early stop)" if stop else ""))
            bar.total = trees
            bar.close()

        if stop:
            raise lgb.callback.EarlyStopException(best["tree"] - 1, env.evaluation_result_list)

    callback.order = 30  # after record_evaluation, so the curve includes the last tree
    return callback
