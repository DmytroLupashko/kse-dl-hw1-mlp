"""The training loop of the MLP: Lightning callbacks, `train` and the fold loop of the experiments."""
import logging
import warnings
from time import perf_counter

import lightning.pytorch as L
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from tqdm import tqdm

from ashrae.metrics import rmsle

from .model import EnergyModel
from .predict import predict

# The data lives in memory, so DataLoader workers wouldn't help; hide the warnings that suggest them.
warnings.filterwarnings("ignore", ".*does not have many workers.*")
# Deprecation notice from inside Lightning, not from this code.
warnings.filterwarnings("ignore", ".*LeafSpec.*")
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


class Progress(L.Callback):
    """A progress bar over all training batches of a run, with the validation RMSLE after each epoch."""

    def __init__(self, desc):
        self.desc = desc
        self.best = {"score": np.inf, "epoch": 0}

    def on_train_start(self, trainer, model):
        self.bar = tqdm(total=trainer.max_epochs * trainer.num_training_batches, desc=self.desc, unit="batch",
                        dynamic_ncols=True)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.bar.update(1)

    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics["val_rmsle"].item()
        epoch = trainer.current_epoch + 1
        if score < self.best["score"]:
            self.best.update(score=score, epoch=epoch)

        best = f"best {self.best['score']:.4f} at {self.best['epoch']}"
        self.bar.set_postfix_str(f"epoch {epoch}, val {score:.4f}, {best}")

    def on_train_end(self, trainer, model):
        self.bar.close()


def train(make_net, train_loader, val_loader=None, *, epochs, logger=False, desc="", callbacks=(), **model_kw):
    """Seeds, builds the net and trains it for `epochs`; `callbacks` are extra Lightning callbacks. Returns the
    model."""
    L.seed_everything(0, verbose=False)  # before make_net, whose weights come from the seeded generator
    model = EnergyModel(make_net(), **model_kw)

    trainer = L.Trainer(max_epochs=epochs, logger=logger, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False, callbacks=[Progress(desc), *callbacks])
    trainer.fit(model, train_loader, val_loader)
    return model


class CleanRMSLE(L.Callback):
    """Logs validation RMSLE on the clean rows after every epoch, so it can be read at the best epoch."""

    def __init__(self, loader, y):
        self.loader, self.y = loader, y

    def on_validation_epoch_end(self, trainer, model):
        if not trainer.sanity_checking:
            model.log("val_rmsle_clean", rmsle(self.y, predict(model, self.loader)))


class EpochSeconds(L.Callback):
    """Logs the wall time of each epoch's training batches, without validation."""

    def on_train_epoch_start(self, trainer, model):
        self.start = perf_counter()

    def on_validation_start(self, trainer, model):
        if not trainer.sanity_checking:
            self.seconds = perf_counter() - self.start

    def on_validation_epoch_end(self, trainer, model):
        if not trainer.sanity_checking:
            model.log("epoch_seconds", self.seconds)


class EpochLR(L.Callback):
    """Logs the learning rate each epoch trained with."""

    def on_train_epoch_start(self, trainer, model):
        # read here: the scheduler has already stepped by validation end
        self.lr = trainer.optimizers[0].param_groups[0]["lr"]

    def on_validation_epoch_end(self, trainer, model):
        if not trainer.sanity_checking:
            model.log("lr", self.lr)


def fit_folds(experiment, folds, make_net, name, *, epochs, tags, params, done=(), key=(), batch_size=None,
              callbacks=(), **fit_kw):
    """Trains `make_net` on each fold of `data.fold_loaders` whose `(*key, fold)` isn't in `done`, as run
    `<name> | <fold>`. `batch_size` re-batches the same training rows; the rest goes to `Experiment.fit`."""
    for fold, d in folds.items():
        if (*key, fold) in done:
            continue
        train_loader, val_loader = d["loaders"]
        if batch_size:
            train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=train_loader.collate_fn)

        experiment.fit(
            make_net,
            train_loader,
            val_loader,
            run_name=f"{name} | {fold}",
            epochs=epochs,
            callbacks=[
                CleanRMSLE(*d["clean"]),
                EpochSeconds(),
                # stops a diverged run only: patience never runs out
                EarlyStopping("val_rmsle", patience=epochs, check_finite=True),
                *callbacks,
            ],
            tags={"fold": fold, **tags},
            params=params,
            **fit_kw,
        )
