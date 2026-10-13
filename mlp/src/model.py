"""The Lightning module that trains any network of networks.py."""
import lightning.pytorch as L
import torch
import torch.nn.functional as F
import torchmetrics


class EnergyModel(L.LightningModule):
    """Trains any network of networks.py with MSE on log1p readings, which is exactly RMSLE.

    Default optimizer: AdamW with `weight_decay` on weight matrices and embeddings only (with 0 it's plain Adam).
    `optimizer`, a function of the parameters, replaces it."""

    def __init__(self, net, lr=1e-3, weight_decay=0.0, optimizer=None):
        super().__init__()
        self.net = net
        self.optimizer = optimizer

        # "Hyperparameters" are aruments of __init__
        self.save_hyperparameters(ignore=["net", "optimizer"])

        # My: it assumes that we already applied log1p to target
        self.train_rmsle = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmsle = torchmetrics.MeanSquaredError(squared=False)

    # My: it relies on assumption that forward(x, meter, building).
    # I will not change it already, but it is important to notice.
    def forward(self, batch):
        return self.net(batch["x"], batch["meter"], batch["building"])

    def training_step(self, batch, batch_idx):
        pred = self(batch) # My: Equivalent to calling self.forward(batch)
        self.train_rmsle(pred, batch["y"]) # My: "rmsle" means that we are calling MSE assuming that log1p is applied
        self.log("train_rmsle", self.train_rmsle, on_step=False, on_epoch=True)

        return F.mse_loss(pred, batch["y"])

    def validation_step(self, batch, batch_idx):
        # My: log1p(x) >= 0 iff x >= 0, bullding can not use negative energy
        pred = self(batch).clamp(min=0)  # readings can't be negative
        self.val_rmsle(pred, batch["y"])
        self.log("val_rmsle", self.val_rmsle, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer(self.parameters())

        weights = [p for p in self.parameters() if p.ndim >= 2]  # weight matrices and the embedding
        others = [p for p in self.parameters() if p.ndim < 2]  # biases and BatchNorm
        groups = [
            {"params": weights, "weight_decay": self.hparams.weight_decay},
            {"params": others, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.hparams.lr)
