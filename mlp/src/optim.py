"""Optimizers and learning-rate schedules passed to `EnergyModel(optimizer=...)`."""
import torch


# `EnergyModel(optimizer=OPTIMIZERS[name](lr))`; no weight decay, so AdamW equals Adam
OPTIMIZERS = {
    "Adam": lambda lr: lambda params: torch.optim.Adam(params, lr=lr),
    "Momentum": lambda lr: lambda params: torch.optim.SGD(params, lr=lr, momentum=0.9),
    "Nesterov": lambda lr: lambda params: torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
}


def adam_schedule(schedule, lr, epochs, patience=3):
    """`EnergyModel(optimizer=...)`: Adam from `lr`, stepped once per epoch by a cosine, step or plateau schedule.
    Plateau halves the rate after `patience` epochs without a lower validation RMSLE."""
    def optimizer(params):
        opt = torch.optim.Adam(params, lr=lr)

        if schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif schedule == "step":
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience)

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_rmsle", "interval": "epoch"}}

    return optimizer


def step_decay(lr, epochs):
    """Adam at `lr`, divided by 10 at half and three quarters of `epochs`."""
    return adam_schedule("step", lr, epochs)
