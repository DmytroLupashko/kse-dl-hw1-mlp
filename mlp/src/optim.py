"""Optimizers and learning-rate schedules passed to `EnergyModel(optimizer=...)`."""
import torch
from torch.optim import Optimizer

class CustomOptimizer(Optimizer):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                self.update(p, group, self.state[p])

        return loss

    def update(self, p, group, state):
        raise NotImplementedError

class GD(CustomOptimizer):
    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})

    def update(self, p, group, state):
        p.add_(p.grad, alpha=-group["lr"])

class Momentum(CustomOptimizer):
    def __init__(self, params, lr, momentum=0.9):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    def update(self, p, group, state):
        if "velocity" not in state:
            state["velocity"] = torch.zeros_like(p)

        velocity = state["velocity"]
        velocity.mul_(group["momentum"]).add_(p.grad)

        p.add_(velocity, alpha=-group["lr"])

class Nesterov(CustomOptimizer):
    def __init__(self, params, lr, momentum=0.9):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    def update(self, p, group, state):
        if "velocity" not in state:
            state["velocity"] = torch.zeros_like(p)

        velocity = state["velocity"]
        velocity.mul_(group["momentum"]).add_(p.grad)

        lookahead = p.grad.add(velocity, alpha=group["momentum"])

        p.add_(lookahead, alpha=-group["lr"])

class Adam(CustomOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    def update(self, p, group, state):
        if "step" not in state:
            state["step"] = 0
            state["m"] = torch.zeros_like(p)
            state["v"] = torch.zeros_like(p)

        beta1, beta2 = group["betas"]
        state["step"] += 1

        m = state["m"]
        v = state["v"]

        m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

        m_hat = m / (1 - beta1 ** state["step"])
        v_hat = v / (1 - beta2 ** state["step"])

        p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])


OPTIMIZERS = {"Adam": Adam, "Momentum": Momentum, "Nesterov": Nesterov}


def adam_schedule(schedule, lr, epochs, patience=3):
    """`EnergyModel(optimizer=...)`: Adam from `lr`, stepped once per epoch by a cosine, step or plateau schedule.
    Plateau halves the rate after `patience` epochs without a lower validation RMSLE."""
    def optimizer(params):
        opt = Adam(params, lr=lr)

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
