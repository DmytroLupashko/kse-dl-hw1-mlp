"""Networks trained by `EnergyModel`: each takes `(x, meter, building)` and returns one log1p prediction per row."""
import torch
from torch import nn

from ashrae.config import METERS


def hidden_layers(n_in, batch_norm=True, dropout=0.0, n_layers=2, width=64):
    """`n_layers` hidden layers of `width` units; dropout goes after BatchNorm and ReLU."""
    def block(n):
        return [
            nn.Linear(n, width),
            *([nn.BatchNorm1d(width)] if batch_norm else []), # Syntax for adding layer based on codition
            nn.ReLU(),
            *([nn.Dropout(dropout)] if dropout else []),
        ]

    layers = block(n_in)  # input block
    for _ in range(n_layers - 1):
        layers += block(width)
    return nn.Sequential(*layers)


class MeterMLP(nn.Module):
    """One output per meter type; with `n_buildings`, a learned `emb_dim`-number vector per building is added to
    the input."""

    def __init__(self, n_features, n_buildings=0, emb_dim=16, dropout=0.0, n_layers=2, batch_norm=False, width=64):
        super().__init__()
        self.emb = nn.Embedding(n_buildings, emb_dim) if n_buildings else None

        n_in = n_features + (emb_dim if n_buildings else 0)
        self.hidden = hidden_layers(n_in, batch_norm=batch_norm, dropout=dropout, n_layers=n_layers, width=width)
        self.out = nn.Linear(width, len(METERS))

    def forward(self, x, meter, building):
        if self.emb is not None:
            x = torch.cat([x, self.emb(building)], dim=1)

        # one prediction per meter type; keep the row's own
        per_meter = self.out(self.hidden(x))
        return per_meter.gather(1, meter.unsqueeze(1)).squeeze(1)


class MLP(nn.Module):
    """A single output shared by all meter types."""

    def __init__(self, n_features, batch_norm=False, width=64):
        super().__init__()
        self.hidden = hidden_layers(n_features, batch_norm=batch_norm, width=width)
        self.out = nn.Linear(width, 1)

    def forward(self, x, meter, building):
        return self.out(self.hidden(x)).squeeze(1) # MY: transform [4096, 1] matrix -> [4096] vector
