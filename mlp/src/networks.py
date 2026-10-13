"""Networks trained by `EnergyModel`: each takes `(x, meter, building)` and returns one log1p prediction per row."""
import torch
from torch import nn

from ashrae.config import METERS

# CUSTOM BLOCK
class Block(nn.Module):
  def __init__(self, n_in, width=64, batch_norm=True, dropout=0.0):
    super().__init__()

    self.linear_layer = nn.Linear(n_in, width)
    self.batch_norm_layer = nn.BatchNorm1d(width) if batch_norm else None
    self.relu_layer = nn.ReLU()
    self.dropout_layer = nn.Dropout(dropout)

  def forward(self, X):
    X = self.linear_layer(X)

    if self.batch_norm_layer:
      X = self.batch_norm_layer(X)

    X = self.relu_layer(X)
    X = self.dropout_layer(X)

    return X

# CUSTOM BLOCK
class HiddenLayers(nn.Sequential):
    def __init__(self, n_in, n_layers=2, width=64, batch_norm=True, dropout=0.0):
        blocks = [Block(n_in, width, batch_norm, dropout)]

        for _ in range(n_layers - 1):
            blocks.append(Block(width, width, batch_norm, dropout))

        super().__init__(*blocks)

class MeterMLP(nn.Module):
    """One output per meter type; with `n_buildings`, a learned `emb_dim`-number vector per building is added to
    the input."""

    def __init__(self, n_features, n_buildings=0, emb_dim=16, dropout=0.0, n_layers=2, batch_norm=False, width=64):
        super().__init__()
        self.emb = nn.Embedding(n_buildings, emb_dim) if n_buildings else None

        n_in = n_features + (emb_dim if n_buildings else 0)
        self.hidden = HiddenLayers(n_in, batch_norm=batch_norm, dropout=dropout, n_layers=n_layers, width=width)
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
        self.hidden = HiddenLayers(n_features, batch_norm=batch_norm, width=width)
        self.out = nn.Linear(width, 1)

    def forward(self, x, meter, building):
        return self.out(self.hidden(x)).squeeze(1) # MY: transform [4096, 1] matrix -> [4096] vector
