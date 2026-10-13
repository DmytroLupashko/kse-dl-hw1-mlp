"""log1p predictions of a trained `EnergyModel`, clipped at 0 since readings can't be negative."""
import torch

from ashrae.features import build_features

from .data import EnergyDataset


def _predict_batch(model, batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    return model(batch).clamp(min=0).cpu()


@torch.no_grad()
def predict(model, loader):
    """log1p predictions for every row of `loader`, clipped at 0."""
    model.eval()
    return torch.cat([_predict_batch(model, batch) for batch in loader]).numpy() # My: tensor -> vector


@torch.no_grad()
def predict_rows(model, rows, scaler, columns, year_median):
    """Clipped log1p predictions for raw `rows`, built with the training feature `columns`, scaler and year median."""
    X = build_features(rows, year_median).reindex(columns=columns, fill_value=0)  # same columns as in training
    x = scaler.transform(X).astype("float32")

    batch = EnergyDataset(rows, x).__getitems__(range(len(rows)))  # all rows as one batch
    return _predict_batch(model, batch).numpy()
