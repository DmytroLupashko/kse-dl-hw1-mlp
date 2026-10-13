"""Datasets and loaders of the MLP: scaled features plus the meter and building ids each network takes."""
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ashrae.config import FOLDS
from ashrae.data import log_target, read_processed, split
from ashrae.features import fold_features


class EnergyDataset(Dataset):
    """Scaled features `x` plus meter, building and (if `df` has readings) log1p target of each row of `df`, all as
    tensors."""

    # Here we convert pd to tensors, while extracting specific features we will use later
    def __init__(self, df, x):
        self.x = torch.from_numpy(x)
        self.meter = torch.from_numpy(df.meter.to_numpy().astype("int64"))
        self.building = torch.from_numpy(df.building_id.to_numpy().astype("int64"))
        self.y = torch.from_numpy(log_target(df)) if "meter_reading" in df else None  # test rows have none

    def __len__(self):
        return len(self.x)

    # MY: it is specifically for pytoch
    # The sense of thunder methods in python is that they are reserved for something. That's it, for toher things it is just naming convention
    # Just a hook for speedup using batches, nothing more
    def __getitems__(self, idx):  # a whole batch in one indexing step, instead of row by row
        idx = torch.as_tensor(idx)
        batch = {"x": self.x[idx], "meter": self.meter[idx], "building": self.building[idx]}
        return batch if self.y is None else {**batch, "y": self.y[idx]}


def _keep_batch(batch):
    return batch  # we implemented __getitems__, so keep this as identity


# The default batch size is for validation: there we just evaluate using forward pass, so it does not matter
def _loader(df, X, scaler, batch_size=65536, shuffle=False):
    dataset = EnergyDataset(df, scaler.transform(X).astype("float32"))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_keep_batch)


def make_loaders(train_df, val_df, X_train, X_val, batch_size=4096):
    """Training and validation loaders, with a scaler fitted on the training rows. With `val_df=None` (final
    training on all data) the validation loader is None.

    Experiments: `train_loader, val_loader, _ = make_loaders(train_df, val_df, X_train, X_val)`
    Submission:  `train_loader, _, scaler = make_loaders(train_df, None, X_train, None)`, then `scaler.transform(X_test)`
    """
    scaler = StandardScaler().fit(X_train)
    # Note that batch size affects training process
    train_loader = _loader(train_df, X_train, scaler, batch_size, shuffle=True)
    val_loader = None if val_df is None else _loader(val_df, X_val, scaler)  # To validate and tune
    return train_loader, val_loader, scaler  # scaler: to scale the test set


def fold_loaders(fold_names, n_train, n_val):
    """Loaders of each fold in `fold_names`, plus a loader and log1p targets of its clean validation rows (for
    `CleanRMSLE`). Returns the folds, the number of buildings and the number of features."""
    df = read_processed()
    n_buildings = int(df.building_id.max()) + 1

    folds = {}
    for fold in fold_names:
        train_df, val_df = split(df, FOLDS[fold], n_train, n_val)
        X_train, X_val = fold_features(train_df, val_df)
        train_loader, val_loader, scaler = make_loaders(train_df, val_df, X_train, X_val)

        clean = ~val_df.anomaly.to_numpy()
        clean_loader = _loader(val_df[clean], X_val[clean], scaler)

        folds[fold] = {
            "loaders": (train_loader, val_loader),
            "clean": (clean_loader, log_target(val_df)[clean]),
        }
    return folds, n_buildings, X_train.shape[1]
