"""LightGBM-only inputs."""


def with_building(X, rows, categories):
    """`X` plus `building_id` as a categorical with `categories` (all ids, so training and test rows share them),
    which LightGBM splits on directly."""
    return X.assign(building_id=rows.building_id.astype(int).astype(categories).to_numpy())
