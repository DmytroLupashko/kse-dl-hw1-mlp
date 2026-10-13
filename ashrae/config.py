"""Paths and constants shared by all notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the Homework 1 folder
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"  # one tracking database for every notebook

METERS = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
KBTU_TO_KWH = 0.2931  # site 0 electricity is in kBTU

# 2016 in four 3-month blocks: each fold validates on one block and trains on the other nine months
FOLDS = {"Jan–Mar": [1, 2, 3], "Apr–Jun": [4, 5, 6], "Jul–Sep": [7, 8, 9], "Oct–Dec": [10, 11, 12]}
