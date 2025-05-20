# -*- coding: utf-8 -*-
"""
hyperparameters.py
Central place for all configurable knobs.
"""

from pathlib import Path

# ---- Paths ------------------------------------------------
BASE_DIR = Path(r"F:/afno_data")        # <-- adjust if your data lives elsewhere

# ---- Data -------------------------------------------------
LEVELS = ["250", "300", "350", "400", "500", "850"]
VAR_MAP = {
    "shum": "q",
    "temp": "t",
    "uwind": "u",
    "vwind": "v",
    "geo":  "z",
}

PAD_PATCH        = (8, 8)   # pad input so H and W are multiples of 8
FORECAST_HORIZON = 6        # how many frames ahead to predict  (← renamed)

# ---- Model ------------------------------------------------
EMBED_DIM = 128
DEPTH     = 4
MLP_RATIO = 4.0

# ---- Training --------------------------------------------
BATCH_SIZE = 32
EPOCHS     = 50
VAL_RATIO  = 0.2   # newest 20 % frames for validation
LR         = 1e-4
PATIENCE   = 5     # early-stopping patience
