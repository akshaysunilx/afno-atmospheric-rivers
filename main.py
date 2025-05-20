# -*- coding: utf-8 -*-
"""
main.py – AFNO Weather Forecasting for Atmospheric River Study (Western U.S.)
Author: Fur
"""

import os
os.environ["HOME"] = os.environ.get("USERPROFILE", "C:/Users/grs05")
import gc
import pickle
import numpy as np
import xarray as xr
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from time import time

from physicsnemo.models.afno import AFNO
import visualise as viz
import hyperparameters as hp

# ---------------------- ENV SETUP ----------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HOME"] = os.environ.get("USERPROFILE", str(Path.home()))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("==============================================")
print(" AFNO Weather Forecast  –  Detailed run log")
print("==============================================")

# ---------------------- DATA LOADING ----------------------
def open_cube(var_key):
    tmpl = f"{var_key}_{{}}_squeezed.nc"
    return {lvl: xr.open_dataset(hp.BASE_DIR / tmpl.format(lvl)) for lvl in hp.LEVELS}

def load_all():
    print("🔹 Opening NetCDF files …")
    t0 = time()
    data = {var: open_cube(var) for var in hp.VAR_MAP}
    print(f"   ✅ Loaded in {time()-t0:.1f}s")
    return data

def _norm(arr):
    mu = arr.mean((1, 2), keepdims=True)
    sd = arr.std((1, 2), keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd

def build_tensor(raw):
    print("🔹 Normalising & stacking …")
    t0 = time()

    norm = {
        v: {lvl: _norm(raw[v][lvl][hp.VAR_MAP[v]].values.astype(np.float32))
            for lvl in hp.LEVELS}
        for v in hp.VAR_MAP
    }

    stk = np.stack(
        [norm["shum"][l] for l in hp.LEVELS] +
        [norm["temp"][l] for l in hp.LEVELS] +
        [norm["uwind"][l] for l in hp.LEVELS] +
        [norm["vwind"][l] for l in hp.LEVELS] +
        [norm["geo"][l] for l in hp.LEVELS],
        axis=1
    )

    h, w = stk.shape[-2:]
    pad_h = (hp.PAD_PATCH[0] - h % hp.PAD_PATCH[0]) % hp.PAD_PATCH[0]
    pad_w = (hp.PAD_PATCH[1] - w % hp.PAD_PATCH[1]) % hp.PAD_PATCH[1]
    padded = np.pad(stk, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    print(f"   ✅ Tensor shape {padded.shape}  |  {time()-t0:.1f}s")
    return padded

class NDArrayDS(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ---------------------- MODEL ----------------------
def make_model(C, H, W):
    return AFNO(
        inp_shape=(H, W),
        in_channels=C,
        out_channels=C,
        patch_size=list(hp.PAD_PATCH),
        embed_dim=hp.EMBED_DIM,
        depth=hp.DEPTH,
        mlp_ratio=hp.MLP_RATIO,
        drop_rate=0.1,
        num_blocks=4,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0
    )

# ---------------------- TRAINING LOOP ----------------------
def train_loop(model, tr_loader, va_loader):
    crit = torch.nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=hp.LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2)
    best_val = float("inf")
    patience_counter = 0

    print(f"\n🔹 Training up to {hp.EPOCHS} epochs …")

    for epoch in range(1, hp.EPOCHS + 1):
        model.train()
        tr_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            preds = model(xb)
            loss = crit(preds, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        # Validation
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                va_loss += crit(preds, yb).item()
        va_loss /= len(va_loader)
        sched.step(va_loss)

        print(f"Epoch {epoch:02d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), hp.BASE_DIR / "best_afno.pth")
        else:
            patience_counter += 1
            if patience_counter >= hp.PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

# ---------------------- INFERENCE ----------------------
def run():
    # ---- Load & preprocess
    raw = load_all()
    cube = build_tensor(raw)
    X = cube[:-hp.FORECAST_HORIZON]
    Y = cube[hp.FORECAST_HORIZON:]

    print(f"Total frames {len(cube)}  |  Pairs {len(X)}")

    split = int(len(X) * (1 - hp.VAL_RATIO))
    Xtr, Xva, Ytr, Yva = X[:split], X[split:], Y[:split], Y[split:]

    print(f"Chronological split  →  Train {len(Xtr)} | Val {len(Xva)}")

    tr_loader = DataLoader(NDArrayDS(Xtr, Ytr), batch_size=hp.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(NDArrayDS(Xva, Yva), batch_size=hp.BATCH_SIZE)

    print(f"Batches  →  Train {len(tr_loader)} | Val {len(va_loader)}")

    _, C, H, W = X.shape
    model = make_model(C, H, W).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model on {DEVICE}  |  {n_params:.2f} M parameters\n")

    # ---- Train
    train_loop(model, tr_loader, va_loader)

    # ---- Inference
    print("🔹 Re-loading best model & predicting full set …")
    model.load_state_dict(torch.load(hp.BASE_DIR / "best_afno.pth", map_location=DEVICE))
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.tensor(X[i:i + 64]).float().to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)

    out_file = hp.BASE_DIR / "afno_preds.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({"predictions": preds, "targets": Y}, f)

    print(f"✅ Saved predictions → {out_file}")

    viz.plot_sample(preds, Y, index=0, means={}, stds={}, variables=list(hp.VAR_MAP))

    print("\nRun complete ✔️")

# ---------------------- EXECUTE ----------------------
if __name__ == "__main__":
    run()
