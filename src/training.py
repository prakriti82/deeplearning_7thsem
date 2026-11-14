"""Shared training loop. Owner: M3. Early stopping on VALIDATION loss only."""
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from . import config
from .models import build_model


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _loader(xh, xf, y, batch, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(xh), torch.from_numpy(xf), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)


def predict_scaled(model: nn.Module, xh: np.ndarray, xf: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(xh), batch):
            outs.append(model(torch.from_numpy(xh[i:i + batch]), torch.from_numpy(xf[i:i + batch])).numpy())
    return np.concatenate(outs) if outs else np.zeros((0, config.HORIZON, len(config.TARGETS)), np.float32)


def train_model(name: str, train: tuple, val: tuple, hparams: Dict = None, verbose: bool = True):
    """train/val are (x_hist, x_fut, y). Returns (best_model, info dict)."""
    hp = {**config.HPARAMS, **(hparams or {})}
    set_seed()
    xh, xf, y = train
    model = build_model(name, xh.shape[2], xf.shape[2], **hp)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()
    tl = _loader(xh, xf, y, hp["batch_size"], True)
    vh, vf, vy = val
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(hp["epochs"]):
        model.train()
        tr = 0.0
        for bh, bf, by in tl:
            opt.zero_grad()
            loss = loss_fn(model(bh, bf), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(bh)
        tr /= max(len(xh), 1)
        vl = float(np.mean((predict_scaled(model, vh, vf) - vy) ** 2))
        hist.append({"epoch": ep + 1, "train_loss": tr, "val_loss": vl})
        if verbose:
            print(f"[{name}] epoch {ep + 1:02d} train={tr:.4f} val={vl:.4f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    info = {"model": name, "best_val_loss": best, "epochs_run": len(hist),
            "train_time_s": time.time() - t0, "history": hist, "hparams": hp,
            "n_hist": int(xh.shape[2]), "n_fut": int(xf.shape[2])}
    return model, info


def checkpoint_path(models_dir: Path, name: str, variant: str) -> Path:
    return Path(models_dir) / f"{name}_{variant}.pt"


def save_checkpoint(model, info: Dict, scaler, models_dir: Path, name: str, variant: str) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    p = checkpoint_path(models_dir, name, variant)
    torch.save({"state_dict": model.state_dict(), "info": {k: v for k, v in info.items()},
                "scaler": scaler.to_dict()}, p)
    return p
