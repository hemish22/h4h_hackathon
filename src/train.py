"""Part 4.8 — joint multi-task training loop.

Config-driven so one flag toggles the whole ablation table:
  pairing = random | matched   ->  which manifest
  ordinal, modality_dropout, aux ->  architecture/loss switches
Run: python -m src.train --config configs/final.yaml
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import config as C
from . import encoders as ENC
from . import losses as L
from . import evaluate as E
from .fusion import MultiTaskModel
from .datasets import MultimodalDataset, fit_scaler
import pandas as pd


def build_model(dummy=False, modality_dropout=0.15):
    if dummy:
        face = ENC.DummyEncoder(128); voice = ENC.DummyEncoder(128); tab = ENC.DummyEncoder(64)
    else:
        face = ENC.FaceCNN(); voice = ENC.AudioCNN(); tab = ENC.TabularMLP()
    return MultiTaskModel(face, voice, tab, modality_dropout=modality_dropout)


def _class_counts(manifest_path):
    df = pd.read_csv(manifest_path)
    tr = df[df.split == "train"]
    return [int((tr.stress_ordinal == i).sum()) for i in range(C.N_CLASSES)]


def evaluate_model(model, loader, device, invert_reg=True):
    model.eval()
    ys, ps, proba, yreg, preg, gates = [], [], [], [], [], []
    import torch.nn.functional as F
    with torch.no_grad():
        for b in loader:
            o = model(b["face"].to(device), b["voice"].to(device), b["tab"].to(device))
            sm = F.softmax(o.cls_logits, 1).cpu().numpy()
            proba.append(sm); ps.append(sm.argmax(1)); ys.append(b["y_cls"].numpy())
            preg.append(o.reg.cpu().numpy()); yreg.append(b["y_reg"].numpy())
            gates.append(o.gate.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps); pr = np.concatenate(proba)
    yr = np.concatenate(yreg); prr = np.concatenate(preg)
    maxima = np.array(C.REG_MAXIMA)
    if invert_reg:
        yr = yr * maxima; prr = prr * maxima
    cls_m = E.classification_metrics(y, p, pr)
    reg_m = E.regression_metrics(yr, prr)
    return cls_m, reg_m, np.concatenate(gates)


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", C.SEED)); np.random.seed(cfg.get("seed", C.SEED))
    manifest = os.path.join(C.DATA_DIR, f"manifest_{cfg['pairing']}.csv")

    fit_scaler(pd.read_csv(manifest))
    tr = MultimodalDataset(manifest, "train", augment=cfg.get("augment", True))
    va = MultimodalDataset(manifest, "val")
    bs = cfg.get("batch", 64)
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=cfg.get("workers", 2))
    dl_va = DataLoader(va, batch_size=bs, num_workers=cfg.get("workers", 2))

    model = build_model(dummy=cfg.get("dummy", False),
                        modality_dropout=cfg.get("modality_dropout", 0.15)).to(device)
    cw = L.class_weights(_class_counts(manifest),
                         asym=cfg.get("asym", 1.5)).to(device) if cfg.get("class_weight", True) else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=1e-4)
    epochs = cfg.get("epochs", 30)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    beta = cfg.get("beta", 0.3) if cfg.get("ordinal", True) else 0.0
    lam = cfg.get("lambda_reg", 0.5)

    best_f1, best_state, patience, wait = -1, None, cfg.get("patience", 7), 0
    for ep in range(epochs):
        model.train(); tot = 0.0
        for b in dl_tr:
            opt.zero_grad()
            o = model(b["face"].to(device), b["voice"].to(device), b["tab"].to(device))
            loss = L.joint_loss(o, b["y_cls"].to(device), b["y_reg"].to(device),
                                class_w=cw, lambda_reg=lam,
                                aux_w=cfg.get("aux_w", 0.3), beta=beta)
            loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        cls_m, _, _ = evaluate_model(model, dl_va, device)
        f1 = cls_m["macro_f1"]
        print(f"ep {ep+1}/{epochs} loss={tot/len(dl_tr):.4f} val_macroF1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state, wait = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print("early stop"); break

    if best_state:
        model.load_state_dict(best_state)
    run_id = cfg.get("run_id", f"{cfg['pairing']}_run")
    torch.save(model.state_dict(), os.path.join(C.ARTIFACTS_DIR, f"model_{run_id}.pt"))

    # final test eval
    te = MultimodalDataset(manifest, "test")
    dl_te = DataLoader(te, batch_size=bs)
    cls_m, reg_m, gates = evaluate_model(model, dl_te, device)
    E.save_results(run_id, cls_m, reg_m, extra={"mean_gate": gates.mean(0).tolist()})
    print("\nTEST:", E.ABLATION_HEADER)
    print(E.markdown_row(run_id, cls_m, reg_m, cfg["pairing"]))
    return model, cls_m, reg_m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    if a.run_id is not None:
        cfg["run_id"] = a.run_id
    if a.workers is not None:
        cfg["workers"] = a.workers
    train(cfg)
