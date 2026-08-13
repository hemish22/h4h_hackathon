"""Part 5.1-5.2 — MC-dropout uncertainty + modality concordance (with conflict segmentation)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _enable_mc_dropout(model):
    """Dropout ON, BatchNorm forced to eval (else BN stats corrupt MC passes)."""
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()


def predict_with_uncertainty(model, batch, T=30, device="cpu"):
    _enable_mc_dropout(model)
    face, voice, tab = batch["face"].to(device), batch["voice"].to(device), batch["tab"].to(device)
    cls_s, reg_s = [], []
    with torch.no_grad():
        for _ in range(T):
            o = model(face, voice, tab)
            cls_s.append(F.softmax(o.cls_logits, 1))
            reg_s.append(o.reg)
    model.eval()
    cls = torch.stack(cls_s); reg = torch.stack(reg_s)
    return (cls.mean(0).cpu().numpy(), cls.std(0).cpu().numpy(),
            reg.mean(0).cpu().numpy(), reg.std(0).cpu().numpy())


# ------------------------------------------------------------------ concordance
def concordance(p_face, p_voice, p_tab):
    """1.0 = perfect agreement, 0.0 = 3 stress levels apart.
    Each p_* is (N,4). Returns (score (N,), spread (N,))."""
    lev = np.arange(p_face.shape[1])
    e = [(p * lev).sum(-1) for p in (p_face, p_voice, p_tab)]
    e = np.stack(e, 0)                                     # (3,N)
    spread = e.max(0) - e.min(0)
    return 1.0 - spread / (p_face.shape[1] - 1), spread


ROUTING_BANDS = [(0.75, 1.01, "report"), (0.45, 0.75, "caveat"), (-0.01, 0.45, "human_review")]


def route(score):
    for lo, hi, action in ROUTING_BANDS:
        if lo <= score < hi:
            return action
    return "human_review"


def concordance_report(scores, y_true, y_pred, mask=None):
    """Fraction per band + accuracy per band. mask selects a subset (e.g. clean)."""
    if mask is not None:
        scores, y_true, y_pred = scores[mask], y_true[mask], y_pred[mask]
    out = {}
    for lo, hi, action in ROUTING_BANDS:
        band = (scores >= lo) & (scores < hi)
        n = int(band.sum())
        acc = float((y_true[band] == y_pred[band]).mean()) if n else None
        out[action] = {"fraction": n / max(len(scores), 1), "n": n, "accuracy": acc}
    out["mean_concordance"] = float(scores.mean()) if len(scores) else None
    return out
