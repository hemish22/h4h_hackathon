"""Part 4.3-4.5 — ordinal-aware classification loss + joint multi-task loss."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from . import config as C


def ordinal_ce(logits, y, class_w=None, beta=0.3):
    """Cross-entropy + expectation regulariser (plan Tier 1).

    Penalises the distance between expected ordinal level and the true level,
    so predicting Healthy for a Severe case costs more than predicting Moderate.
    Keeps a valid softmax -> ROC-AUC / confusion matrix stay computable.
    """
    p = F.softmax(logits, dim=1)
    lev = torch.arange(C.N_CLASSES, device=logits.device, dtype=p.dtype)
    exp = (p * lev).sum(1)
    ce = F.cross_entropy(logits, y, weight=class_w)
    return ce + beta * F.l1_loss(exp, y.to(p.dtype))


def huber_reg(pred, target):
    """SmoothL1 over the 3 normalised regression targets."""
    return F.smooth_l1_loss(pred, target)


def joint_loss(out, y_cls, y_reg, class_w=None, lambda_reg=0.5, aux_w=0.3, beta=0.3):
    """out: object with .cls_logits, .reg, .aux (list of per-modality logits)."""
    L = ordinal_ce(out.cls_logits, y_cls, class_w, beta)
    L = L + lambda_reg * huber_reg(out.reg, y_reg)
    if getattr(out, "aux", None):
        L = L + aux_w * sum(ordinal_ce(a, y_cls, class_w, beta) for a in out.aux)
    return L


def class_weights(counts, asym=1.5):
    """Inverse-frequency weights; Severe & Moderate scaled by `asym` (asymmetric
    cost — under-calling severity should hurt more)."""
    counts = torch.as_tensor(counts, dtype=torch.float32)
    w = counts.sum() / (counts.clamp(min=1) * len(counts))
    w[C.CLASS_TO_ORD["Moderate_Stress"]] *= asym
    w[C.CLASS_TO_ORD["Severe_Stress"]] *= asym
    return w
