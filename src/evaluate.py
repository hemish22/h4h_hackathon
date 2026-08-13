"""Part 6 — evaluation. Every metric named in the committee's Metrics document,
plus QWK and NRMSE as clearly-labelled extras. Emits JSON + markdown table.
"""
from __future__ import annotations
import json
import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, roc_auc_score, confusion_matrix,
                             cohen_kappa_score, classification_report,
                             mean_absolute_error, mean_squared_error,
                             r2_score, explained_variance_score)

from . import config as C


def classification_metrics(y_true, y_pred, y_proba, labels=None):
    labels = labels or C.STRESS_CLASSES
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(C.N_CLASSES), zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_per_class": dict(zip(labels, p.tolist())),
        "recall_per_class": dict(zip(labels, r.tolist())),        # sensitivity
        "f1_per_class": dict(zip(labels, f.tolist())),
        "support_per_class": dict(zip(labels, s.tolist())),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=range(C.N_CLASSES)).tolist(),
        "qwk_EXTRA": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
    # ROC-AUC needs all classes present + normalised proba
    try:
        present = sorted(set(int(v) for v in y_true))
        if len(present) == C.N_CLASSES and y_proba is not None:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"))
            out["roc_auc_ovr_weighted"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"))
        else:
            out["roc_auc_ovr_macro"] = None
            out["roc_auc_note"] = f"only classes {present} present in y_true"
    except Exception as e:
        out["roc_auc_ovr_macro"] = None
        out["roc_auc_note"] = str(e)
    out["report"] = classification_report(y_true, y_pred, labels=range(C.N_CLASSES),
                                          target_names=labels, zero_division=0)
    return out


def regression_metrics(y_true, y_pred, names=None, maxima=None):
    """y_true, y_pred in ORIGINAL units (invert normalisation before calling)."""
    names = names or ["Depression", "Anxiety", "Stress"]
    maxima = maxima or C.REG_MAXIMA
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    out = {}
    for i, n in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        mse = mean_squared_error(t, p)
        out[n] = {
            "MAE": float(mean_absolute_error(t, p)),
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(t, p)),
            "ExplainedVariance": float(explained_variance_score(t, p)),
            "NRMSE_pct_EXTRA": float(100 * np.sqrt(mse) / maxima[i]),
        }
    return out


def save_results(run_id, cls=None, reg=None, extra=None):
    d = os.path.join(C.RESULTS_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    payload = {"classification": cls, "regression": reg, **(extra or {})}
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {d}/metrics.json")
    return payload


def markdown_row(name, cls, reg, pairing="—"):
    """One ablation-table row (plan §6.3)."""
    def g(d, k):
        return f"{d.get(k):.3f}" if d and d.get(k) is not None else "—"
    row = [name, pairing]
    if cls:
        row += [g(cls, "accuracy"), g(cls, "macro_f1"), g(cls, "weighted_f1"),
                g(cls, "roc_auc_ovr_macro"), g(cls, "qwk_EXTRA")]
    else:
        row += ["—"] * 5
    if reg:
        row += [f"{reg['Depression']['MAE']:.2f}", f"{reg['Anxiety']['MAE']:.2f}",
                f"{reg['Stress']['MAE']:.2f}"]
    else:
        row += ["—", "—", "—"]
    return "| " + " | ".join(row) + " |"


ABLATION_HEADER = (
    "| Model | Pairing | Acc | MacroF1 | WeightedF1 | ROC-AUC | QWK | "
    "Dep MAE | Anx MAE | Str MAE |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)
