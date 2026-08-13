"""Part 4.9 — decision-level fusion baseline. NO pairing anywhere in training.

Each modality produces a 4-class stress posterior from its own data + the
committee's per-modality emotion->stress mapping. Combine at decision level.
This answers "did you need the alignment engine?" with a number.
"""
from __future__ import annotations
import numpy as np
from itertools import product


def decision_level_fusion(p_face, p_voice, p_tab, w=None):
    """Each p_* is (N,4), rows sum to 1. Returns fused (N,4) posterior."""
    w = np.array([1 / 3, 1 / 3, 1 / 3]) if w is None else np.asarray(w, dtype=float)
    P = w[0] * p_face + w[1] * p_voice + w[2] * p_tab
    return P / P.sum(1, keepdims=True)


def tune_weights(p_face, p_voice, p_tab, y_true, res=0.05):
    """Grid-search weights on the simplex to maximise accuracy."""
    best_w, best_acc = None, -1.0
    steps = np.arange(0, 1 + 1e-9, res)
    for a, b in product(steps, steps):
        if a + b > 1 + 1e-9:
            continue
        c = 1 - a - b
        P = decision_level_fusion(p_face, p_voice, p_tab, [a, b, c])
        acc = (P.argmax(1) == y_true).mean()
        if acc > best_acc:
            best_acc, best_w = acc, [a, b, c]
    return np.array(best_w), best_acc


def stacked_lr(p_face, p_voice, p_tab, y_true):
    """Stacked logistic regression on the concatenated 12-d posterior."""
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([p_face, p_voice, p_tab], axis=1)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X, y_true)
    return clf
