"""Synthetic smoke test — validates model/loss/fusion/metrics wiring with random
tensors, no dataset or librosa needed. Run: python -m tests.test_smoke
"""
import numpy as np
import torch

from src import config as C
from src import losses as L
from src import evaluate as E
from src.train import build_model
from src.decision_fusion import decision_level_fusion, tune_weights
from src.uncertainty import concordance, concordance_report


def test_model_forward_backward():
    for dummy in (True, False):
        model = build_model(dummy=dummy, modality_dropout=0.15)
        B = 8
        face = torch.randn(B, 48, 48)
        voice = torch.randn(B, 128, 188)
        tab = torch.randn(B, 18)
        y_cls = torch.randint(0, 4, (B,))
        y_reg = torch.rand(B, 3)
        cw = L.class_weights([100, 80, 60, 10])
        o = model(face, voice, tab)
        assert o.cls_logits.shape == (B, 4)
        assert o.reg.shape == (B, 3)
        assert o.gate.shape == (B, 3)
        assert len(o.aux) == 3
        loss = L.joint_loss(o, y_cls, y_reg, class_w=cw)
        loss.backward()
        assert torch.isfinite(loss), "loss not finite"
    print("OK model forward/backward (dummy + real encoders)")


def test_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    proba = rng.dirichlet(np.ones(4), 200)
    pred = proba.argmax(1)
    cls = E.classification_metrics(y, pred, proba)
    assert 0 <= cls["accuracy"] <= 1
    assert "qwk_EXTRA" in cls and "macro_f1" in cls
    yr = rng.uniform(0, [34, 24, 39], (200, 3))
    pr = yr + rng.normal(0, 2, (200, 3))
    reg = E.regression_metrics(yr, pr)
    assert set(reg) == {"Depression", "Anxiety", "Stress"}
    assert all("RMSE" in reg[k] and "R2" in reg[k] for k in reg)
    print("OK metrics (all required classification + regression fields present)")


def test_decision_fusion_and_concordance():
    rng = np.random.default_rng(1)
    N = 100
    pf = rng.dirichlet(np.ones(4), N)
    pv = rng.dirichlet(np.ones(4), N)
    pt = rng.dirichlet(np.ones(4), N)
    y = rng.integers(0, 4, N)
    P = decision_level_fusion(pf, pv, pt)
    assert np.allclose(P.sum(1), 1)
    w, acc = tune_weights(pf, pv, pt, y, res=0.25)
    assert abs(w.sum() - 1) < 1e-6
    conc, spread = concordance(pf, pv, pt)
    assert conc.shape == (N,) and (conc <= 1).all() and (conc >= 0).all()
    rep = concordance_report(conc, y, P.argmax(1))
    assert "mean_concordance" in rep
    print("OK decision fusion + concordance + routing")


if __name__ == "__main__":
    test_model_forward_backward()
    test_metrics()
    test_decision_fusion_and_concordance()
    print("\nALL SMOKE TESTS PASSED")
