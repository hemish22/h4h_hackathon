"""SHAP tabular importance — MUST run in a process that does NOT import torch.
torch's libomp + lightgbm's libomp duplicate-load segfaults. Kept standalone and
invoked as a subprocess by src.run_explain.
Run: python -m src.run_shap
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import data_utils as D


def run(out=None):
    import joblib, shap
    model = joblib.load(os.path.join(C.ARTIFACTS_DIR, "lgbm.joblib"))["clf"]
    X = D.load_csv()[C.FEATURE_COLS].sample(min(500, 4000), random_state=C.SEED)
    expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = np.asarray(expl.shap_values(X, check_additivity=False))   # (N,F,n_classes)
    imp = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
    order = np.argsort(imp)
    out = out or os.path.join(C.FIGURES_DIR, "shap_beeswarm.png")
    plt.figure(figsize=(6, 6))
    plt.barh([C.FEATURE_COLS[i] for i in order], imp[order], color="#4C78A8")
    plt.xlabel("mean |SHAP value| (impact on stress class)")
    plt.title("TreeSHAP feature importance — tabular (LightGBM)")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
