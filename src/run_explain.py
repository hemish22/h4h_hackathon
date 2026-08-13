"""Part 7 — generate explainability figures:
  figures/shap_beeswarm.png      TreeSHAP on the LightGBM tabular model
  figures/gradcam_faces.png      Grad-CAM over 4 faces (one per stress class)
  figures/gradcam_spectrograms.png  Grad-CAM over 4 mel-spectrograms
Run: python -m src.run_explain
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .train import build_model
from .datasets import MultimodalDataset

FINAL = os.path.join(C.ARTIFACTS_DIR, "model_final_matched.pt")


# ------------------------------------------------------------------ SHAP (tabular)
def shap_beeswarm(out=None):
    """Run SHAP in a torch-free subprocess (torch+lightgbm libomp clash segfaults)."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "src.run_shap"],
                       cwd=C.ROOT, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-300:])


# ------------------------------------------------------------------ Grad-CAM helpers
def _last_conv(module):
    conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
    return conv


def _gradcam(model, batch, modality, target_layer):
    """CAM on `target_layer` w.r.t. the predicted stress logit. modality in
    {'face','voice'}. batch tensors are (1,...)."""
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    model.zero_grad()
    o = model(batch["face"], batch["voice"], batch["tab"])
    pred = int(o.cls_logits.argmax(1))
    o.cls_logits[0, pred].backward()
    h1.remove(); h2.remove()
    A = acts["v"][0]; G = grads["v"][0]                 # (Cc,h,w)
    w = G.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * A).sum(0)).detach().cpu().numpy()
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    return cam, pred


def _pick_one_per_class(ds):
    idx = {}
    for i in range(len(ds)):
        c = int(ds.df.iloc[i].stress_ordinal)
        if c not in idx:
            idx[c] = i
        if len(idx) == C.N_CLASSES:
            break
    return [idx[c] for c in range(C.N_CLASSES) if c in idx]


def gradcam_faces(model, ds, out=None):
    from scipy.ndimage import zoom
    layer = _last_conv(model.face)
    picks = _pick_one_per_class(ds)
    fig, ax = plt.subplots(1, len(picks), figsize=(3 * len(picks), 3.2))
    if len(picks) == 1:
        ax = [ax]
    for a, i in zip(ax, picks):
        b = ds[i]
        batch = {k: b[k].unsqueeze(0) for k in ["face", "voice", "tab"]}
        cam, pred = _gradcam(model, batch, "face", layer)
        img = b["face"].numpy()
        cam_up = zoom(cam, (img.shape[0] / cam.shape[0], img.shape[1] / cam.shape[1]), order=1)
        a.imshow(img, cmap="gray")
        a.imshow(cam_up, cmap="jet", alpha=0.45)
        a.set_title(f"true {C.STRESS_CLASSES[int(b['y_cls'])].replace('_Stress','')}\n"
                    f"pred {C.STRESS_CLASSES[pred].replace('_Stress','')}", fontsize=9)
        a.axis("off")
    plt.suptitle("Grad-CAM — face encoder")
    out = out or os.path.join(C.FIGURES_DIR, "gradcam_faces.png")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"wrote {out}")


def gradcam_spectrograms(model, ds, out=None):
    from scipy.ndimage import zoom
    layer = _last_conv(model.voice)
    picks = _pick_one_per_class(ds)
    fig, ax = plt.subplots(len(picks), 1, figsize=(6, 2.2 * len(picks)))
    if len(picks) == 1:
        ax = [ax]
    for a, i in zip(ax, picks):
        b = ds[i]
        batch = {k: b[k].unsqueeze(0) for k in ["face", "voice", "tab"]}
        cam, pred = _gradcam(model, batch, "voice", layer)
        mel = b["voice"].numpy()
        cam_up = zoom(cam, (mel.shape[0] / cam.shape[0], mel.shape[1] / cam.shape[1]), order=1)
        a.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        a.imshow(cam_up, origin="lower", aspect="auto", cmap="jet", alpha=0.4)
        a.set_ylabel(C.STRESS_CLASSES[int(b['y_cls'])].replace("_Stress", ""), fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    ax[0].set_title("Grad-CAM — voice encoder (mel-spectrogram, time →)")
    out = out or os.path.join(C.FIGURES_DIR, "gradcam_spectrograms.png")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"wrote {out}")


def run():
    shap_beeswarm()
    model = build_model(dummy=False)
    model.load_state_dict(torch.load(FINAL, map_location="cpu"))
    model.eval()
    ds = MultimodalDataset(os.path.join(C.DATA_DIR, "manifest_matched.csv"), "test")
    gradcam_faces(model, ds)
    gradcam_spectrograms(model, ds)


if __name__ == "__main__":
    run()
