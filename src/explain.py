"""Part 5.3-5.4 + 7 — explainability: report card, counterfactuals, SHAP, Grad-CAM.

The report card is template formatting over numbers already computed elsewhere
(class posterior, MC-dropout std, gate weights, counterfactual re-predictions).
"""
from __future__ import annotations
import numpy as np

from . import config as C

# actionable features a person can change (plan §5.4)
ACTIONABLE = ["Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min",
              "Idle_Time_Min", "Session_Frequency", "Typing_Speed_WPM"]


def counterfactual(predict_fn, row_feats, feature, new_value):
    """predict_fn: dict[feat]->value -> stress_score scalar. Returns delta."""
    base = predict_fn(row_feats)
    mod = dict(row_feats); mod[feature] = new_value
    return predict_fn(mod) - base


def top_counterfactuals(predict_fn, row_feats, csv_df, k=3):
    """Sweep actionable features across their IQR, rank by |effect on stress|."""
    results = []
    for feat in ACTIONABLE:
        q1, q3 = csv_df[feat].quantile([0.25, 0.75])
        for target in (q1, q3):
            d = counterfactual(predict_fn, row_feats, feat, float(target))
            results.append((feat, float(target), float(d)))
    results.sort(key=lambda x: -abs(x[2]))
    # keep best (most negative) per feature
    seen, out = set(), []
    for feat, val, d in results:
        if feat in seen:
            continue
        seen.add(feat); out.append((feat, val, d))
        if len(out) == k:
            break
    return out


def report_card(pid, cls_proba, cls_std, reg_mean, reg_std, gate,
                drivers=None, concordance=None, conflict=False, counterfactuals=None):
    """Plain-language screening report card (string)."""
    ci = int(np.argmax(cls_proba))
    conf = float(cls_proba[ci])
    names = ["Depression", "Anxiety", "Stress"]
    L = []
    W = 55
    L.append("─" * W)
    L.append(f"PARTICIPANT #{pid}".ljust(38) + "Screening report")
    L.append("─" * W)
    L.append(f"ASSESSMENT      {C.STRESS_CLASSES[ci].replace('_', ' ')} indicated")
    band = "high" if conf > 0.75 else ("moderate" if conf > 0.5 else "low")
    L.append(f"CONFIDENCE      {conf*100:.0f}%  ({band})")
    L.append("")
    L.append("SEVERITY        " + f"Depression  {reg_mean[0]:.0f} ± {reg_std[0]:.0f}   (0–34)")
    L.append("                " + f"Anxiety     {reg_mean[1]:.0f} ± {reg_std[1]:.0f}   (0–24)")
    L.append("                " + f"Stress      {reg_mean[2]:.0f} ± {reg_std[2]:.0f}   (0–39)")
    L.append("")
    if drivers:
        L.append("PRIMARY DRIVERS")
        for name, note in drivers:
            L.append(f"  {name:<28} {note}")
        L.append("")
    L.append("MODALITY EVIDENCE")
    for lbl, g in zip(["Facial", "Voice", "Physiological/behavioural"], gate):
        L.append(f"  {lbl:<28} {g*100:.0f}%")
    L.append("")
    if concordance is not None:
        flag = "LOW" if concordance < 0.45 else ("moderate" if concordance < 0.75 else "OK")
        L.append(f"CONCORDANCE     {concordance:.2f}  {flag}")
        if conflict:
            L.append("                Modality disagreement present; note this")
            L.append("                severity class has known annotation")
            L.append("                inconsistency (angry/disgust mapping).")
        elif concordance < 0.45:
            L.append("                Facial signal diverges from voice and")
            L.append("                physiology — pattern consistent with masked")
            L.append("                affect. Recommend human review.")
        L.append("")
    if counterfactuals:
        L.append("WHAT WOULD CHANGE THIS")
        for feat, val, d in counterfactuals:
            L.append(f"  {feat} → {val:.0f}".ljust(32) + f"predicted stress {d:+.1f}")
    L.append("─" * W)
    return "\n".join(L)


# ------------------------------------------------------------------ SHAP / Grad-CAM
def shap_beeswarm(lgbm_path=None, out=None):
    import joblib, shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import data_utils as D
    import os
    lgbm_path = lgbm_path or os.path.join(C.ARTIFACTS_DIR, "lgbm.joblib")
    model = joblib.load(lgbm_path)["clf"]
    X = D.load_csv()[C.FEATURE_COLS]
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X.sample(min(500, len(X)), random_state=C.SEED))
    shap.summary_plot(sv, X.sample(min(500, len(X)), random_state=C.SEED),
                      show=False, plot_type="bar")
    out = out or os.path.join(C.FIGURES_DIR, "shap_beeswarm.png")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"wrote {out}")


def grad_cam(model_conv, x, target_layer, device="cpu"):
    """Grad-CAM on a conv encoder. x: (1,1,H,W). Returns heatmap (H,W) in [0,1]."""
    import torch
    acts, grads = {}, {}

    def fwd(m, i, o): acts["v"] = o.detach()
    def bwd(m, gi, go): grads["v"] = go[0].detach()
    h1 = target_layer.register_forward_hook(fwd)
    h2 = target_layer.register_full_backward_hook(bwd)
    x = x.to(device).requires_grad_(True)
    z = model_conv(x)
    score = z.max()
    model_conv.zero_grad(); score.backward()
    h1.remove(); h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["v"]).sum(1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
    return cam
