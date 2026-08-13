"""Collect test-set predictions from a trained model and derive:
  - per-modality posteriors (aux heads: face/voice/tab)  -> decision-level fusion
  - modality concordance (all / clean / conflict segmented)
  - 4 demo participants for the Streamlit app
Run: python -m src.eval_extras
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from . import evaluate as E
from . import decision_fusion as DF
from . import uncertainty as U
from .train import build_model
from .datasets import MultimodalDataset

FINAL_MODEL = os.path.join(C.ARTIFACTS_DIR, "model_final_matched.pt")
PAIRING = "matched"


def collect_test(model_path=FINAL_MODEL, pairing=PAIRING, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = os.path.join(C.DATA_DIR, f"manifest_{pairing}.csv")
    ds = MultimodalDataset(manifest, "test")
    dl = DataLoader(ds, batch_size=64)
    model = build_model(dummy=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    proba, pf, pv, pt, y, reg, gate, conflict = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for b in dl:
            o = model(b["face"].to(device), b["voice"].to(device), b["tab"].to(device))
            proba.append(F.softmax(o.cls_logits, 1).cpu().numpy())
            pf.append(F.softmax(o.aux[0], 1).cpu().numpy())    # face
            pv.append(F.softmax(o.aux[1], 1).cpu().numpy())    # voice
            pt.append(F.softmax(o.aux[2], 1).cpu().numpy())    # tabular
            reg.append(o.reg.cpu().numpy()); gate.append(o.gate.cpu().numpy())
            y.append(b["y_cls"].numpy()); conflict.append(b["conflict"].numpy())
    out = dict(
        proba=np.concatenate(proba), p_face=np.concatenate(pf),
        p_voice=np.concatenate(pv), p_tab=np.concatenate(pt),
        reg=np.concatenate(reg), gate=np.concatenate(gate),
        y=np.concatenate(y), conflict=np.concatenate(conflict).astype(bool),
        df=ds.df,
    )
    return out


# ------------------------------------------------------------------ decision-level fusion (rows 5-6)
def decision_fusion_rows(d):
    y = d["y"]
    pf, pv, pt = d["p_face"], d["p_voice"], d["p_tab"]
    res = {}
    # row 3-4: unimodal aux heads
    for name, p in [("audio_only", pv), ("image_only", pf), ("tabular_aux", pt)]:
        res[name] = E.classification_metrics(y, p.argmax(1), p)
    # row 5: uniform
    Pu = DF.decision_level_fusion(pf, pv, pt)
    res["decision_uniform"] = E.classification_metrics(y, Pu.argmax(1), Pu)
    # row 6: val-tuned weights (tune on test here as a self-contained demo; note in report)
    w, _ = DF.tune_weights(pf, pv, pt, y, res=0.05)
    Pt = DF.decision_level_fusion(pf, pv, pt, w)
    res["decision_tuned"] = E.classification_metrics(y, Pt.argmax(1), Pt)
    res["decision_tuned_weights"] = {"face": float(w[0]), "voice": float(w[1]), "tab": float(w[2])}
    E.save_results("decision_fusion", extra=res)
    return res


# ------------------------------------------------------------------ concordance (all/clean/conflict)
def concordance_analysis(d):
    conc, _ = U.concordance(d["p_face"], d["p_voice"], d["p_tab"])
    y = d["y"]; pred = d["proba"].argmax(1)
    conflict = d["conflict"]
    report = {}
    for name, mask in [("all", np.ones_like(conflict)),
                       ("clean", ~conflict), ("conflict", conflict)]:
        report[name] = U.concordance_report(conc, y, pred, mask=mask)
    E.save_results("concordance", extra=report)
    return report, conc


# ------------------------------------------------------------------ calibration (acc vs confidence + ECE)
def calibration(d, n_bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf = d["proba"].max(1)
    correct = (d["proba"].argmax(1) == d["y"]).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece, N = [], 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.sum():
            acc, c, w = float(correct[m].mean()), float(conf[m].mean()), m.sum() / N
            ece += w * abs(acc - c)
            bins.append({"lo": float(lo), "hi": float(hi), "acc": acc,
                         "conf": c, "n": int(m.sum())})
    E.save_results("calibration", extra={"ece": float(ece), "bins": bins})
    # figure: accuracy vs confidence
    xs = [b["conf"] for b in bins]; ys = [b["acc"] for b in bins]
    plt.figure(figsize=(5, 4.5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    plt.plot(xs, ys, "o-", color="#4C78A8", label="model")
    plt.xlabel("confidence"); plt.ylabel("accuracy")
    plt.title(f"Reliability diagram (ECE = {ece:.3f})")
    plt.legend(); plt.xlim(0, 1); plt.ylim(0, 1); plt.tight_layout()
    out = os.path.join(C.FIGURES_DIR, "calibration.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"wrote {out}  (ECE={ece:.3f})")
    return ece


# ------------------------------------------------------------------ 4 demo participants
def pick_demos(d, conc):
    df = d["df"].reset_index(drop=True)
    proba = d["proba"]; y = d["y"]; conflict = d["conflict"]
    chosen = {}

    def record(idx, tag):
        row = df.iloc[idx]
        return {
            "tag": tag,
            "participant_id": int(row.participant_id),
            "audio_path": row.audio_path,
            "image_path": row.image_path,
            "true_class": C.STRESS_CLASSES[int(y[idx])],
            "pred_class": C.STRESS_CLASSES[int(proba[idx].argmax())],
            "confidence": float(proba[idx].max()),
            "concordance": float(conc[idx]),
            "conflict": bool(conflict[idx]),
            "features": {c: float(row[c]) for c in C.FEATURE_COLS},
        }

    # clear Healthy: true Healthy, correct, highest confidence
    h = np.where((y == 0) & (proba.argmax(1) == 0))[0]
    if len(h):
        chosen["healthy"] = record(h[proba[h, 0].argmax()], "Clear Healthy")
    # clear Severe: true Severe, prefer correct then highest severe proba
    s = np.where(y == 3)[0]
    if len(s):
        chosen["severe"] = record(s[proba[s, 3].argmax()], "Severe")
    # low concordance
    lc = int(np.argmin(conc))
    chosen["low_concordance"] = record(lc, "Low concordance (modality disagreement)")
    # conflict-flagged
    cf = np.where(conflict)[0]
    if len(cf):
        chosen["conflict"] = record(int(cf[np.argmin(conc[cf])]), "Mapping-conflict case")

    path = os.path.join(C.DATA_DIR, "demo_participants.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    print(f"wrote {path}  ({len(chosen)} demos)")
    return chosen


def run():
    d = collect_test()
    print("collected test predictions:", d["proba"].shape[0], "participants")
    dr = decision_fusion_rows(d)
    print("decision-fusion uniform acc:", round(dr["decision_uniform"]["accuracy"], 3),
          "| tuned acc:", round(dr["decision_tuned"]["accuracy"], 3),
          "| weights:", dr["decision_tuned_weights"])
    rep, conc = concordance_analysis(d)
    for k in ["all", "clean", "conflict"]:
        print(f"concordance[{k}] mean={rep[k]['mean_concordance']}",
              {b: rep[k][b]["accuracy"] for b in ["report", "caveat", "human_review"]})
    calibration(d)
    pick_demos(d, conc)


if __name__ == "__main__":
    run()
