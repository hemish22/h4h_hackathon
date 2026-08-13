"""Part 9 — Streamlit prototype.

Upload face + wav + 18 sliders -> class + confidence, 3 severity scores with
MC-dropout uncertainty, gate-weight chart, concordance gauge + routing verdict,
report card. Graceful degradation on missing modality. Conflict-aware messaging.
4 one-click demo participants (Healthy / Severe / low-concordance / conflict).

Run: streamlit run app/streamlit_app.py
"""
import json
import os
import sys
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import data_utils as D
from src.uncertainty import concordance, route

st.set_page_config(page_title="Multimodal Mental-Health Screening", layout="wide")
MODEL_PATHS = ["model_final_matched.pt", "model_concat_matched.pt", "model_final_run.pt"]


@st.cache_resource
def load_model():
    import torch
    from src.train import build_model
    for name in MODEL_PATHS:
        p = os.path.join(C.ARTIFACTS_DIR, name)
        if os.path.exists(p):
            m = build_model(dummy=False)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
            return m, name
    return None, None


@st.cache_resource
def load_scaler():
    from src.datasets import load_scaler as ls
    try:
        return ls()
    except Exception:
        return None


@st.cache_data
def dataset_stats():
    csv = D.load_csv()
    med = {c: float(csv[c].median()) for c in C.FEATURE_COLS}
    rng = {c: (float(csv[c].min()), float(csv[c].max())) for c in C.FEATURE_COLS}
    return med, rng


@st.cache_data
def load_demos():
    p = os.path.join(C.DATA_DIR, "demo_participants.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def infer(model, face_arr, mel_arr, tab_vec, drop_face, drop_voice, T=25):
    """Returns mean/std posterior + reg + gate + per-modality aux posteriors."""
    import torch, torch.nn.functional as F
    from src.uncertainty import _enable_mc_dropout
    face = torch.zeros(1, 48, 48) if drop_face else torch.from_numpy(face_arr).unsqueeze(0)
    voice = torch.zeros(1, 128, 188) if drop_voice else torch.from_numpy(mel_arr).unsqueeze(0)
    tab = torch.from_numpy(tab_vec.astype(np.float32)).unsqueeze(0)
    # MC-dropout for uncertainty bands
    _enable_mc_dropout(model)
    cs, rs = [], []
    with torch.no_grad():
        for _ in range(T):
            o = model(face, voice, tab)
            cs.append(F.softmax(o.cls_logits, 1).numpy()[0])
            rs.append(o.reg.numpy()[0])
    model.eval()
    with torch.no_grad():
        o = model(face, voice, tab)
    cs = np.array(cs); rs = np.array(rs) * np.array(C.REG_MAXIMA)
    aux = [F.softmax(a, 1).numpy()[0] for a in o.aux]  # face, voice, tab
    return (cs.mean(0), cs.std(0), rs.mean(0), rs.std(0),
            o.gate.numpy()[0], aux)


def render(proba, pstd, reg, rstd, gate, aux, dropped, conflict_hint=False):
    conc, _ = concordance(aux[0][None], aux[1][None], aux[2][None])
    conc = float(conc[0]); verdict = route(conc)
    ci = int(proba.argmax())
    st.header(f"Assessment: {C.STRESS_CLASSES[ci].replace('_',' ')}  ·  {proba[ci]*100:.0f}% confidence")
    if dropped:
        st.info(f"Graceful degradation: {', '.join(dropped)} missing → embedding zeroed.")
    a, b = st.columns(2)
    with a:
        st.subheader("Severity (MC-dropout ±)")
        for n, v, s, mx in zip(["Depression", "Anxiety", "Stress"], reg, rstd, C.REG_MAXIMA):
            st.metric(n, f"{v:.0f} ± {s:.0f}", help=f"scale 0–{mx}")
        st.subheader("Modality gate weights")
        st.bar_chart({"weight": {l: float(g) for l, g in
                     zip(["Facial", "Voice", "Behav/Phys"], gate)}})
    with b:
        st.subheader("Concordance")
        st.progress(min(max(conc, 0.0), 1.0))
        msg = {"report": "✅ Report with confidence",
               "caveat": "⚠ Report with disagreement caveat",
               "human_review": "🚩 Flag for human review"}[verdict]
        st.write(f"**{conc:.2f}** — {msg}")
        st.caption("Per-modality reads: "
                   + " · ".join(f"{l} {C.STRESS_CLASSES[p.argmax()].replace('_Stress','')}"
                                for l, p in zip(["Face", "Voice", "Tab"], aux)))
    from src.explain import report_card
    rc = report_card("LIVE", proba, pstd, reg, rstd, gate,
                     concordance=conc, conflict=conflict_hint)
    st.subheader("Screening report card")
    st.code(rc)
    st.download_button("Download report", rc, file_name="screening_report.txt")


st.title("🧠 Multimodal Mental-Health Screening")
st.caption("A **screening** tool that flags people who may benefit from professional "
           "assessment. Not a diagnostic instrument.")

model, model_name = load_model()
scaler = load_scaler()
med, rng = dataset_stats()
demos = load_demos()

if model is None:
    st.error("No trained model in artifacts/. Add model_final_matched.pt.")
    st.stop()
st.caption(f"Loaded model: `{model_name}`")

# ---- demo participants
if demos:
    st.subheader("Demo participants (one click)")
    cols = st.columns(len(demos))
    for col, (key, d) in zip(cols, demos.items()):
        if col.button(d["tag"], key=f"demo_{key}"):
            st.session_state["demo"] = d

# ---- manual inputs
st.subheader("Or provide inputs")
c1, c2 = st.columns(2)
up_img = c1.file_uploader("Face image", type=["png", "jpg", "jpeg"])
up_wav = c2.file_uploader("Speech .wav", type=["wav"])
with st.expander("Tabular features (dataset medians)"):
    tcols = st.columns(3)
    tab_vals = {}
    for i, f in enumerate(C.FEATURE_COLS):
        lo, hi = rng[f]
        tab_vals[f] = tcols[i % 3].slider(f, lo, hi, med[f])

go = st.button("Predict", type="primary")

# ---- demo path
if "demo" in st.session_state and (go is False):
    d = st.session_state.pop("demo")
    from src.datasets import load_face, load_logmel
    face = load_face(d["image_path"]); mel = load_logmel(d["audio_path"])
    tv = scaler.transform(np.array([[d["features"][f] for f in C.FEATURE_COLS]]))[0]
    st.info(f"**{d['tag']}** — participant #{d['participant_id']} "
            f"(true: {d['true_class'].replace('_',' ')})")
    proba, pstd, reg, rstd, gate, aux = infer(model, face, mel, tv, False, False)
    render(proba, pstd, reg, rstd, gate, aux, [], conflict_hint=d["conflict"])

# ---- manual path
elif go:
    drop_face, drop_voice = up_img is None, up_wav is None
    face = mel = None
    if not drop_face:
        from PIL import Image
        img = Image.open(up_img).convert("L").resize((48, 48))
        face = ((np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5).astype(np.float32)
    if not drop_voice:
        from src.datasets import load_logmel
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(up_wav.read()); tmp.close()
        mel = load_logmel(tmp.name)
    tv = scaler.transform(np.array([[tab_vals[f] for f in C.FEATURE_COLS]]))[0]
    dropped = [m for m, dd in [("face", drop_face), ("voice", drop_voice)] if dd]
    proba, pstd, reg, rstd, gate, aux = infer(model, face, mel, tv, drop_face, drop_voice)
    render(proba, pstd, reg, rstd, gate, aux, dropped)

st.divider()
st.caption("Synthetic multimodal pairing · acted emotions · not clinically validated · "
           "decision support only.")
