"""Part 9 — Streamlit prototype.

Upload face + wav + 18 sliders -> class + confidence, 3 severity scores with
uncertainty, gate-weight chart, concordance gauge + routing verdict, report card.
Graceful degradation: missing modality -> zeroed embedding, noted in output.
Conflict-aware messaging via the 7-class face emotion head.

Run: streamlit run app/streamlit_app.py
"""
import os
import sys
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import data_utils as D

st.set_page_config(page_title="Multimodal Mental-Health Screening", layout="wide")

MODEL_PATHS = ["model_final_matched.pt", "model_final_run.pt", "model_matched_run.pt"]


@st.cache_resource
def load_assets():
    import torch, joblib
    from src.train import build_model
    from src.datasets import load_scaler
    model, scaler = None, None
    for name in MODEL_PATHS:
        p = os.path.join(C.ARTIFACTS_DIR, name)
        if os.path.exists(p):
            model = build_model(dummy=False)
            model.load_state_dict(torch.load(p, map_location="cpu"))
            model.eval()
            break
    try:
        scaler = load_scaler()
    except Exception:
        scaler = None
    return model, scaler


@st.cache_data
def dataset_medians():
    csv = D.load_csv()
    return {c: float(csv[c].median()) for c in C.FEATURE_COLS}, csv


def infer(model, scaler, face_arr, mel_arr, tab_vec, drop_face, drop_voice):
    import torch, torch.nn.functional as F
    face = torch.zeros(1, 48, 48) if drop_face else torch.from_numpy(face_arr).unsqueeze(0)
    voice = torch.zeros(1, 128, 188) if drop_voice else torch.from_numpy(mel_arr).unsqueeze(0)
    tab = torch.from_numpy(tab_vec.astype(np.float32)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        o = model(face, voice, tab)
        proba = F.softmax(o.cls_logits, 1).numpy()[0]
        reg = o.reg.numpy()[0] * np.array(C.REG_MAXIMA)
        gate = o.gate.numpy()[0]
        aux = [F.softmax(a, 1).numpy()[0] for a in o.aux]  # face, voice, tab posteriors
    return proba, reg, gate, aux


st.title("🧠 Multimodal Mental-Health Screening")
st.caption("A **screening** tool that flags people who may benefit from professional "
           "assessment. Not a diagnostic instrument.")

model, scaler = load_assets()
medians, csv = dataset_medians()

if model is None:
    st.warning("No trained model found in artifacts/. Run training on the GPU machine "
               "first (`python -m src.train --config configs/final.yaml`). "
               "UI shown for layout; predictions disabled.")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Face")
    up_img = st.file_uploader("Upload face image", type=["png", "jpg", "jpeg"])
with col2:
    st.subheader("Voice")
    up_wav = st.file_uploader("Upload speech .wav", type=["wav"])
with col3:
    st.subheader("Behavioural / physiological")
    st.caption("18 features (pre-filled with dataset medians)")

with st.expander("Tabular features", expanded=False):
    tab_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(C.FEATURE_COLS):
        with cols[i % 3]:
            lo, hi = float(csv[feat].min()), float(csv[feat].max())
            tab_vals[feat] = st.slider(feat, lo, hi, medians[feat])

if st.button("Predict", type="primary", disabled=(model is None)):
    drop_face = up_img is None
    drop_voice = up_wav is None
    face_arr = mel_arr = None
    if not drop_face:
        from src.datasets import load_face
        import tempfile
        from PIL import Image
        img = Image.open(up_img).convert("L").resize((48, 48))
        face_arr = ((np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5).astype(np.float32)
    if not drop_voice:
        from src.datasets import load_logmel
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(up_wav.read()); tmp.close()
        mel_arr = load_logmel(tmp.name)

    tab_vec = scaler.transform(np.array([[tab_vals[f] for f in C.FEATURE_COLS]]))[0]
    proba, reg, gate, aux = infer(model, scaler, face_arr, mel_arr, tab_vec,
                                  drop_face, drop_voice)

    # concordance from aux posteriors
    from src.uncertainty import concordance, route
    conc, _ = concordance(aux[0][None], aux[1][None], aux[2][None])
    conc = float(conc[0]); verdict = route(conc)

    ci = int(proba.argmax())
    st.header(f"Assessment: {C.STRESS_CLASSES[ci].replace('_',' ')}  ·  {proba[ci]*100:.0f}% confidence")
    if drop_face or drop_voice:
        missing = [m for m, d in [("face", drop_face), ("voice", drop_voice)] if d]
        st.info(f"Graceful degradation: {', '.join(missing)} modality missing → "
                f"embedding zeroed, prediction from remaining modalities.")

    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Severity scores")
        for name, v, mx in zip(["Depression", "Anxiety", "Stress"], reg, C.REG_MAXIMA):
            st.metric(name, f"{v:.0f} / {mx}")
        st.subheader("Modality gate weights")
        st.bar_chart({"weight": {l: float(g) for l, g in
                     zip(["Facial", "Voice", "Behav/Phys"], gate)}})
    with m2:
        st.subheader("Concordance")
        st.progress(min(max(conc, 0.0), 1.0))
        vmsg = {"report": "✅ Report with confidence",
                "caveat": "⚠ Report with disagreement caveat",
                "human_review": "🚩 Flag for human review"}[verdict]
        st.write(f"**{conc:.2f}** — {vmsg}")

    # report card
    from src.explain import report_card
    rc = report_card(
        pid="LIVE", cls_proba=proba, cls_std=None,
        reg_mean=reg, reg_std=np.zeros(3), gate=gate,
        concordance=conc, conflict=False)
    st.subheader("Screening report card")
    st.code(rc)
    st.download_button("Download report", rc, file_name="screening_report.txt")

st.divider()
st.caption("Synthetic multimodal pairing · acted emotions · not clinically validated · "
           "decision support only.")
