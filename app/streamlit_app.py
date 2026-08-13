"""Part 9 — Streamlit prototype.

Upload face + wav + 18 sliders -> class + confidence, 3 severity scores with
MC-dropout uncertainty, gate-weight chart, concordance gauge + routing verdict,
report card. Graceful degradation on missing modality. Conflict-aware messaging.
4 one-click demo participants (Healthy / Severe / low-concordance / conflict).

Run: streamlit run app/streamlit_app.py
"""
import json
import html
import csv
import io
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

PALETTE = {
    "ink": "#222831",
    "slate": "#393E46",
    "yellow": "#FFD369",
    "mist": "#EEEEEE",
    "white": "#FFFFFF",
}


def apply_theme():
    """Apply the Color Hunt-inspired visual system to Streamlit components."""
    st.markdown(
        f"""
        <style>
        :root {{
            --ink: {PALETTE['ink']};
            --slate: {PALETTE['slate']};
            --yellow: {PALETTE['yellow']};
            --mist: {PALETTE['mist']};
            --white: {PALETTE['white']};
        }}
        .stApp {{ background: var(--mist); color: var(--ink); }}
        [data-testid="stAppViewContainer"] {{ background: var(--mist); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stDecoration"] {{ background-image: none; }}
        .block-container {{ max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }}
        h1, h2, h3, h4, p, label {{ color: var(--ink); }}
        .stCaption {{ color: var(--slate); }}
        [data-testid="stSidebar"] {{ background: var(--ink); }}
        [data-testid="stSidebar"] * {{ color: var(--mist) !important; }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(238,238,238,.22); }}
        .stButton > button, .stDownloadButton > button {{
            border: 1px solid var(--slate);
            border-radius: 10px;
            color: var(--ink);
            background: var(--white);
            font-weight: 600;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--ink);
            background: var(--yellow);
            color: var(--ink);
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: rgba(255,255,255,.68);
            border: 1px dashed var(--slate);
            border-radius: 12px;
        }}
        /* the uploader's internal Upload/Browse button was dark-on-dark */
        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--white) !important;
            color: var(--ink) !important;
            border: 1px solid var(--slate) !important;
            font-weight: 600;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            background: var(--yellow) !important;
            color: var(--ink) !important;
            border-color: var(--ink) !important;
        }}
        [data-testid="stFileUploaderDropzone"] button * {{ color: var(--ink) !important; }}
        [data-testid="stFileUploaderDropzone"] svg {{ fill: var(--ink) !important; }}
        [data-testid="stExpander"] {{
            border: 1px solid rgba(57,62,70,.22);
            border-radius: 12px;
            background: rgba(255,255,255,.5);
        }}
        [data-testid="stMetric"] {{
            background: var(--white);
            border: 1px solid rgba(57,62,70,.16);
            border-radius: 12px;
            padding: 0.8rem;
        }}
        .hero {{
            background: var(--ink);
            border-radius: 20px;
            padding: 2.3rem 2.5rem;
            color: var(--mist);
            box-shadow: 0 12px 30px rgba(34,40,49,.12);
        }}
        .hero h1, .hero p, .hero span {{ color: var(--mist); }}
        .hero h1 {{ margin: .35rem 0 .7rem; font-size: clamp(2rem, 4vw, 3.7rem); line-height: 1.03; }}
        .eyebrow {{
            color: var(--yellow) !important;
            font-size: .74rem;
            font-weight: 800;
            letter-spacing: .13em;
            text-transform: uppercase;
        }}
        .hero-copy {{ max-width: 680px; font-size: 1.05rem; line-height: 1.65; opacity: .88; }}
        .hero-note {{ margin-top: 1.35rem; font-size: .84rem; opacity: .72; }}
        .workflow {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 1.1rem 0 2rem; }}
        .workflow-step {{
            background: var(--white);
            border: 1px solid rgba(57,62,70,.16);
            border-radius: 14px;
            padding: 1.1rem;
        }}
        .workflow-number {{ color: var(--slate); font-weight: 800; font-size: .78rem; letter-spacing: .1em; }}
        .workflow-step h4 {{ margin: .45rem 0 .3rem; }}
        .workflow-step p {{ margin: 0; color: var(--slate); font-size: .88rem; line-height: 1.5; }}
        .result-banner {{
            background: var(--white);
            border-left: 7px solid var(--yellow);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin: 1.2rem 0;
        }}
        .result-banner h2 {{ margin: .25rem 0; }}
        .result-banner p {{ margin: 0; color: var(--slate); }}
        .section-kicker {{ color: var(--slate); font-size: .76rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
        .severity-card, .modality-card, .why-card, .trust-card {{
            background: var(--white);
            border: 1px solid rgba(57,62,70,.15);
            border-radius: 14px;
            padding: 1rem;
            height: 100%;
        }}
        .severity-card h4, .modality-card h4 {{ margin: .25rem 0 .15rem; }}
        .muted {{ color: var(--slate); font-size: .84rem; }}
        .bar-track {{ height: 10px; width: 100%; background: var(--mist); border-radius: 99px; overflow: hidden; margin: .8rem 0 .45rem; }}
        .bar-fill {{ height: 100%; border-radius: 99px; }}
        .bar-label {{ display: flex; justify-content: space-between; gap: .5rem; font-size: .78rem; color: var(--slate); }}
        .status-pill {{
            display: inline-block; background: var(--yellow); color: var(--ink);
            border-radius: 99px; padding: .3rem .65rem; font-size: .75rem; font-weight: 800;
        }}
        .prob-row {{ margin-top: .5rem; }}
        .prob-label {{ display: flex; justify-content: space-between; font-size: .75rem; color: var(--slate); }}
        .reason-list {{ margin: .5rem 0 0; padding-left: 1.1rem; color: var(--slate); line-height: 1.8; }}
        .trust-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: .8rem; }}
        .trust-card h4 {{ margin-top: 0; }}
        .trust-card p {{ color: var(--slate); font-size: .88rem; line-height: 1.55; }}
        .history-row {{ background: var(--white); border-radius: 10px; padding: .7rem .9rem; margin: .35rem 0; border: 1px solid rgba(57,62,70,.13); }}
        @media (max-width: 760px) {{
            .hero {{ padding: 1.5rem; }}
            .workflow, .trust-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def pretty_class(value):
    return value.replace("_Stress", "").replace("_", " ")


def clamp01(value):
    return min(max(float(value), 0.0), 1.0)


def severity_band(ratio):
    ratio = clamp01(ratio)
    if ratio < 0.34:
        return "Lower signal", PALETTE["slate"]
    if ratio < 0.67:
        return "Moderate signal", PALETTE["yellow"]
    return "Higher signal", PALETTE["ink"]


def render_landing():
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Multimodal screening studio</div>
            <h1>See the whole signal,<br>not just one input.</h1>
            <p class="hero-copy">
                A research prototype that combines facial, voice, and behavioural signals
                to surface stress indicators, uncertainty, and moments where human review
                matters most.
            </p>
            <p class="hero-note">Screening support only. This tool does not provide a diagnosis.</p>
        </section>
        <div class="workflow">
            <div class="workflow-step">
                <div class="workflow-number">01 / COLLECT</div>
                <h4>Bring the signals together</h4>
                <p>Use a sample participant or add a face image, speech clip, and behavioural features.</p>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">02 / FUSE</div>
                <h4>Compare before combining</h4>
                <p>Each modality gets its own read, then the model produces a fused screening signal.</p>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">03 / REVIEW</div>
                <h4>Make uncertainty visible</h4>
                <p>Explore severity, confidence, agreement, and the reasons a result may need attention.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_name=None):
    with st.sidebar:
        st.markdown("## Screening studio")
        st.markdown("A calm, transparent interface for multimodal research demos.")
        st.divider()
        st.markdown("### Demo checklist")
        st.markdown("1. Start with a sample participant")
        st.markdown("2. Compare the three modality reads")
        st.markdown("3. Open Advanced details for the numbers")
        st.markdown("4. Check the Trust and responsible use panel")
        st.divider()
        st.markdown("### Current model")
        st.markdown(f"`{model_name or 'Loading...'}`")
        st.caption("25 MC-dropout passes are used for uncertainty estimates.")
        if st.button("Reset session history", key="reset_history"):
            st.session_state["assessment_history"] = []
            st.rerun()


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


@st.cache_data
def load_model_metrics():
    p = os.path.join(C.RESULTS_DIR, "final_matched", "metrics.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def payload_csv(payload):
    """Flatten the live result enough for spreadsheet-friendly export."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["field", "value"])
    scalar_fields = ["source", "class", "confidence", "concordance", "verdict"]
    for field in scalar_fields:
        writer.writerow([field, payload[field]])
    for name, values in payload["severity"].items():
        writer.writerow([f"severity.{name}.score", values["score"]])
        writer.writerow([f"severity.{name}.uncertainty", values["uncertainty"]])
    for name, values in payload["modalities"].items():
        writer.writerow([f"modality.{name}.class", values["class"]])
        writer.writerow([f"modality.{name}.confidence", values["confidence"]])
        writer.writerow([f"modality.{name}.gate_weight", values["gate_weight"]])
    return output.getvalue()


def render_model_snapshot():
    metrics = load_model_metrics()
    st.markdown("### Validation snapshot")
    if not metrics:
        st.caption("No saved validation metrics were found in results/final_matched/.")
        return
    classification = metrics.get("classification", {})
    metric_cols = st.columns(3)
    for col, label, value in zip(
        metric_cols,
        ["Accuracy", "Macro F1", "Weighted F1"],
        [classification.get("accuracy"), classification.get("macro_f1"), classification.get("weighted_f1")],
    ):
        with col:
            st.metric(label, f"{value:.3f}" if value is not None else "n/a")
    matrix = classification.get("confusion_matrix")
    if matrix:
        labels = [pretty_class(cls) for cls in C.STRESS_CLASSES]
        rows = []
        for label, row in zip(labels, matrix):
            rows.append({"Actual / predicted": label, **dict(zip(labels, row))})
        st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("Held-out validation snapshot for the saved model; it is not a calibration guarantee for a new participant.")


def remember_result(result):
    history = st.session_state.setdefault("assessment_history", [])
    signature = (
        result["source"], result["class"], round(result["confidence"], 4),
        round(result["concordance"], 4),
    )
    if not history or history[-1].get("signature") != signature:
        result = dict(result)
        result["signature"] = signature
        history.append(result)
        st.session_state["assessment_history"] = history[-6:]


def result_payload(proba, pstd, reg, rstd, gate, aux, concordance_score,
                   verdict, dropped, conflict_hint, source_label):
    modality_names = ["Face", "Voice", "Behavioural / physiological"]
    return {
        "source": source_label,
        "class": pretty_class(C.STRESS_CLASSES[int(proba.argmax())]),
        "confidence": float(proba.max()),
        "concordance": float(concordance_score),
        "verdict": verdict,
        "dropped_modalities": list(dropped),
        "conflict_hint": bool(conflict_hint),
        "severity": {
            name: {"score": float(value), "uncertainty": float(std), "maximum": int(maximum)}
            for name, value, std, maximum in zip(
                ["Depression", "Anxiety", "Stress"], reg, rstd, C.REG_MAXIMA
            )
        },
        "modalities": {
            name: {
                "class": pretty_class(C.STRESS_CLASSES[int(p.argmax())]),
                "confidence": float(p.max()),
                "probabilities": {
                    pretty_class(cls): float(prob)
                    for cls, prob in zip(C.STRESS_CLASSES, p)
                },
                "gate_weight": float(weight),
            }
            for name, p, weight in zip(modality_names, aux, gate)
        },
        "class_probabilities": {
            pretty_class(cls): float(prob)
            for cls, prob in zip(C.STRESS_CLASSES, proba)
        },
        "class_uncertainty": {
            pretty_class(cls): float(std)
            for cls, std in zip(C.STRESS_CLASSES, pstd)
        },
    }


def render_severity(reg, rstd):
    st.markdown('<div class="section-kicker">Severity snapshot</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, name, value, std, maximum in zip(
        cols, ["Depression", "Anxiety", "Stress"], reg, rstd, C.REG_MAXIMA
    ):
        ratio = clamp01(float(value) / maximum)
        band, color = severity_band(ratio)
        with col:
            st.markdown(
                f"""
                <div class="severity-card">
                    <div class="muted">{name}</div>
                    <h4>{band}</h4>
                    <div class="bar-track"><div class="bar-fill" style="width:{ratio * 100:.1f}%; background:{color};"></div></div>
                    <div class="bar-label"><span>Relative signal</span><span>{ratio * 100:.0f}%</span></div>
                    <div class="muted">Open Advanced details for score and uncertainty.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_modality_comparison(aux, gate):
    st.markdown('<div class="section-kicker">Side-by-side modality comparison</div>', unsafe_allow_html=True)
    names = ["Face", "Voice", "Behavioural / physiological"]
    cols = st.columns(3)
    for col, name, probabilities, weight in zip(cols, names, aux, gate):
        top_index = int(probabilities.argmax())
        with col:
            rows = []
            for cls, probability in zip(C.STRESS_CLASSES, probabilities):
                width = clamp01(probability) * 100
                rows.append(
                    f'<div class="prob-row">'
                    f'<div class="prob-label"><span>{pretty_class(cls)}</span>'
                    f'<span>{probability * 100:.0f}%</span></div>'
                    f'<div class="bar-track"><div class="bar-fill" '
                    f'style="width:{width:.1f}%; background:{PALETTE["slate"]};"></div></div>'
                    f'</div>'
                )
            st.markdown(
                f"""
                <div class="modality-card">
                    <div class="muted">{name}</div>
                    <h4>{pretty_class(C.STRESS_CLASSES[top_index])}</h4>
                    <span class="status-pill">{probabilities[top_index] * 100:.0f}% top read</span>
                    {''.join(rows)}
                    <div class="muted" style="margin-top:.8rem;">Fusion weight: {weight * 100:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_why_result(proba, gate, aux, concordance_score, verdict, conflict_hint):
    class_name = pretty_class(C.STRESS_CLASSES[int(proba.argmax())])
    strongest_modality = ["Face", "Voice", "Behavioural / physiological"][int(np.argmax(gate))]
    reasons = [
        f"The fused read is {class_name} with {proba.max() * 100:.0f}% model confidence.",
        f"The fusion gave the largest contribution to the {strongest_modality.lower()} signal ({gate.max() * 100:.0f}%).",
    ]
    if concordance_score < 0.45:
        reasons.append("The modalities disagree substantially, so this result should be reviewed carefully.")
    elif concordance_score < 0.75:
        reasons.append("The modalities show moderate agreement; the combined result includes some uncertainty.")
    else:
        reasons.append("The modalities are broadly aligned, supporting a more consistent combined read.")
    if conflict_hint:
        reasons.append("This demo includes a known cross-modal annotation conflict; treat the output as a research signal.")
    st.markdown('<div class="section-kicker">Why this result?</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="why-card">
            <h4>How the model arrived here</h4>
            <ul class="reason-list">{''.join(f'<li>{html.escape(reason)}</li>' for reason in reasons)}</ul>
            <p class="muted" style="margin-top:.8rem;">Concordance: {concordance_score * 100:.0f}% · Routing: {html.escape(verdict.replace('_', ' '))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trust_panel(dropped, conflict_hint):
    st.markdown('<div class="section-kicker">Responsible use</div>', unsafe_allow_html=True)
    dropped_text = "No modality was dropped for this assessment."
    if dropped:
        dropped_text = f"Missing inputs: {', '.join(dropped)}. The model zeroed the missing embedding, so confidence may be affected."
    conflict_text = (
        "This scenario contains a known annotation conflict between modality mappings."
        if conflict_hint else
        "Always inspect modality disagreement rather than relying on the fused label alone."
    )
    st.markdown(
        f"""
        <div class="trust-grid">
            <div class="trust-card">
                <h4>Screening, not diagnosis</h4>
                <p>This prototype flags patterns that may benefit from professional assessment. It should not be used to diagnose, triage, or make decisions about a person without qualified human review.</p>
            </div>
            <div class="trust-card">
                <h4>Uncertainty is part of the result</h4>
                <p>Confidence, concordance, and missing-modality states are shown so a reviewer can see when the output is less reliable.</p>
            </div>
            <div class="trust-card">
                <h4>Data and model limits</h4>
                <p>The demo uses synthetic multimodal pairing and acted emotions. It is not clinically validated and should not be treated as generalisable clinical evidence.</p>
            </div>
            <div class="trust-card">
                <h4>Review notes</h4>
                <p>{html.escape(dropped_text)} {html.escape(conflict_text)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history():
    history = st.session_state.get("assessment_history", [])
    if not history:
        return
    st.markdown('<div class="section-kicker">Stretch feature: session history</div>', unsafe_allow_html=True)
    for item in reversed(history):
        st.markdown(
            f"""
            <div class="history-row">
                <strong>{html.escape(item['source'])}</strong> · {html.escape(item['class'])}
                <span class="muted"> · confidence {item['confidence'] * 100:.0f}% · concordance {item['concordance'] * 100:.0f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


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


@st.cache_data
def _csv_cached():
    return D.load_csv()


def render_probabilities(proba):
    """#3 — full 4-class ordinal probability bar (not just the top label)."""
    st.markdown('<div class="section-kicker">Class probabilities (ordered scale)</div>',
                unsafe_allow_html=True)
    top = int(proba.argmax())
    for i, cls in enumerate(C.STRESS_CLASSES):
        pct = float(proba[i]) * 100
        color = PALETTE["yellow"] if i == top else PALETTE["slate"]
        st.markdown(
            f"""<div class="prob-row">
                  <div class="prob-label"><span>{pretty_class(cls)}</span><span>{pct:.0f}%</span></div>
                  <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color};"></div></div>
                </div>""",
            unsafe_allow_html=True,
        )


def _predict_stress_fn(model, scaler, face_t, mel_t):
    """dict[feature]->value  ->  predicted Stress score (original units)."""
    import torch
    def f(feats):
        x = scaler.transform(np.array([[feats[c] for c in C.FEATURE_COLS]])).astype(np.float32)
        with torch.no_grad():
            o = model(face_t, mel_t, torch.from_numpy(x))
        return float(o.reg[0, 2]) * C.REG_MAXIMA[2]
    return f


def render_counterfactuals(model, scaler, face_np, mel_np, base_feats):
    """#1 — what would change this: sweep actionable features, show top movers."""
    import torch
    from src.explain import top_counterfactuals
    face_t = torch.zeros(1, 48, 48) if face_np is None else torch.from_numpy(face_np).unsqueeze(0)
    mel_t = torch.zeros(1, 128, 188) if mel_np is None else torch.from_numpy(mel_np).unsqueeze(0)
    fn = _predict_stress_fn(model, scaler, face_t, mel_t)
    try:
        cfs = top_counterfactuals(fn, base_feats, _csv_cached(), k=3)
    except Exception as e:
        st.caption(f"Counterfactuals unavailable: {e}")
        return
    st.markdown('<div class="section-kicker">What would change this (actionable)</div>',
                unsafe_allow_html=True)
    rows = "".join(
        f"<li><strong>{feat.replace('_',' ')}</strong> → {val:.0f}"
        f" &nbsp; predicted stress <strong>{d:+.1f}</strong></li>"
        for feat, val, d in cfs)
    st.markdown(f'<ul class="reason-list">{rows}</ul>', unsafe_allow_html=True)
    st.caption("Estimated effect of moving one behavioural feature, holding face & voice fixed.")


def render_gradcam(model, face_np, mel_np):
    """#2 — live Grad-CAM overlay on THIS participant's face + spectrogram."""
    if face_np is None and mel_np is None:
        return
    import io, torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from src.run_explain import _gradcam, _last_conv
    face_t = torch.zeros(1, 48, 48) if face_np is None else torch.from_numpy(face_np).unsqueeze(0)
    mel_t = torch.zeros(1, 128, 188) if mel_np is None else torch.from_numpy(mel_np).unsqueeze(0)
    tab_t = torch.zeros(1, 18)
    batch = {"face": face_t, "voice": mel_t, "tab": tab_t}
    st.markdown('<div class="section-kicker">Where the model looked (Grad-CAM)</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    if face_np is not None:
        cam, _ = _gradcam(model, batch, "face", _last_conv(model.face))
        cam = zoom(cam, (48 / cam.shape[0], 48 / cam.shape[1]), order=1)
        fig, ax = plt.subplots(figsize=(3, 3)); ax.imshow(face_np, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45); ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
        cols[0].image(buf.getvalue(), caption="Face attention", width=220)
    if mel_np is not None:
        cam, _ = _gradcam(model, batch, "voice", _last_conv(model.voice))
        cam = zoom(cam, (mel_np.shape[0] / cam.shape[0], mel_np.shape[1] / cam.shape[1]), order=1)
        fig, ax = plt.subplots(figsize=(4, 2.2)); ax.imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
        ax.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.4); ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
        cols[1].image(buf.getvalue(), caption="Voice attention (mel, time →)", width=340)


@st.cache_data
def _load_result(run):
    p = os.path.join(C.RESULTS_DIR, run, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def render_performance_panel():
    """#4 — 'About the model': ablation, calibration/ECE, concordance evidence,
    confusion matrix, effective sample size. All from committed results/figures."""
    st.markdown('<div class="section-kicker">Evaluated on the held-out test set '
                '(600 participants)</div>', unsafe_allow_html=True)

    # --- ablation table (built from results/*.json)
    rows = [("lgbm_tabular", "Tabular only (LightGBM)", "n/a"),
            ("concat_random", "Gated fusion", "random"),
            ("concat_matched", "Gated fusion", "matched"),
            ("final_matched", "Final (+ordinal +moddrop)", "matched")]
    lines = ["| Model | Pairing | Acc | MacroF1 | ROC-AUC | QWK | Str MAE |",
             "|---|---|---|---|---|---|---|"]
    for run, name, pair in rows:
        d = _load_result(run)
        if not d:
            continue
        c = d["classification"]; r = d["regression"]
        roc = c.get("roc_auc_ovr_macro")
        roc = f"{roc:.3f}" if roc else "—"
        lines.append(f"| {name} | {pair} | {c['accuracy']:.3f} | {c['macro_f1']:.3f} | "
                     f"{roc} | {c['qwk_EXTRA']:.3f} | {r['Stress']['MAE']:.2f} |")
    dfz = _load_result("decision_fusion")
    if dfz:
        for key, nm in [("decision_uniform", "Decision-level fusion (no pairing)")]:
            c = dfz.get(key)
            if c:
                roc = c.get("roc_auc_ovr_macro"); roc = f"{roc:.3f}" if roc else "—"
                lines.append(f"| {nm} | none | {c['accuracy']:.3f} | {c['macro_f1']:.3f} | "
                             f"{roc} | {c['qwk_EXTRA']:.3f} | — |")
    st.markdown("\n".join(lines))
    st.caption("Matched pairing beats random by ~10 accuracy points under an identical "
               "model — isolating the contribution of the feature-matched alignment engine.")

    col1, col2 = st.columns(2)
    # --- calibration / ECE
    with col1:
        cal = _load_result("calibration")
        fig = os.path.join(C.FIGURES_DIR, "calibration.png")
        if cal and os.path.exists(fig):
            st.image(fig, caption=f"Reliability diagram · ECE = {cal['ece']:.3f}")
    # --- concordance band evidence (clean subset)
    with col2:
        conc = _load_result("concordance")
        if conc and conc.get("clean"):
            cl = conc["clean"]
            st.markdown("**Accuracy by concordance band (clean subset)**")
            band_acc = {b.replace("_", " "): (cl[b]["accuracy"] or 0)
                        for b in ["report", "caveat", "human_review"]}
            st.bar_chart(band_acc)
            st.caption("Accuracy collapses when the modalities disagree → concordance "
                       "is a real reliability signal, used for human-review routing.")

    # --- confusion matrix
    cm = os.path.join(C.FIGURES_DIR, "confusion_final_matched.png")
    if os.path.exists(cm):
        st.image(cm, caption="Confusion matrix (final model) — watch the Moderate↔Severe cell")
    st.caption("⚠ Effective sample size: the Severe class has only 19 test participants "
               "(the CSV is ~3% Severe) — Severe-class metrics are reported with that caveat.")


def render(proba, pstd, reg, rstd, gate, aux, dropped, conflict_hint=False,
           source_label="Live assessment", face_np=None, mel_np=None, base_feats=None):
    conc, _ = concordance(aux[0][None], aux[1][None], aux[2][None])
    conc = float(conc[0]); verdict = route(conc)
    ci = int(proba.argmax())
    class_name = pretty_class(C.STRESS_CLASSES[ci])
    payload = result_payload(proba, pstd, reg, rstd, gate, aux, conc, verdict,
                             dropped, conflict_hint, source_label)
    remember_result(payload)
    st.markdown(
        f"""
        <div class="result-banner">
            <div class="section-kicker">{html.escape(source_label)}</div>
            <h2>{html.escape(class_name)} screening signal</h2>
            <p>Fused confidence: <strong>{proba[ci] * 100:.0f}%</strong> · Concordance: <strong>{conc * 100:.0f}%</strong> · Routing: <strong>{html.escape(verdict.replace('_', ' '))}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if dropped:
        st.warning(f"Graceful degradation: {', '.join(dropped)} missing. The missing embedding was zeroed.")

    render_probabilities(proba)
    st.markdown("<br>", unsafe_allow_html=True)
    render_severity(reg, rstd)
    st.markdown("<br>", unsafe_allow_html=True)
    render_modality_comparison(aux, gate)
    st.markdown("<br>", unsafe_allow_html=True)
    if face_np is not None or mel_np is not None:
        render_gradcam(model, face_np, mel_np)
        st.markdown("<br>", unsafe_allow_html=True)
    if base_feats is not None:
        render_counterfactuals(model, scaler, face_np, mel_np, base_feats)
        st.markdown("<br>", unsafe_allow_html=True)
    render_why_result(proba, gate, aux, conc, verdict, conflict_hint)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Decision path</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow">
            <div class="workflow-step"><div class="workflow-number">SIGNALS</div><h4>Three views of the participant</h4><p>Face, voice, and behavioural or physiological inputs are read independently.</p></div>
            <div class="workflow-step"><div class="workflow-number">FUSION</div><h4>One combined screening signal</h4><p>Modality weights show which input contributed most to the fused result.</p></div>
            <div class="workflow-step"><div class="workflow-number">REVIEW</div><h4>Context before action</h4><p>Confidence, concordance, and limitations guide a qualified reviewer.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Advanced details: numeric outputs", expanded=False):
        st.subheader("Severity scores and uncertainty")
        metric_cols = st.columns(3)
        for col, name, value, std, maximum in zip(
            metric_cols, ["Depression", "Anxiety", "Stress"], reg, rstd, C.REG_MAXIMA
        ):
            with col:
                st.metric(name, f"{value:.0f} ± {std:.0f}", help=f"Scale 0–{maximum}")
        st.subheader("Fused class probabilities")
        st.bar_chart({"probability": {pretty_class(cls): float(prob) for cls, prob in zip(C.STRESS_CLASSES, proba)}})
        st.subheader("Modality gate weights")
        st.bar_chart({"weight": {l: float(g) for l, g in
                     zip(["Face", "Voice", "Behavioural / physiological"], gate)}})
        st.subheader("Plain-text report")
        from src.explain import report_card
        rc = report_card("LIVE", proba, pstd, reg, rstd, gate,
                         concordance=conc, conflict=conflict_hint)
        st.code(rc)
        report_col, data_col = st.columns(2)
        with report_col:
            st.download_button("Download report", rc, file_name="screening_report.txt", key="download_report")
        with data_col:
            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    "Export result JSON",
                    json.dumps(payload, indent=2),
                    file_name="screening_result.json",
                    mime="application/json",
                    key="download_json",
                )
            with export_cols[1]:
                st.download_button(
                    "Export result CSV",
                    payload_csv(payload),
                    file_name="screening_result.csv",
                    mime="text/csv",
                    key="download_csv",
                )

    with st.expander("Trust and responsible use", expanded=False):
        render_trust_panel(dropped, conflict_hint)

    with st.expander("Model card and demo notes", expanded=False):
        st.markdown(
            """
            **What is visible here**

            - The fused screening class and its probability distribution
            - Regression-style severity estimates with MC-dropout uncertainty
            - Per-modality predictions and learned fusion weights
            - Concordance between modality predictions

            **What is not available from this live screen**

            - A clinical diagnosis or treatment recommendation
            - A validated population-level risk estimate
            - A reliable causal explanation of a person's mental health

            Use the output as a structured conversation starter for qualified reviewers.
            """
        )
        render_model_snapshot()

    render_history()


apply_theme()
render_landing()

model, model_name = load_model()
scaler = load_scaler()
med, rng = dataset_stats()
demos = load_demos()
render_sidebar(model_name)

if model is None:
    st.error("No trained model in artifacts/. Add model_final_matched.pt.")
    st.stop()
st.caption(f"Ready for assessment · loaded model: `{model_name}`")

with st.expander("📊 About the model — held-out performance & evidence", expanded=False):
    render_performance_panel()

# ---- demo participants
if demos:
    st.subheader("Start with a demo participant")
    st.caption("Each scenario is preloaded so the story can be demonstrated without file preparation.")
    cols = st.columns(len(demos))
    for col, (key, d) in zip(cols, demos.items()):
        if col.button(d["tag"], key=f"demo_{key}"):
            st.session_state["demo"] = dict(d, _key=key)

# ---- manual inputs
st.subheader("Or build your own assessment")
st.caption("Add any available inputs. Missing modalities are handled explicitly in the result.")
c1, c2 = st.columns(2)
with c1:
    face_mode = st.radio("Face input", ["Upload", "Webcam"], horizontal=True, key="face_mode")
    if face_mode == "Webcam":
        up_img = st.camera_input("Take a photo")
    else:
        up_img = st.file_uploader("Face image", type=["png", "jpg", "jpeg"])
with c2:
    voice_mode = st.radio("Voice input", ["Upload", "Record"], horizontal=True, key="voice_mode")
    if voice_mode == "Record":
        up_wav = st.audio_input("Record speech")
    else:
        up_wav = st.file_uploader("Speech .wav", type=["wav"])
st.caption("⚠ Live webcam/mic inputs are out-of-distribution (model trained on "
           "registered 48×48 faces + acted studio speech) — treat live results as "
           "illustrative. The demo participants above are the reliable story.")
with st.expander("Behavioural and physiological features", expanded=False):
    st.caption("These controls also work as a simple what-if exploration tool. Adjust them and run Predict again.")
    tcols = st.columns(3)
    tab_vals = {}
    for i, f in enumerate(C.FEATURE_COLS):
        lo, hi = rng[f]
        tab_vals[f] = tcols[i % 3].slider(f, lo, hi, med[f])

go = st.button("Run screening assessment", type="primary")

# ---- demo path
if "demo" in st.session_state and (go is False):
    d = st.session_state.pop("demo")
    from src.datasets import load_face, load_logmel
    # prefer the small bundled media so the app runs without the full dataset
    _key = d.get("_key", "")
    _img_b = os.path.join(C.ROOT, "demo_assets", "media", f"{_key}.png")
    _wav_b = os.path.join(C.ROOT, "demo_assets", "media", f"{_key}.wav")
    img_path = _img_b if os.path.exists(_img_b) else D.resolve(d["image_path"])
    wav_path = _wav_b if os.path.exists(_wav_b) else D.resolve(d["audio_path"])
    face = load_face(img_path); mel = load_logmel(wav_path)
    tv = scaler.transform(np.array([[d["features"][f] for f in C.FEATURE_COLS]]))[0]
    st.info(f"**{d['tag']}** — participant #{d['participant_id']} "
            f"(true: {d['true_class'].replace('_',' ')})")
    # show the actual inputs used for this demo
    ic, ac = st.columns([1, 2])
    with ic:
        st.caption("Face input")
        st.image(img_path, width=160)
    with ac:
        st.caption("Voice input")
        with open(wav_path, "rb") as _f:
            st.audio(_f.read(), format="audio/wav")
    proba, pstd, reg, rstd, gate, aux = infer(model, face, mel, tv, False, False)
    render(proba, pstd, reg, rstd, gate, aux, [], conflict_hint=d["conflict"],
           source_label=f"Demo · {d['tag']}",
           face_np=face, mel_np=mel, base_feats=dict(d["features"]))

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
    render(proba, pstd, reg, rstd, gate, aux, dropped, source_label="Manual assessment",
           face_np=face, mel_np=mel, base_feats=dict(tab_vals))

st.divider()
st.caption("Synthetic multimodal pairing · acted emotions · not clinically validated · decision support only.")
