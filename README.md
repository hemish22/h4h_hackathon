# Multimodal Mental-Health Screening

Face image + speech clip + 18 tabular features → 4-class stress triage
(Healthy / Mild / Moderate / Severe) + 3 severity scores (Depression, Anxiety,
Stress), with per-modality confidence, disagreement (concordance) detection, and
a plain-language clinical report card.

> **Framing:** a *screening* tool that flags people who may benefit from
> professional assessment. **Not a diagnostic instrument.**

Built for the hackathon per `hackathon_master_plan_v2.md`.

## The core idea

Three datasets, **zero shared participants** (RAVDESS speech, FER2013 faces, a
4000-row behavioural/physiological CSV). Instead of pairing rows randomly within
a stress class, we **match participants in measured-feature space**: the CSV
already contains audio-derived columns (`MFCC_Mean`, `Pitch_Mean`,
`Speech_Rate = 6/duration`, …) and a face-derived `Facial_Emotion_Variance`
(HOG). We compute those same quantities from the raw wavs/images and do
nearest-neighbour matching within class → `manifest_matched.csv`. We build a
`manifest_random.csv` too and run the **identical** model on both to prove the
matching helps.

Key findings surfaced by the audit (`reports/data_audit.md`):
- **Cross-modal mapping conflict**: `angry` and `disgust` map to *opposite*
  stress levels in the audio vs image committee tables — confined to the two
  highest classes, an exact inversion. Flagged per-row (`mapping_conflict`) and
  segmented out of the concordance analysis.
- **CSV is heavily skewed** (Severe ≈ 3%) and its 18 features are near-zero
  correlated with the targets — reported honestly, drives the class-weighting /
  asymmetric-cost choices.

## Quickstart (clone and run the demo)

Trained models + scaler are committed, so you only need deps + the raw dataset:

```bash
git clone https://github.com/hemish22/h4h_hackathon.git && cd h4h_hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_dataset.sh            # downloads ~476MB dataset from the release
streamlit run app/streamlit_app.py       # demo works: 4 one-click participants
```

On Windows the librosa/torch audio path can hard-crash — use Linux/Colab
(`notebooks/colab_train.ipynb`) for anything that touches audio.

## What's in the repo vs fetched

| In git | Fetched separately |
|---|---|
| all code, configs, notebook | raw `dataset/` → `scripts/fetch_dataset.sh` (release `data-v1`) |
| manifests, splits, match CSVs | — |
| trained models (`artifacts/*.pt`), scaler, lgbm | — |
| results/metrics, figures, demo participants | — |

Dataset layout after fetch:

```
dataset/
├── Audios/Actor_01 .. Actor_24/*.wav      # 1440 RAVDESS clips (03-01-* audio-only speech)
├── Extracted_images/{Angry,Disgust,...}/  # 28,709 FER2013 48x48 grayscale
└── mental_health_multimodal.csv           # 4000 x 22
```

## Pipeline (run in order)

```bash
python -m src.audit                 # reports/data_audit.md (counts, conflict, sanity)
python -m src.splits                # frozen splits (audio speaker-independent!)
python -m src.match_features        # data/{audio,image}_match.csv  (SLOW: librosa pyin + HOG)
python -m src.build_manifest        # data/manifest_{random,matched}.csv (+ conflict flag)
python -m src.baseline              # LightGBM tabular safety baseline + SHAP model
python -m src.train --config configs/final.yaml   # joint gated-fusion multi-task model
streamlit run app/streamlit_app.py  # live demo
```

Smoke test (no dataset / no librosa needed):

```bash
python -m tests.test_smoke
```

## Ablation table

One config flag toggles each row (`configs/*.yaml`):

| # | Model | Pairing in training | config |
|---|---|---|---|
| 1 | Tabular only (LightGBM) | n/a | `src.baseline` |
| 3–4 | Audio / Image only (aux heads) | none | read from any trained model |
| 5–6 | Decision-level fusion (uniform / tuned) | **none** | `src.decision_fusion` |
| 7 | Concat/gated fusion, random pairing | random | `configs/random.yaml` |
| 8 | Concat/gated fusion, matched pairing | matched | `configs/matched.yaml` |
| 10 | + ordinal loss + modality dropout (final) | matched | `configs/final.yaml` |

Report each row on the **`clean` (non-conflict)** test subset as headline, with
all-rows numbers alongside.

## Module map

| File | Role |
|---|---|
| `src/config.py` | paths, label maps, mapping tables, conflict definition |
| `src/audit.py` | Part 1 data audit |
| `src/splits.py` | frozen splits (speaker-independent audio) |
| `src/match_features.py` | audio/HOG matching-space features |
| `src/build_manifest.py` | greedy NN matching, capacity caps, conflict flag |
| `src/datasets.py` | manifest-only dataset, log-mel + augmentation, scaler |
| `src/encoders.py` | AudioCNN, FaceCNN (7-class head), TabularMLP, DummyEncoder |
| `src/fusion.py` | GatedFusion + MultiTaskModel (aux heads, modality dropout) |
| `src/losses.py` | ordinal-aware CE + joint multi-task loss + class weights |
| `src/decision_fusion.py` | no-pairing decision-level baseline |
| `src/train.py` | config-driven training loop |
| `src/evaluate.py` | every required metric + QWK/NRMSE extras |
| `src/uncertainty.py` | MC-dropout + concordance + routing |
| `src/explain.py` | report card, counterfactuals, SHAP, Grad-CAM |
| `app/streamlit_app.py` | prototype with graceful degradation |

## Team / task assignment

See GitHub Issues. Roles per plan Part 8: A-Audio, B-Vision, C-Fusion,
D-Eval&product.
