# Multimodal Mental Health Prediction — Master Build Plan **v2**

**System:** face image + speech clip + 18 tabular features → 4-class stress triage + 3 severity scores, with per-modality confidence, disagreement detection, and a plain-language clinical report card.

**Framing for judges:** a *screening* tool that flags people who may benefit from professional assessment. Not a diagnostic instrument. Say this in slide 1 and slide N.

---

# What changed from v1

Reconciled against the three committee documents (Problem Statement, Dataset Description, Metrics Used).

**Confirmed unchanged — no edits needed:** all file counts (1440 / 28,709 / 4000×22), both emotion→stress mapping tables, score maxima (34 / 24 / 39), the full metric list (all 8 classification + all 5 regression metrics were already covered in Part 6), the 18 feature names, and the three objectives.

| # | Change | Where |
|---|---|---|
| 1 | **Cross-modal mapping conflict** — angry and disgust are mapped to *opposite* stress levels in audio vs images. Now audited, flagged per-row in the manifest, segmented out of the concordance analysis, and given its own slide. | 1.3, 2.7, 5.1, 10, 12 |
| 2 | **Decision-level fusion added as a first-class baseline** — three unimodal stress posteriors combined with no pairing in training. Two new ablation rows. | 4.9, 6.3, 11 |
| 3 | **Capacity caps recomputed per split** with real per-actor arithmetic. Audio is the only capacity-constrained modality; image caps are 1 throughout. | 2.6 |
| 4 | **Only one of four video-derived tabular columns is recoverable** from static 48×48 stills. Stated explicitly rather than silently worked around. | 2.5, 10 |
| 5 | **`Speech_Rate = 6 / duration` upgraded from assumption to confirmed fact** — spec confirms all 1440 files are modality 03 / channel 01, both statements exactly six words. | 2.4 |
| 6 | **Emotional intensity (filename field 4) used as a matching tie-breaker.** Free arousal proxy. | 2.4, 2.6 |
| 7 | **Rubric flagged as unverified** — the mark weights below are not in any committee document. Confirm before allocating effort. | 0 |

---

# Part 0 — Rubric traceability

> ⚠ **Verify before you build.** The mark weights in this table are **not present** in the Problem Statement, Dataset Description, or Metrics document. Confirm the marking scheme with the committee in the first hour, and ask two further questions while you're at it: **(a) is there a held-out evaluation set they will score against, or do you report your own test split?** and **(b) is there a required submission format?** If the weights differ from what's below, re-derive your effort allocation before writing code — the entire plan is optimised against this table.

| # | Criterion | Marks | What you build for it | Section |
|---|---|---|---|---|
| 1 | Problem understanding & design | 5 | Screening-vs-diagnosis framing, disjoint-dataset problem statement, mapping-conflict analysis, system diagram | Part 8 |
| 2 | Preprocessing & feature engineering | 7 | Per-modality pipelines, augmentation, class weights, handcrafted feature extraction | Part 3 |
| 3 | **Multimodal integration** | **8** | **Feature-matched alignment + pairing ablation + decision-level baseline + gated fusion + modality dropout** | **Part 2, 4** |
| 4 | Classification performance | 7 | Multi-task head, ordinal-aware loss, all 8 required metrics, confusion matrix | Part 4, 6 |
| 5 | Regression performance | 7 | 3 heads, all 5 required metrics per target, uncertainty bands | Part 4, 6 |
| 6 | Innovation & novelty | 5 | Feature-space alignment, ordinal loss, concordance score, mapping-conflict finding, MC-dropout uncertainty | Part 2, 4, 5 |
| 7 | Explainability & clinical interpretability | 4 | SHAP + Grad-CAM + gate weights + **report card** + counterfactuals | Part 7 |
| 8 | Implementation & prototype | 4 | Streamlit app, graceful degradation, reset/demo samples | Part 9 |
| 9 | Presentation & demonstration | 3 | Rehearsed demo, ablation slide, honest limitations slide | Part 10 |

**Weight distribution insight:** criteria 3+4+5 = 22 marks and all three flow from the same artefact — a correct, well-constructed pairing manifest feeding one multi-task model. Build that spine first; nearly half the rubric hangs off it.

---

# Part 1 — Data audit (first 30 minutes, non-negotiable)

Before writing any model code, produce `reports/data_audit.md`.

## 1.1 Verify counts

```python
# audio
len(glob('ravdess/**/*.wav', recursive=True))     # expect 1440
# images
{d: len(os.listdir(f'fer/{d}')) for d in os.listdir('fer')}
# expect Angry 3995, Disgust 436, Fear 4097, Happy 7215, Neutral 4965, Sad 4830, Surprise 3171
df = pd.read_csv('numerical.csv')                  # expect (4000, 22)
```

The image counts sum to exactly 28,709 and the audio counts to exactly 1440 — assert both, since a partial download is the most boring way to lose a hackathon.

**Audio structure, derived from the spec and worth asserting in code:** 24 actors × 60 clips. Per actor that is neutral 4 (2 statements × 2 repetitions × 1 intensity) and 8 each for the other seven emotions (2 statements × 2 reps × 2 intensities). Every filename is `03-01-*` — audio-only, speech channel. Assert this; it is the premise of §2.4.

## 1.2 Post-mapping class distributions

Apply both mapping tables and record:

| Stress level | Audio | Images | CSV |
|---|---|---|---|
| Healthy | 480 (neutral 96 + calm 192 + happy 192) | 12,180 (happy 7215 + neutral 4965) | ? |
| Mild | 384 (sad 192 + surprised 192) | 8,001 (sad 4830 + surprise 3171) | ? |
| Moderate | 384 (fearful 192 + angry 192) | 4,533 (fear 4097 + disgust 436) | ? |
| Severe | 192 (disgust 192) | 3,995 (angry 3995) | ? |

**Audio Severe = 192 clips is the binding constraint on the whole build.** Every design decision downstream (sampling with replacement, augmentation intensity, capacity caps in matching) traces back to this number.

## 1.3 ⚠ The cross-modal mapping conflict — audit this explicitly

Lay the two committee mapping tables side by side. They do not agree:

| Emotion | Audio → | Image → | Status |
|---|---|---|---|
| Neutral | Healthy | Healthy | ✅ agree |
| Happy | Healthy | Healthy | ✅ agree |
| Calm | Healthy | *(not in image set)* | — audio only |
| Sad | Mild | Mild | ✅ agree |
| Surprised | Mild | Mild | ✅ agree |
| Fearful | Moderate | Moderate | ✅ agree |
| **Angry** | **Moderate** | **Severe** | ❌ **conflict** |
| **Disgust** | **Severe** | **Moderate** | ❌ **conflict** |

The conflict is not scattered — it is confined entirely to the two highest-severity classes, and it is an exact inversion:

- **Severe** participants are paired with a **disgust voice** and an **angry face**.
- **Moderate** participants draw faces from {fear, disgust} but voices from {fearful, angry}.

The two lowest classes are cleanly aligned; the two that matter clinically are cross-wired.

**Why this is not a nitpick.** Your concordance score (Part 5.1) measures per-modality disagreement and interprets it as masked affect. But for every Severe participant, the face branch is looking at anger and the voice branch is listening to disgust — two different emotions carrying the same label by committee fiat. The branches will disagree *systematically*, and that disagreement is a labelling artefact, not a clinical signal. Left unhandled, your most novel contribution is measuring your own annotation scheme.

**What to record in the audit:**

```python
CONFLICT_EMOTIONS = {'angry', 'disgust'}   # differ across the two mapping tables

# fraction of each class's candidate pool that is conflict-sourced
# audio: Severe 192/192 = 100%,  Moderate 192/384 = 50%
# image: Severe 3995/3995 = 100%, Moderate 436/4533 = 9.6%
```

Note the asymmetry in the Moderate row — 50% of the Moderate audio pool is conflict-sourced versus under 10% of the image pool. That number goes on the slide.

Handling is specified in §2.7 (manifest flag), §5.1 (segmented concordance), Part 10 slide 8, and Part 12.

## 1.4 CSV checks

```python
df.Mental_Health_Status.value_counts(normalize=True)
df.isnull().sum()
df.describe().T                       # ranges, outliers, impossible values
df.corr(numeric_only=True)            # feature↔target correlations
# sanity: do the 3 scores correlate with the categorical label as expected?
df.groupby('Mental_Health_Status')[['Depression_Score','Anxiety_Score','Stress_Score']].mean()
```

That last line is your first presentation-worthy finding — if the mean scores rise monotonically across Healthy → Severe, you have empirical justification for treating the classes as ordinal (Part 4.3). If they don't, you need to know that now, not at hour 18.

Also check `Heart_Rate_BPM` for impossible values (<30, >220), `Skin_Temperature` outside 30–40 °C, scores outside their stated ranges (0–34 / 0–24 / 0–39), negative durations. Note anything you clip or winsorize — that's criterion 2 evidence.

**Record the CSV class balance immediately.** Every capacity-cap number in §2.6 assumes roughly 1000 participants per class; if the real distribution is skewed, recompute the caps before building the manifest.

## 1.5 Splits — decide now, write to disk, never change

```python
# AUDIO: speaker-independent. Actor ID is filename field 7.
audio_train = actors 1..18      # 1080 clips
audio_val   = actors 19,20,21   # 180 clips
audio_test  = actors 22,23,24   # 180 clips

# IMAGES: stratified random on the 7 emotion labels
img_train/val/test = 80/10/10

# CSV: stratified on Mental_Health_Status
csv_train/val/test = 70/15/15
```

Persist as `splits/{modality}_{split}.txt`. **Speaker-independent audio splitting is the single most important correctness decision in the project** — a random split lets the audio CNN memorise 24 voices and your reported accuracy becomes meaningless. Say the phrase out loud in your presentation.

Per-split audio counts follow directly from the per-actor arithmetic (Healthy 20, Mild 16, Moderate 16, Severe 8 clips per actor):

| Stress level | Train (18 actors) | Val (3) | Test (3) |
|---|---|---|---|
| Healthy | 360 | 60 | 60 |
| Mild | 288 | 48 | 48 |
| Moderate | 288 | 48 | 48 |
| **Severe** | **144** | **24** | **24** |

Write these down. §2.6 uses them, and **24 unique Severe voices in the entire test set** is a limitation you will state out loud before a judge finds it.

---

# Part 2 — The alignment engine (your primary differentiator)

## 2.1 The problem

Three datasets, zero shared participants. Criterion 3 (8 marks, the largest) asks for participant-level integration. Something has to be constructed.

## 2.2 What everyone else will do

Pair a CSV row with a randomly chosen audio clip and image from the matching stress class. This is defensible but arbitrary — no reason participant #17's voice should be *that* clip rather than any of the other 400.

## 2.3 What you do instead

**Read the tabular column list again.** It contains, already extracted:

- `MFCC_Mean`, `MFCC_Variance`, `Pitch_Mean`, `Speech_Rate` → audio-derived
- `Facial_Emotion_Variance` (described as "HOG-like"), `Smile_Intensity`, `Eye_Blink_Rate`, `Head_Motion_Index` → face/video-derived

These are *measurements of the same physical quantities you can compute from the raw wavs and images*. So instead of matching on label alone, match on **measured feature proximity within the correct stress class.**

The CSV isn't just the target — it's a specification of what each participant's voice and face should be like. Honour it.

## 2.4 Audio feature extraction (matching space)

```python
import librosa, numpy as np

def audio_match_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    y, _  = librosa.effects.trim(y, top_db=30)
    dur   = len(y) / sr

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
    f0v = f0[voiced & ~np.isnan(f0)]

    return {
        'MFCC_Mean':     float(mfcc.mean()),
        'MFCC_Variance': float(mfcc.var()),
        'Pitch_Mean':    float(np.median(f0v)) if len(f0v) else np.nan,
        'Speech_Rate':   6.0 / dur,          # exact — see below
    }


def parse_filename(name):
    """03-01-06-01-02-01-12.wav → structured metadata."""
    mod, chan, emo, inten, stmt, rep, actor = [int(x) for x in name[:-4].split('-')]
    assert mod == 3 and chan == 1, "spec: all files are audio-only speech"
    return dict(emotion=emo, intensity=inten, statement=stmt,
                repetition=rep, actor=actor,
                sex='M' if actor % 2 else 'F')
```

**`Speech_Rate` is the star, and the spec now confirms it.** The Dataset Description states every file is modality `03` (audio-only) on vocal channel `01` (speech), and that the only two statements are "Kids are talking by the door" and "Dogs are sitting by the door" — both exactly six words. So words-per-second is `6 / trimmed_duration`: not estimated, *computed*. This is an exact correspondence between the raw audio and a CSV column, and pointing it out is the moment a judge realises you read the problem properly. In v1 this was an assumption; it is now a documented fact, so say **"confirmed from the specification"** on slide 4.

**Emotional intensity is free signal.** Filename field 4 is `01 = normal` / `02 = strong` (absent for neutral). It is a direct arousal proxy and costs nothing to parse. Two uses: (a) a tie-breaker in the matching loop (§2.6), and (b) an optional auxiliary binary target for the audio encoder, which gives the branch a second gradient signal at no data cost. Ten minutes of work.

`Pitch_Mean` will be NaN for a handful of clips (heavy breathiness, whispered disgust). Impute with the class median rather than dropping — 192 Severe clips is too few to lose any.

**Definitional honesty:** `MFCC_Mean` is ambiguous in the spec (mean over coefficients? over frames? both?). Pick the grand mean over all 13 coefficients and all frames, **write your definition in the report**, and note that z-scoring makes the matching robust to the choice since it only depends on relative ordering. Judges respect a stated assumption far more than an unstated one.

## 2.5 Image feature extraction (matching space) — and what is *not* recoverable

```python
from skimage.feature import hog

def image_match_features(img48):            # uint8 (48,48)
    desc = hog(img48, orientations=9,
               pixels_per_cell=(8,8), cells_per_block=(2,2),
               feature_vector=True)
    return {'Facial_Emotion_Variance': float(desc.var())}
```

The spec literally says "HOG-like." Take the hint.

**⚠ State this limitation explicitly rather than working around it silently.** The Dataset Description specifies four face/video-derived tabular columns, and three of them **cannot be computed from the provided data at all**:

| Column | Spec description | Recoverable from 48×48 stills? |
|---|---|---|
| `Facial_Emotion_Variance` | "Variance of facial expressions from video (HOG-like)" | ✅ **Yes** — HOG descriptor variance |
| `Smile_Intensity` | "Average smile intensity extracted from video" | ⚠ Proxy only — face CNN happy-class probability |
| `Eye_Blink_Rate` | "Number of blinks per minute from video" | ❌ **No** — requires temporal frames |
| `Head_Motion_Index` | "Head movement index from video frames" | ❌ **No** — requires temporal frames |

The facial dataset is static, automatically registered, 48×48 grayscale stills. You cannot count blinks in a single frame, and registration has removed head motion by construction. So face matching runs on **one** feature where audio matching runs on four.

This is a real asymmetry and it has a consequence worth predicting out loud: **image match distances will be less discriminative than audio match distances**, so if the matched-vs-random ablation (§2.8) shows a smaller gain on the face branch than the voice branch, that is the expected result, not a bug.

Pre-empt the obvious question — *"why did you only match faces on one feature?"* — on the limitations slide. A team that names the gap reads as more competent than one that pads the feature list with numbers it invented.

`Smile_Intensity` needs a trained classifier, which doesn't exist yet — so **do not block on it.** Match images on `Facial_Emotion_Variance` alone in the first pass. If you have spare time at hour 12, add smile as the face CNN's happy-class probability and re-run the match as a refinement experiment.

## 2.6 The matching algorithm

```python
from scipy.spatial.distance import cdist

AUDIO_COLS = ['MFCC_Mean','MFCC_Variance','Pitch_Mean','Speech_Rate']
IMAGE_COLS = ['Facial_Emotion_Variance']

def match(csv_df, pool_df, cols, split, tiebreak=None):
    """Greedy nearest-neighbour with per-split capacity caps, within stress class."""
    pairs = {}
    for cls in ['Healthy','Mild_Stress','Moderate_Stress','Severe_Stress']:
        C = csv_df[(csv_df.Mental_Health_Status == cls) & (csv_df.split == split)]
        P = pool_df[(pool_df.stress == cls) & (pool_df.split == split)]
        assert len(P) > 0, f'empty pool: {cls}/{split}'

        # z-score BOTH sides using the CSV's mean/std so the spaces are comparable
        mu, sd = C[cols].mean(), C[cols].std().replace(0, 1)
        Cz = ((C[cols] - mu) / sd).values
        Pz = ((P[cols] - mu) / sd).values

        D = cdist(Cz, Pz, 'euclidean')

        # tie-breaker: nudge high-arousal participants toward strong-intensity clips
        if tiebreak is not None:
            D = D + 0.05 * tiebreak(C, P)

        cap  = int(np.ceil(len(C) / len(P)))       # computed PER SPLIT, per class
        used = np.zeros(len(P), dtype=int)

        for i in np.argsort(D.min(axis=1)):        # hardest-to-match first
            order = np.argsort(D[i])
            j = next(k for k in order if used[k] < cap)
            used[j] += 1
            pairs[C.index[i]] = P.index[j]
    return pairs
```

Four details that matter:

**1. Z-score using the CSV's statistics for both sides.** The two feature distributions have different scales; anchoring on the CSV puts the pool into participant space.

**2. Capacity caps — computed per split, per class.** v1 quoted a single global cap of ~6 for Severe. That number is wrong in both directions once splits are applied. The real arithmetic, assuming a roughly balanced CSV (1000 per class → 700 train / 150 val / 150 test):

| Class | Split | Participants | Audio pool | Audio cap | Image pool | Image cap |
|---|---|---|---|---|---|---|
| Healthy | train | 700 | 360 | 2 | ~9,744 | 1 |
| Mild | train | 700 | 288 | 3 | ~6,401 | 1 |
| Moderate | train | 700 | 288 | 3 | ~3,626 | 1 |
| **Severe** | **train** | **700** | **144** | **5** | ~3,196 | 1 |
| Healthy | test | 150 | 60 | 3 | ~1,218 | 1 |
| Mild | test | 150 | 48 | 4 | ~800 | 1 |
| Moderate | test | 150 | 48 | 4 | ~453 | 1 |
| **Severe** | **test** | **150** | **24** | **7** | ~400 | 1 |

Two conclusions to put on a slide:

- **Audio is the only capacity-constrained modality.** Every image cap is 1 — the face pool exceeds the participant count by an order of magnitude in every cell, so image matching is effectively an injection into a rich pool and the matched-vs-random contrast there is a pure quality comparison. All the reuse pressure is on voice.
- **The test Severe cell is the weak point: 150 participants drawn from 24 unique voices.** Your Severe-class test metrics rest on 24 distinct speakers. State this in limitations before anyone asks, and quote the effective sample size rather than the nominal one.

Recompute the whole table from the actual CSV balance in §1.4 — do not carry these numbers forward if the classes turn out skewed.

**3. Match within split.** Train CSV rows pair only with train audio and train images. This is the leakage firewall — get it wrong and every number you report is fiction. Assert it (Part 12).

**4. Intensity tie-breaker (optional, 10 min).** Rank participants within a class by an arousal composite (e.g. z(`Heart_Rate_BPM`) − z(`HRV_Index`) + z(`GSR_Level`)) and clips by their intensity field, then add a small penalty for rank mismatch. Weight it low (0.05) so it only breaks near-ties and never overrides the measured-feature match. Report it as a refinement, not a headline.

Optional upgrade if you have slack: replace greedy with `scipy.optimize.linear_sum_assignment` on a replicated pool (each item duplicated `cap` times). Globally optimal, runs in seconds at this scale, one extra line in the methods slide.

## 2.7 Manifest schema — now carrying the conflict flag

`data/manifest_{random|matched}.csv`:

```
participant_id, csv_index, split, audio_path, image_path,
stress_class, stress_ordinal, depression, anxiety, stress_score,
audio_emotion, image_emotion, audio_intensity, actor_id, actor_sex,
match_dist_audio, match_dist_image,
mapping_conflict,                      # ← new
<18 tabular feature columns>
```

```python
CONFLICT_EMOTIONS = {'angry', 'disgust'}

manifest['mapping_conflict'] = (
    manifest.audio_emotion.isin(CONFLICT_EMOTIONS) |
    manifest.image_emotion.isin(CONFLICT_EMOTIONS)
)
```

Carrying `audio_emotion` and `image_emotion` through to the manifest costs nothing and buys you every downstream analysis in §5.1. Carrying `actor_id` lets you assert speaker disjointness in one line.

Everything downstream reads only this file. Freeze it by hour 3 and treat it as an API — teammates should never touch raw data paths again.

## 2.8 The pairing ablation — the slide that wins criterion 3

Build **both** manifests and train the identical model on each:

| Pairing method | Accuracy | Macro F1 | QWK | Dep MAE | Anx MAE | Str MAE |
|---|---|---|---|---|---|---|
| Random within class (baseline everyone uses) | | | | | | |
| **Feature-matched (ours)** | | | | | | |

Two training runs, same code, one config flag. You will be the only team that ran an experiment on their own methodology instead of asserting it was reasonable.

## 2.9 The honesty checks — prepare these answers before you're asked

**Q: "Doesn't matching on MFCC mean the audio branch just relearns a tabular column?"**
Yes, partially. Which is why we report tabular-only as a baseline — if fusion beats it, the audio branch contributes information beyond the four matched summary statistics (spectro-temporal structure the four numbers don't capture). Show the number.

**Q: "Isn't pairing by label leaking the answer?"**
The label determines the candidate pool, not the assignment within it. And it's the only construction that makes participant-level fusion possible with disjoint sources. Crucially, this inflates absolute accuracy relative to real cohort data — which is why our conclusions rest on the *relative* comparison between fusion strategies, all of which share the same inflation.

**Q: "Are these real participants?"**
No. These are synthetic participants constructed to simulate multimodal acquisition. We're demonstrating the fusion architecture, not making clinical claims.

**Q: "Did you need the pairing at all?"** ← new, and the one you're most likely to be asked
No — and we tested that. Because the committee supplied per-modality emotion→stress mappings, each modality can be trained independently and its stress posterior combined at decision level with no pairing anywhere in training. That's row 6 of our ablation table. Pairing earns its place only where decision-level fusion structurally cannot go: the regression objective, per-participant explanations, and the concordance analysis. See §4.9.

**Q: "Why do angry and disgust map inconsistently?"** ← new
That's the committee's specification, not ours, and we treat it as a finding rather than a footnote — it's confined entirely to the two highest-severity classes and it's an exact inversion. We flag every affected pair in the manifest and report the concordance analysis with those rows segmented out, because otherwise our disagreement metric would be measuring the annotation scheme rather than the participant. See slide 8.

**Put all five on the "Limitations" slide.** A team that names its own weaknesses reads as more competent than one that hides them, and it converts hostile questions into confirmations.

---

# Part 3 — Per-modality preprocessing (criterion 2, 7 marks)

## 3.1 Tabular

- Median-impute nulls; clip physiologically impossible values (record what you clipped).
- `StandardScaler` fit on **train only**, persisted with joblib.
- **LightGBM baseline** — classification + 3 regressors. This is your safety submission, running by hour 2.
- **MLP encoder** for fusion: `18 → 64 → 64`, BatchNorm, dropout 0.3, GELU. LightGBM isn't differentiable so it can't live inside the fusion model; keep both — LightGBM for the SHAP story and baseline table, MLP for the joint network.

## 3.2 Audio

```
load 16 kHz mono → trim silence (top_db=30) → pad/crop to 3.0 s
→ log-mel: n_fft=1024, hop=256, n_mels=128 → (128, 188)
→ per-utterance mean/var normalise
```

**Augmentation (train only, essential given 144 Severe clips in train):**

| Technique | Setting | Why |
|---|---|---|
| Time shift | ±0.3 s roll | Onset invariance |
| Pitch shift | ±2 semitones | Speaker-timbre invariance |
| Additive noise | SNR 15–30 dB | Robustness |
| Gain jitter | ±6 dB | Recording-level invariance |
| SpecAugment | 2 freq masks (≤16 bins), 2 time masks (≤24 frames) | Regularisation |

Apply augmentation more aggressively to Severe/Moderate to partially counter the imbalance. Note that pitch shift interacts with `Pitch_Mean`, which you matched on — augment the *training* audio freely, but compute match features on the **un-augmented** signal. Matching happens once, before training; do not let the two stages touch.

**Encoder:** 4× (Conv2d → BN → GELU → MaxPool), channels 32/64/128/128, then adaptive-avg-pool → 128-d.

**Optional aux target (free, §2.4):** a second binary head predicting emotional intensity (normal/strong). Two gradient signals from one clip.

**Fallback (build this *first*, it takes 15 minutes):** 40 MFCCs + Δ + ΔΔ + pitch + ZCR + RMS + chroma → mean/std/min/max → ~180-d vector → LightGBM. If the CNN misbehaves at hour 10, you swap this in and lose almost nothing. Never be in a position where a failing audio CNN blocks the fusion model.

## 3.3 Image

- Cache all 28,709 as a single `uint8` `.npy` (~66 MB). Re-reading JPEGs every epoch silently costs an hour over a full build.
- Normalise to [0,1], then standardise with dataset mean/std.
- **Augment:** horizontal flip (p=0.5), rotation ±10°, translate ±10%, zoom ±10%, random erasing (p=0.25). **No vertical flip** — faces aren't upside down.
- **Class weighting:** Disgust has 436 samples vs Happy's 7,215. Use `WeightedRandomSampler` or inverse-frequency loss weights at the 7-class level *before* mapping to stress, so the encoder learns disgust at all. This matters doubly now: disgust is one of the two conflict emotions, so a face encoder that can't recognise it will corrupt the conflict analysis as well as the Moderate class.
- **Encoder:** 4 conv blocks (32/64/128/256), BN, dropout 0.3, GAP → 128-d. Expect ~62–67% on 7-class FER2013; on the mapped 4-class problem, meaningfully better.
- **Keep the 7-class head alongside the 4-class one.** You need per-image emotion predictions, not just stress predictions, to populate `image_emotion` for unseen inputs in the Streamlit app and to compute the conflict flag at inference time.
- Upgrade path if it plateaus: ResNet18, resize 48→96, repeat grayscale to 3 channels, freeze early layers.

---

# Part 4 — Model architecture

## 4.1 Overall shape

```
face  → CNN  → 128-d ─┐               ┌→ aux head (4-class)   [ablation + concordance + decision-level, free]
voice → CNN  → 128-d ─┼→ gated fusion ─┼→ trunk 320→256→128 ─┬→ classification head
tabular → MLP → 64-d ─┘               └→ aux heads            └→ 3 regression heads
```

## 4.2 Auxiliary heads — the highest-leverage 10 lines in the build

Attach a small classifier to each encoder output, trained with the same loss at weight 0.3.

**Four** payoffs from one change:

1. **Deep supervision** — each encoder gets direct gradient, so weak modalities still learn instead of being drowned out by the tabular branch.
2. **Single-modality ablation rows for free** — no separate training runs needed for your ablation table.
3. **Per-modality predictions at inference**, which is what the concordance score (§5.1) is built from.
4. **Decision-level fusion for free** (§4.9) — the three posteriors are exactly what that baseline combines.

Do this early. It changes what's possible later.

## 4.3 Ordinal-aware classification loss (criteria 4 + 6)

Healthy → Mild → Moderate → Severe is an **ordered** scale. Plain cross-entropy treats "predicted Healthy for a Severe case" as exactly as wrong as "predicted Moderate" — clinically absurd, since one is a missed escalation and the other is a rounding error.

**Tier 1 — expectation regulariser (recommended; use this).**

```python
def ordinal_ce(logits, y, class_w, beta=0.3):
    p   = F.softmax(logits, dim=1)
    lev = torch.arange(4, device=logits.device, dtype=p.dtype)
    exp = (p * lev).sum(1)                     # expected ordinal level
    return F.cross_entropy(logits, y, weight=class_w) \
         + beta * F.l1_loss(exp, y.to(p.dtype))
```

Three lines. Keeps a valid softmax, so **ROC-AUC, per-class precision/recall and the confusion matrix all stay trivially computable** — which matters because the metrics document explicitly demands all of them. Tune β on validation (try 0.1 / 0.3 / 0.5).

**Tier 2 — CORAL head** (only if Tier 1 is running well and you have hours to spare). K−1 binary "is y > k" tasks sharing one weight vector with independent biases; guarantees rank-monotonic predictions. Convert cumulative probabilities back to class probabilities via successive differences, then clip at 1e-6 and renormalise before computing ROC-AUC.

Why Tier 1 is the recommendation: CORAL's outputs need careful conversion to satisfy the required metrics, and debugging that at hour 16 is a bad trade. The expectation regulariser captures ~80% of the benefit at ~5% of the risk.

**Asymmetric cost:** under-calling severity should cost more than over-calling. Scale class weights so `Severe` and `Moderate` carry ~1.5× their inverse-frequency weight. A false alarm wastes a screening conversation; a miss means nothing happens at all.

## 4.4 Regression heads

Normalise each target to [0,1] by its stated maximum (Depression 34 / Anxiety 24 / Stress 39) before training — otherwise Stress_Score's larger range dominates the gradient and the other two are undertrained. **Invert before reporting**, so MAE and RMSE come out in original questionnaire units as the metrics document requires.

Use Huber (`SmoothL1`) rather than MSE — more robust to the handful of extreme scorers.

## 4.5 Joint loss

```python
L = ordinal_ce(cls_logits, y, class_w)                    \
  + LAMBDA_REG * huber(reg_pred, reg_target)              \
  + 0.3 * sum(ordinal_ce(a, y, class_w) for a in aux)
```

Tune `LAMBDA_REG` ∈ {0.3, 0.5, 1.0} on validation. Multi-task learning here is genuinely motivated, not decorative — the questionnaire scores and the stress class are two readings of one latent state, so each regularises the other. It is also Objective 2 of the problem statement, which explicitly asks for a *multi-output* regression framework rather than three separate models. Say that.

## 4.6 Gated fusion (criteria 3 + 6)

```python
class GatedFusion(nn.Module):
    def __init__(self, dims=(128,128,64), h=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, h) for d in dims])
        self.gate = nn.Sequential(nn.Linear(h*3, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, embs):
        z = [p(e) for p, e in zip(self.proj, embs)]        # all → h
        g = torch.softmax(self.gate(torch.cat(z, 1)), 1)   # (B,3) per-participant
        return torch.cat([z[i] * g[:, i:i+1] for i in range(3)], 1), g
```

Returning `g` is the point. It gives you a **per-participant modality-contribution readout** — "for this person the model leaned 61% on voice" — which becomes both an explainability artefact (criterion 7) and the most memorable visual in your deck.

*Tier 3, only with hours spare:* cross-modal attention over the three embeddings as a 3-token sequence. Diminishing returns; skip unless everything else is finished.

## 4.7 Modality dropout

Randomly zero an entire modality's embedding with p=0.15 during training (never all three at once).

Three payoffs again: regularisation, robustness, and — critically — **your Streamlit demo still works when a judge uploads only a photo.** Fifteen minutes of work that prevents the most likely live-demo failure.

## 4.8 Training config

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| Schedule | Cosine anneal, 3-epoch linear warmup |
| Batch | 64 |
| Epochs | Pretrain encoders 20 each → joint fine-tune 30 |
| Early stop | Patience 7 on val macro-F1 |
| Mixed precision | Yes if GPU |
| Seeds | Fix and log 3 seeds; report mean ± std on the headline row |

Pretrain each encoder standalone on its own emotion task first, then fine-tune jointly at 0.1× lr for the encoders. Training all three from scratch inside the fusion model converges slowly and unevenly.

## 4.9 ⚠ Decision-level fusion — the baseline you must not skip

**Why this is now in the plan.** The committee supplied an emotion→stress mapping for *each* dataset independently. That means every modality can produce a 4-class stress posterior on its own, trained entirely on its own data, with **no pairing anywhere in training**. A judge will notice this and ask why you built a pairing engine at all. Answer it with a number, not a paragraph.

```python
def decision_level_fusion(p_face, p_voice, p_tab, w=None):
    """Each p_* is (N,4) and sums to 1. No paired training data required —
       the three unimodal models never see a manifest."""
    w = np.array([1/3, 1/3, 1/3]) if w is None else np.asarray(w)
    P = w[0]*p_face + w[1]*p_voice + w[2]*p_tab
    return P / P.sum(1, keepdims=True)
```

Three variants, all cheap:

1. **Uniform average** — zero parameters, ~10 minutes.
2. **Validation-tuned weights** — grid search `w` on the simplex at 0.05 resolution, ~5 minutes of compute.
3. **Stacked logistic regression** on the concatenated 12-d posterior vector — fits in seconds.

Pairing is still required to *evaluate* these at participant level (you need a paired test row to score against), but it never enters training. That's a materially lower-risk methodology and you should say so.

**Then state precisely what decision-level fusion cannot do**, because that is the argument for your joint model:

| Capability | Decision-level | Joint gated fusion |
|---|---|---|
| 4-class classification (Objective 1) | ✅ | ✅ |
| Depression/Anxiety/Stress regression (Objective 2) | ❌ — face and voice sets carry no questionnaire scores; regression collapses to tabular-only | ✅ |
| Per-participant modality gate weights (Objective 3) | ⚠ fixed global weights only | ✅ per-participant |
| Cross-modal representation learning | ❌ | ✅ |
| Graceful degradation with a missing modality | ✅ | ✅ (via modality dropout) |

**The line for the slide:** *decision-level fusion satisfies Objective 1 without any pairing, so we report it as an honest baseline — but Objectives 2 and 3 require joint representations, which is why the alignment engine exists.* That reframes your riskiest design choice as a deliberate, justified one.

If the joint model *fails to beat* decision-level fusion on classification, report that too. It is a legitimate result and hiding it is the only way to turn it into a bad one.

---

# Part 5 — Standout features

## 5.1 Modality concordance score — now with conflict segmentation

Everyone fuses. Nobody asks what happens when channels **contradict** each other.

```python
def concordance(p_face, p_voice, p_tab):
    """1.0 = perfect agreement, 0.0 = maximal disagreement (3 stress levels apart)."""
    lev = np.arange(4)
    e = [ (p * lev).sum(-1) for p in (p_face, p_voice, p_tab) ]  # expected level each
    spread = np.max(e, 0) - np.min(e, 0)
    return 1.0 - spread / 3.0, spread
```

Free, given the aux heads from §4.2.

**Why it matters clinically:** masked or suppressed affect is well documented — people conceal distress facially far more readily than vocally. A participant whose face reads Healthy while their voice reads Moderate is precisely the case a screening tool must not wave through.

### ⚠ The correction v1 needed

Under the committee's mappings, **every Severe participant is paired with an angry face and a disgust voice** — two different emotions the mapping tables assign to the same stress class in opposite directions (§1.3). The face and voice branches will therefore disagree on those participants *by construction*. Report raw concordance and you are reporting the annotation scheme.

**The fix — segment, don't discard:**

```python
clean    = manifest[~manifest.mapping_conflict]     # angry/disgust in neither modality
conflict = manifest[ manifest.mapping_conflict]

for name, subset in [('all', manifest), ('clean', clean), ('conflict', conflict)]:
    report_concordance_bands(subset)
```

Report all three. The headline claim uses **`clean`**; the other two go in the appendix slide. Expected pattern, which you should predict before you look: mean concordance materially lower in `conflict` than in `clean`. If that shows up, you have quantified the cost of the specification's inconsistency — a genuine methodological contribution, and one no other team will have.

Note the practical consequence: `clean` excludes the entire Severe class, so concordance-based routing is only validated on Healthy/Mild/Moderate. Say that rather than letting a judge find it.

**Turn it into system behaviour, not just a number.** Define a routing policy:

| Concordance | Action |
|---|---|
| ≥ 0.75 | Report prediction with confidence |
| 0.45 – 0.75 | Report with a "modality disagreement" caveat |
| < 0.45 | **Flag for human review** — withhold the confident label |

Then report, on the `clean` test subset: what fraction of participants land in each band, and — the interesting number — **is accuracy lower in the low-concordance band?** If yes, you've shown the score is a genuine reliability signal, not decoration. That's a real finding, presentable in one sentence.

## 5.2 Uncertainty quantification via MC dropout

```python
def predict_with_uncertainty(model, batch, T=30):
    model.train()                       # keep dropout ON
    with torch.no_grad():
        outs = [model(batch) for _ in range(T)]
    cls = torch.stack([F.softmax(o.cls, 1) for o in outs])   # (T,B,4)
    reg = torch.stack([o.reg for o in outs])                 # (T,B,3)
    model.eval()
    return cls.mean(0), cls.std(0), reg.mean(0), reg.std(0)
```

Careful: `model.train()` also puts BatchNorm in batch-statistics mode. Either use only dropout for MC sampling (set BN layers back to eval explicitly), or swap BN for GroupNorm/LayerNorm in the trunk. Do this:

```python
for m in model.modules():
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.eval()
```

**Output becomes "Depression 21 ± 4" instead of "Depression 21.3."** In a mental health context that difference is the whole argument. Add an abstention band — if predictive entropy exceeds a validation-tuned threshold, the model declines to call it.

Then plot **accuracy vs. confidence** on the test set. If accuracy rises monotonically with confidence, your uncertainty estimates are meaningful. One chart, high credibility.

Bonus, ~20 minutes: **Expected Calibration Error** + a reliability diagram. Well-calibrated probabilities are a real contribution in a clinical setting, and it's `sklearn` plus a bar chart.

## 5.3 Plain-language clinical report card (criterion 7 — the memorable artefact)

Everyone pastes a SHAP beeswarm on a slide. Instead, generate this per participant from numbers you already have:

```
─────────────────────────────────────────────────
PARTICIPANT #1847                    Screening report
─────────────────────────────────────────────────
ASSESSMENT      Moderate stress indicated
CONFIDENCE      71%  (moderate)

SEVERITY        Depression  21 ± 4   (0–34)
                Anxiety     14 ± 3   (0–24)
                Stress      26 ± 5   (0–39)

PRIMARY DRIVERS
  ↓ Heart-rate variability      12th percentile
  ↓ Sleep quality               2/5
  ↓ Social engagement           2/5

MODALITY EVIDENCE
  Physiological/behavioural     58%  → Moderate
  Voice                         31%  → Moderate
  Facial                        11%  → Healthy

CONCORDANCE     0.41  ⚠ LOW
                Facial signal diverges from voice and
                physiology. Pattern consistent with
                masked affect — recommend human review.

WHAT WOULD CHANGE THIS
  Sleep quality 2 → 4      predicted stress −6.2
  HRV +1 SD                predicted stress −4.1
─────────────────────────────────────────────────
```

It's template formatting over SHAP values, gate weights, MC-dropout std and counterfactual re-predictions — **all already computed.** Under an hour of work, and it is the only output in the room that looks like something a clinician could act on.

**One addition:** when the participant's paired emotions trip the conflict flag, suppress the masked-affect wording and substitute a neutral note ("modality disagreement present; note that this severity class has known annotation inconsistency"). Never let the demo assert a clinical interpretation of a labelling artefact — that is exactly the claim a sharp judge will take apart.

## 5.4 Counterfactuals ("what would change this")

```python
def counterfactual(model, row, feature, new_value, scaler):
    base = predict(model, row)
    mod  = row.copy(); mod[feature] = new_value
    return predict(model, mod) - base
```

Sweep each of the 18 tabular features across its interquartile range, rank by effect size, report the top 3 **actionable** ones (sleep, social engagement, app usage, idle time — things a person can change; not heart rate or skin temperature).

People don't want to know their score. They want to know what moves it. This is the difference between a model and a tool.

## 5.5 Summary of what makes you different

| Feature | Cost | Rubric impact |
|---|---|---|
| Feature-matched alignment + ablation | 2 h | **3 (8 marks)**, 6 |
| Decision-level fusion baseline | 30 min | **3**, 1, 6 |
| Mapping-conflict audit + segmented concordance | 45 min | 1, 6, 7 |
| Ordinal loss + QWK | 45 min | 4, 6 |
| Auxiliary heads (free ablation + concordance + decision fusion) | 15 min | 3, 4, 6 |
| Concordance score + routing policy | 45 min | 3, 6, 7 |
| MC-dropout uncertainty + calibration | 1 h | 5, 6, 7 |
| Modality dropout | 15 min | 3, 6, 8 |
| Clinical report card | 1 h | **7 (4 marks)**, 9 |
| Counterfactuals | 30 min | 7, 9 |

Roughly 7.75 hours for every differentiator on the list.

---

# Part 6 — Evaluation protocol (criteria 4 + 5, 14 marks)

The metrics document names specific metrics. Implement **all of them**, in one module, emitting both JSON and a markdown table. Missing a named metric is the cheapest possible way to lose marks.

**Coverage check against the committee's list — all 13 are already covered below.** Classification: accuracy, precision, recall/sensitivity, F1, macro F1, weighted F1, ROC-AUC, confusion matrix. Regression: MAE, MSE, RMSE, R², explained variance. QWK and NRMSE are **extras**, presented as additional to the required set — never as substitutes.

## 6.1 Required classification metrics

```python
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, roc_auc_score, confusion_matrix,
                             cohen_kappa_score, classification_report)

def classification_metrics(y_true, y_pred, y_proba, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(4),
                                                 zero_division=0)
    return {
      'accuracy':            accuracy_score(y_true, y_pred),
      'precision_per_class': dict(zip(labels, p)),      # required: per class
      'recall_per_class':    dict(zip(labels, r)),      # required: sensitivity
      'f1_per_class':        dict(zip(labels, f)),
      'support_per_class':   dict(zip(labels, s.tolist())),
      'macro_f1':            f1_score(y_true, y_pred, average='macro'),
      'weighted_f1':         f1_score(y_true, y_pred, average='weighted'),
      'roc_auc_ovr_macro':   roc_auc_score(y_true, y_proba, multi_class='ovr',
                                           average='macro'),
      'roc_auc_ovr_weighted': roc_auc_score(y_true, y_proba, multi_class='ovr',
                                            average='weighted'),
      'confusion_matrix':    confusion_matrix(y_true, y_pred, labels=range(4)).tolist(),
      'qwk':                 cohen_kappa_score(y_true, y_pred, weights='quadratic'),  # EXTRA
      'report':              classification_report(y_true, y_pred, target_names=labels),
    }
```

Gotchas:
- `roc_auc_score` with `multi_class='ovr'` needs probabilities that sum to 1 and **all four classes present** in `y_true`. Guard the stratified split so no class is empty in test.
- Plot ROC curves per class (one-vs-rest) — four curves on one axis, plus macro-average. Better than a bare number on a slide.
- Confusion matrix: plot **both** raw counts and row-normalised. Row-normalised is what reveals which class is being systematically confused — and given §1.3, watch the Moderate↔Severe cell specifically. Off-diagonal mass there may be the mapping conflict rather than model error, and you should be the one who says so.

**QWK is your extra.** It's the standard measure of ordinal agreement and it directly rewards the ordinal loss from §4.3. Introduce it in one line: "because the classes are ordered, we also report quadratic weighted kappa, the standard metric for ordinal agreement."

## 6.2 Required regression metrics — per target, never averaged

The metrics document says "for each target." Three separate rows.

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, explained_variance_score)

def regression_metrics(y_true, y_pred, names=('Depression','Anxiety','Stress'),
                       maxima=(34, 24, 39)):
    out = {}
    for i, n in enumerate(names):
        t, p = y_true[:, i], y_pred[:, i]
        mse  = mean_squared_error(t, p)
        out[n] = {
          'MAE':  mean_absolute_error(t, p),
          'MSE':  mse,
          'RMSE': np.sqrt(mse),
          'R2':   r2_score(t, p),
          'ExplainedVariance': explained_variance_score(t, p),
          'NRMSE_pct': 100 * np.sqrt(mse) / maxima[i],   # EXTRA: cross-target comparable
        }
    return out
```

**Report in original units**, so invert the [0,1] normalisation first. An MAE of 0.12 is meaningless; an MAE of 4.1 points on a 0–34 scale is interpretable.

`NRMSE_pct` is a small extra that lets you say "our anxiety predictions are proportionally the most accurate" — the three scales differ, so raw RMSE isn't comparable across targets.

Also produce **predicted-vs-actual scatter plots** with the identity line, one per target. Instantly shows range compression (a very common failure — the model predicts everything near the mean and still gets a decent MAE).

## 6.3 The ablation table — build it as you go

Every row is a run you're doing anyway. Fill it in continuously; do not reconstruct it at hour 20.

| # | Model | Pairing in training? | Acc | Macro F1 | Weighted F1 | ROC-AUC | QWK | Dep MAE | Anx MAE | Str MAE | Mean R² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Tabular only (LightGBM) | n/a | | | | | | | | | |
| 2 | Tabular only (MLP) | n/a | | | | | | | | | |
| 3 | Audio only (aux head) | none | | | | | — | — | — | — | — |
| 4 | Image only (aux head) | none | | | | | — | — | — | — | — |
| 5 | **Decision-level fusion, uniform** | **none** | | | | | | — | — | — | — |
| 6 | **Decision-level fusion, val-tuned weights** | **none** | | | | | | — | — | — | — |
| 7 | Concat fusion, random pairing | random | | | | | | | | | |
| 8 | Concat fusion, matched pairing | matched | | | | | | | | | |
| 9 | Gated fusion, matched pairing | matched | | | | | | | | | |
| 10 | **+ ordinal loss + modality dropout (final)** | matched | | | | | | | | | |

Rows 5–6 are new in v2 and they are load-bearing: they are the only rows in the table where **no pairing enters training at all**, which is what makes the "did you need the alignment engine?" question answerable with evidence. The em-dashes in their regression columns are the argument for rows 7–10 — decision-level fusion structurally cannot address Objective 2.

Report each row on the **`clean` (non-conflict) test subset** as the headline, with the all-rows numbers alongside. Same table, two columns of context.

This single table answers criteria 1, 3, 4, 5 and 6 simultaneously. It is the most information-dense slide in your deck.

## 6.4 Evaluation hygiene

- Test set touched **once**, at the end. All tuning on validation.
- Report mean ± std over 3 seeds on the final row.
- Log every run to `results/{run_id}/metrics.json` + `config.yaml`.
- Save the confusion matrix, ROC curves and scatter plots as PNGs to `figures/` as they're produced — don't regenerate them under time pressure at hour 19.
- **Report effective sample size, not just nominal.** Test Severe = 150 participants drawn from 24 unique voices and ~400 unique faces (§2.6). Quote both numbers.

---

# Part 7 — Explainability (criterion 7, 4 marks)

Cheapest marks on the sheet. Never let this get cut. This is Objective 3 of the problem statement, which asks you to *quantify the contribution of each indicator* — so lead with the gate weights and SHAP magnitudes, not with pictures.

| Artefact | Method | Time |
|---|---|---|
| Tabular attribution | TreeSHAP on LightGBM → beeswarm + per-participant waterfall | 20 min |
| Facial attribution | Grad-CAM on last conv block, overlaid on 4 faces (one per class) | 30 min |
| Vocal attribution | Grad-CAM on the mel-spectrogram → highlights which time-frequency regions drove it | 30 min |
| Modality attribution | Gate weights `g` from §4.6, stacked bar per participant | 10 min |
| Report card | §5.3 | 60 min |
| Counterfactuals | §5.4 | 30 min |

**Frame everything clinically.** Not "SHAP shows HRV_Index has high importance" but:

> The model's dominant drivers for severe stress are heart-rate variability suppression and poor sleep quality, which aligns with established stress physiology — HRV is a well-known marker of autonomic dysregulation under chronic stress.

That framing is the difference between 2 and 4 marks on this criterion. Same numbers, different sentence.

The spectrogram Grad-CAM is worth the extra half hour — almost nobody will do it, and "the model attends to the low-frequency energy in the first 400 ms of the utterance" sounds like research.

---

# Part 8 — Repo structure and team contract

```
project/
├── data/
│   ├── manifest_random.csv
│   └── manifest_matched.csv         # both carry mapping_conflict
├── splits/                          # frozen, hour 1
├── src/
│   ├── audit.py                     # Part 1, incl. mapping-conflict table
│   ├── match_features.py            # §2.4–2.5
│   ├── build_manifest.py            # §2.6–2.7
│   ├── datasets.py                  # reads manifest ONLY
│   ├── encoders.py                  # AudioCNN, FaceCNN, TabularMLP
│   ├── fusion.py                    # GatedFusion, MultiTaskModel
│   ├── decision_fusion.py           # §4.9  ← new
│   ├── losses.py                    # ordinal_ce, joint loss
│   ├── train.py
│   ├── evaluate.py                  # Part 6 — all required metrics
│   ├── explain.py                   # Part 7
│   └── uncertainty.py               # §5.2, §5.1 concordance
├── app/streamlit_app.py
├── results/, figures/, reports/
└── configs/{baseline,random,matched,decision,final}.yaml
```

## The interface contract — write this in hour 1

Every encoder obeys the same signature so three people can work in parallel without blocking:

```python
class Encoder(nn.Module):
    out_dim: int                                    # 128, 128, 64
    def forward(self, x) -> torch.Tensor:           # (B, out_dim)
        ...
```

The fusion owner writes this contract first with **dummy encoders that return random tensors of the right shape**. Then the full training loop is testable at hour 2, before a single real encoder exists. This is what stops the classic hackathon deadlock where nobody can integrate until everyone is finished.

## Roles (4 people)

| Person | Owns |
|---|---|
| **A — Audio** | `match_features.py` audio half, filename parsing + intensity, mel pipeline, augmentation, AudioCNN, MFCC fallback |
| **B — Vision** | npy cache, HOG features, augmentation, FaceCNN (7-class + 4-class heads), Grad-CAM |
| **C — Fusion** | Splits, manifest, matching algorithm, conflict flag, LightGBM baseline, TabularMLP, fusion, losses, training loop |
| **D — Eval & product** | `evaluate.py`, decision-level fusion, uncertainty, concordance + conflict segmentation, report card, Streamlit, slides, ablation table |

**D is the integration owner** and writes the contract in hour 1. D also owns the ablation table and chases A/B/C for numbers — someone must own "is the table full?" or it won't be.

**Solo?** Order: audit + conflict table → manifest → LightGBM → face CNN → decision-level fusion → joint fusion → evaluation → report card → Streamlit → audio (MFCC fallback only). Skip the audio CNN entirely. Note that decision-level fusion moved *up* the solo list — it is 30 minutes and it gives you a defensible multimodal result before the joint model exists.

---

# Part 9 — Prototype (criterion 8, 4 marks)

Streamlit. Not React.

**Layout:** upload face image · upload wav · 18 sliders pre-filled with dataset medians · Predict button.

**Output:** predicted class + confidence · three scores with ± uncertainty bands · gate-weight bar chart · concordance gauge with routing verdict · Grad-CAM overlay · SHAP waterfall · counterfactual table · the full report card, downloadable as text.

**Non-negotiables:**
- `@st.cache_resource` on model loading, or every interaction takes 30 seconds.
- **Graceful degradation** — if no image or no audio is uploaded, zero that modality's embedding and note it in the output. Modality dropout (§4.7) is what makes this work. This is your insurance against the single most common live-demo failure.
- **Conflict-aware messaging** — run the 7-class face head and the 8-class voice head at inference; if either returns angry or disgust, show the neutral disagreement wording from §5.3 instead of the masked-affect interpretation.
- **Four pre-loaded demo participants**, reachable in one click: one clear Healthy, one Severe, one deliberately low-concordance, and one conflict-flagged case so you can *demonstrate* the honesty behaviour rather than just claiming it. If uploads fail on venue wifi, you still demo.
- Test it **on the presentation laptop, on the venue network, before you present.**

---

# Part 10 — Presentation (criterion 9, 3 marks)

| # | Slide | Content |
|---|---|---|
| 1 | Problem | Screening not diagnosis. Questionnaires are infrequent and self-reported; devices observe continuously. |
| 2 | Why multimodal | Each channel alone is ambiguous — flat voice could be a cold; frown could be concentration. Agreement is the signal. |
| 3 | **The alignment problem** | Three datasets, zero shared participants. Here's what everyone will do. Here's what we did instead. |
| 4 | **Feature-matched alignment** | The audio/face columns already in the CSV. `Speech_Rate = 6 / duration`, confirmed from the spec. Nearest-neighbour in measured feature space. Per-split capacity caps; audio is the only constrained pool. |
| 5 | **Did we need pairing?** | Decision-level fusion, no pairing in training, as an honest baseline — and the two objectives it structurally cannot reach. (§4.9) |
| 6 | Architecture | The system diagram, gated fusion, modality dropout, multi-task head. |
| 7 | Ordinal insight | Classes are ordered; CE ignores that; here's the loss and QWK. |
| 8 | **The mapping conflict** | Angry and disgust invert across the two committee tables, confined to the two highest classes. Here's how it corrupts a naive concordance metric, and here's our segmented analysis. |
| 9 | **Ablation table** | The full §6.3 table. Let it sit on screen for 20 seconds. |
| 10 | Concordance | Masked affect. The routing policy. Accuracy is lower in the low-concordance band (clean subset). |
| 11 | Explainability | Report card, full screen. Grad-CAM strip. Counterfactuals. |
| 12 | Live demo | Two minutes. Rehearsed. |
| 13 | **Limitations** | Synthetic pairing · acted emotions · mapping conflict · one of four face columns recoverable · 24 unique Severe test voices · no clinical validation · not a diagnostic tool. |
| 14 | Next steps | Real longitudinal cohort, temporal modelling (which would recover blink rate and head motion), on-device privacy-preserving inference. |

**Slides 3 and 8 are the moments.** Slide 3 sets up your method; slide 8 proves you read the specification more carefully than the people who wrote it. Land both clearly and don't rush slide 8 — it is the single most defensible novelty claim in the deck, because it is a finding about the data rather than a choice about the model.

The limitations slide is not modesty — it's a demonstration that you understand your own method's boundaries, and it converts the hardest questions into confirmations of what you already said.

---

# Part 11 — Schedule (24 h; run bold items only if you have 8)

| Hour | Milestone | Owner |
|---|---|---|
| 0:00 | **Download, verify counts, data audit** | All |
| 0:15 | **Email committee: rubric weights, held-out set, submission format** | D |
| 0:30 | **Mapping-conflict table + CSV class balance recorded** | C |
| 0:45 | **Freeze splits.** Actor-based for audio. Recompute capacity caps from real balance. | C |
| 1:00 | Interface contract + dummy encoders + training loop skeleton | C, D |
| 1:30 | **LightGBM tabular baseline → first full metrics table** ← safety net secured | C |
| 2:00 | Match-feature extraction running (audio + HOG); filename/intensity parsing | A, B |
| 3:00 | **Both manifests built and frozen** (random + matched), conflict flag populated | C |
| 3:30 | npy image cache; mel-spectrograms precomputed to disk | A, B |
| 6:00 | Encoders pretrained standalone; solo accuracies logged (ablation rows 3–4) | A, B |
| 7:00 | **Decision-level fusion, rows 5–6 filled** ← multimodal result exists, zero pairing risk | D |
| 8:00 | **First end-to-end fusion run. Ablation rows 1–8 filled.** | C |
| 10:00 | Gated fusion + aux heads + modality dropout | C |
| 12:00 | Ordinal loss; β and λ tuned on val | C |
| 13:00 | Pairing ablation run (random vs matched, identical config) | C |
| 14:00 | **MODEL FREEZE. No architecture changes past this line.** | All |
| 15:00 | Full evaluation, 3 seeds, all required metrics, all figures | D |
| 15:30 | **Conflict-segmented concordance analysis (all / clean / conflict)** | D |
| 16:00 | SHAP, Grad-CAM (face + spectrogram), gate charts | B, D |
| 17:00 | Uncertainty, concordance bands, counterfactuals, **report card** | D |
| 18:30 | **Streamlit app** (incl. conflict-aware messaging + 4 demo participants) | D |
| 20:00 | **Slides. Ablation table final. Rehearse demo twice.** | All |
| 22:00 | Buffer — something will break | All |

**Hard checkpoints:**
- **Hour 2:** a submittable result exists. If not, drop everything until it does.
- **Hour 7:** a *multimodal* result exists (decision-level). This is new in v2 and it is a much earlier safety net than hour 8's joint run.
- **Hour 8:** end-to-end fusion runs. If not, cut the audio CNN to the MFCC fallback immediately.
- **Hour 14:** freeze. Every hour after this is presentation value, not accuracy.

---

# Part 12 — Risk register

| Risk | Detection | Mitigation |
|---|---|---|
| Rubric weights differ from Part 0 | Committee reply | Ask at hour 0:15; re-allocate before hour 2 |
| Pairing before splitting | Suspiciously high accuracy (>95%) | Assert `set(train_audio) ∩ set(test_audio) == ∅` in `build_manifest.py` |
| Speaker leakage | Audio-only accuracy implausibly high | Actor-based split, asserted on `actor_id` in the manifest |
| **Concordance measures the mapping conflict, not the participant** | Mean concordance much lower in Severe than other classes | `mapping_conflict` flag; report all/clean/conflict separately (§5.1) |
| **Judge asks why pairing was needed at all** | — | Ablation rows 5–6 + the capability table in §4.9 |
| Capacity caps computed globally not per split | One clip assigned hundreds of times; audio branch overfits | Caps derived per split per class (§2.6); assert `used.max() <= cap` |
| Severe test metrics rest on 24 voices | Wide seed-to-seed variance on Severe recall | Report 3-seed ± std and effective sample size; state in limitations |
| Audio CNN won't converge | Val loss flat after 10 epochs | Swap to MFCC + LightGBM fallback (already built) |
| Severe class collapse | Recall ≈ 0 for Severe | Class weights + heavier augmentation + asymmetric cost |
| Disgust face class ignored (436 images) | Disgust recall ≈ 0 on the 7-class head | Weighted sampler at 7-class level *before* stress mapping |
| Range compression in regression | R² near 0 but MAE looks fine | Check predicted-vs-actual scatter; reduce λ, increase head capacity |
| ROC-AUC crashes | Exception on `multi_class='ovr'` | Guard: all 4 classes present in test; probabilities normalised |
| MC dropout breaks BN | Predictions shift wildly between passes | Explicitly `.eval()` all BatchNorm modules |
| Match features computed on augmented audio | Match distances drift between runs | Compute match features once, pre-training, on un-augmented signal |
| Image loading bottleneck | Epochs take minutes | npy cache (do it at hour 3, not hour 12) |
| Demo fails live | — | Pre-loaded samples + graceful degradation + test on venue wifi |
| Ablation table empty at hour 20 | — | D fills it continuously; it's a standing agenda item |

---

# Part 13 — Deliverables checklist

**Code**
- [ ] `data_audit.md` with all distributions **and the mapping-conflict table**
- [ ] Frozen splits, actor-based for audio; per-split capacity caps recorded
- [ ] Both manifests (random + matched), each carrying `mapping_conflict`
- [ ] Three encoders + gated fusion + aux heads
- [ ] `decision_fusion.py` — uniform and val-tuned variants
- [ ] Ordinal loss, modality dropout, MC dropout
- [ ] `evaluate.py` emitting every metric named in the committee's document
- [ ] Streamlit app with graceful degradation and conflict-aware messaging

**Results**
- [ ] Ablation table, all 10 rows populated
- [ ] Confusion matrix (raw + normalised), Moderate↔Severe cell commented on
- [ ] ROC curves, per class + macro
- [ ] Predicted-vs-actual scatter, all 3 targets
- [ ] Accuracy-vs-confidence curve
- [ ] Concordance band analysis × 3 subsets (all / clean / conflict)
- [ ] 3-seed mean ± std on the final row, with effective sample sizes

**Explainability**
- [ ] SHAP beeswarm + per-participant waterfall
- [ ] Grad-CAM, faces (4 classes) and spectrogram
- [ ] Gate-weight charts
- [ ] Report card, 3 example participants + 1 conflict-flagged case
- [ ] Counterfactual table

**Presentation**
- [ ] 14 slides per Part 10
- [ ] Demo rehearsed twice on the actual laptop
- [ ] Limitations slide written and owned
- [ ] Answers prepared for the five questions in §2.9

---

## The one-sentence version

Everyone will build the same three encoders and concatenate them; you win by **aligning participants in measured feature space and proving it helps**, by **showing you didn't need to and doing it anyway for reasons you can name**, by **finding the contradiction in the committee's own mapping tables before they point it out**, by **treating the four classes as the ordered scale they are**, by **reporting what your model doesn't know**, and by **outputting something a clinician could actually read** — roughly 7.75 hours of work that no other team will have thought to do.
