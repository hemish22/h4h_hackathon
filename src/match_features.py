"""Part 2.4-2.5 — matching-space feature extraction.

Audio: MFCC_Mean, MFCC_Variance, Pitch_Mean, Speech_Rate (=6/duration, exact per spec).
Image: Facial_Emotion_Variance via HOG descriptor variance.

Match features are computed ONCE on the un-augmented signal and cached to
data/audio_match.csv and data/image_match.csv. Never recompute on augmented data.
Run: python -m src.match_features [--limit-images N]
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

from . import config as C
from . import data_utils as D


# ------------------------------------------------------------------ audio
def audio_match_features(path: str) -> dict:
    import librosa
    y, sr = librosa.load(D.resolve(path), sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    dur = max(len(y) / sr, 1e-6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
    f0v = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
    return {
        "MFCC_Mean": float(mfcc.mean()),        # grand mean over 13 coeffs x frames (documented choice)
        "MFCC_Variance": float(mfcc.var()),
        "Pitch_Mean": float(np.median(f0v)) if len(f0v) else np.nan,
        "Speech_Rate": 6.0 / dur,               # exact: 6-word statements, spec-confirmed
    }


def build_audio_match(out=None) -> pd.DataFrame:
    audio = D.list_audio_files()
    feats = []
    n = len(audio)
    for i, row in enumerate(audio.itertuples(index=False)):
        f = audio_match_features(row.path)
        f["path"] = row.path
        feats.append(f)
        if (i + 1) % 100 == 0:
            print(f"  audio {i+1}/{n}")
    df = audio.merge(pd.DataFrame(feats), on="path")
    # impute NaN pitch with class median (192 severe too few to drop)
    df["Pitch_Mean"] = df.groupby("stress")["Pitch_Mean"].transform(
        lambda s: s.fillna(s.median()))
    df["Pitch_Mean"] = df["Pitch_Mean"].fillna(df["Pitch_Mean"].median())
    out = out or os.path.join(C.DATA_DIR, "audio_match.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    return df


# ------------------------------------------------------------------ image
def image_match_features(img48: np.ndarray) -> dict:
    from skimage.feature import hog
    desc = hog(img48, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), feature_vector=True)
    return {"Facial_Emotion_Variance": float(desc.var())}


def build_image_match(limit=None, out=None) -> pd.DataFrame:
    from PIL import Image
    img = D.list_image_files()
    if limit:
        # stratified subsample per emotion for speed (matching only needs a pool)
        img = img.groupby("emotion", group_keys=False).apply(
            lambda g: g.sample(min(len(g), limit), random_state=C.SEED))
    feats = []
    n = len(img)
    for i, row in enumerate(img.itertuples(index=False)):
        arr = np.asarray(Image.open(D.resolve(row.path)).convert("L"), dtype=np.uint8)
        f = image_match_features(arr)
        f["path"] = row.path
        feats.append(f)
        if (i + 1) % 2000 == 0:
            print(f"  image {i+1}/{n}")
    df = img.merge(pd.DataFrame(feats), on="path")
    out = out or os.path.join(C.DATA_DIR, "image_match.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-images", type=int, default=None,
                    help="per-emotion cap for HOG extraction (speed)")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()
    if not a.skip_audio:
        build_audio_match()
    if not a.skip_images:
        build_image_match(limit=a.limit_images)
