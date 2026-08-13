"""Shared data helpers: audio file listing, RAVDESS filename parsing, image listing."""
from __future__ import annotations
import glob
import os
import pandas as pd

from . import config as C


def parse_filename(name: str) -> dict:
    """03-01-06-01-02-01-12.wav -> structured metadata.

    Fields: modality, vocal-channel, emotion, intensity, statement, repetition, actor.
    """
    base = os.path.basename(name)[:-4]
    mod, chan, emo, inten, stmt, rep, actor = [int(x) for x in base.split("-")]
    assert mod == 3 and chan == 1, f"spec: all files audio-only speech, got {base}"
    return dict(
        emotion_code=emo,
        emotion=C.AUDIO_EMO_CODE[emo],
        intensity=inten,                 # 1=normal, 2=strong (absent->1 for neutral)
        statement=stmt,
        repetition=rep,
        actor=actor,
        sex="M" if actor % 2 else "F",
    )


def list_audio_files() -> pd.DataFrame:
    """All 1440 unique wavs from Actor_01..Actor_24. Excludes the duplicate
    `audio_speech_actors_01-24` mirror dir that ships in the drive folder."""
    rows = []
    for actor_dir in sorted(glob.glob(os.path.join(C.AUDIO_DIR, "Actor_*"))):
        if not os.path.isdir(actor_dir):
            continue
        for wav in sorted(glob.glob(os.path.join(actor_dir, "*.wav"))):
            meta = parse_filename(wav)
            meta["path"] = wav
            meta["stress"] = C.AUDIO_EMO_TO_STRESS[meta["emotion"]]
            rows.append(meta)
    df = pd.DataFrame(rows)
    return df


def list_image_files() -> pd.DataFrame:
    """All FER images with emotion + mapped stress class."""
    rows = []
    for emo in C.IMAGE_EMOTIONS:
        d = os.path.join(C.IMAGE_DIR, emo)
        if not os.path.isdir(d):
            continue
        for img in sorted(glob.glob(os.path.join(d, "*"))):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append(dict(path=img, emotion=emo,
                                 stress=C.IMAGE_EMO_TO_STRESS[emo]))
    return pd.DataFrame(rows)


def load_csv() -> pd.DataFrame:
    return pd.read_csv(C.CSV_PATH)
