"""Part 1.5 — frozen splits. Write once, never change.

- AUDIO: speaker-independent. actors 1-18 train / 19-21 val / 22-24 test.
  This is the single most important correctness decision — a random split lets
  the audio CNN memorise 24 voices.
- IMAGES: stratified random on 7 emotion labels, 80/10/10.
- CSV: stratified on Mental_Health_Status, 70/15/15.

Persisted as splits/{modality}_{split}.txt  (audio/image = paths, csv = row indices).
Run: python -m src.splits
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.model_selection import train_test_split

from . import config as C
from . import data_utils as D

AUDIO_TRAIN_ACTORS = list(range(1, 19))
AUDIO_VAL_ACTORS = [19, 20, 21]
AUDIO_TEST_ACTORS = [22, 23, 24]


def _write(name: str, items) -> None:
    path = os.path.join(C.SPLITS_DIR, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in items))
    print(f"  {name}: {len(items)}")


def make_audio_splits():
    audio = D.list_audio_files()
    for split, actors in [("train", AUDIO_TRAIN_ACTORS),
                          ("val", AUDIO_VAL_ACTORS), ("test", AUDIO_TEST_ACTORS)]:
        sel = audio[audio.actor.isin(actors)]
        _write(f"audio_{split}", sel.path.tolist())
    return audio


def make_image_splits():
    img = D.list_image_files()
    idx = np.arange(len(img))
    tr, tmp = train_test_split(idx, test_size=0.20, stratify=img.emotion.values,
                               random_state=C.SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=img.emotion.values[tmp],
                              random_state=C.SEED)
    for split, sel in [("train", tr), ("val", va), ("test", te)]:
        _write(f"image_{split}", img.path.values[sel].tolist())
    return img


def make_csv_splits():
    csv = D.load_csv()
    idx = np.arange(len(csv))
    y = csv[C.TARGET_CAT].values
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=C.SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=C.SEED)
    for split, sel in [("train", tr), ("val", va), ("test", te)]:
        _write(f"csv_{split}", sel.tolist())
    return csv


def load_split(name: str):
    with open(os.path.join(C.SPLITS_DIR, f"{name}.txt")) as f:
        return [ln for ln in f.read().splitlines() if ln]


def run():
    print("audio splits (speaker-independent):")
    make_audio_splits()
    print("image splits (stratified 80/10/10):")
    make_image_splits()
    print("csv splits (stratified 70/15/15):")
    make_csv_splits()
    # leakage firewall assertion
    tr = set(load_split("audio_train")); te = set(load_split("audio_test"))
    assert tr.isdisjoint(te), "AUDIO LEAKAGE: train/test overlap"
    print("OK: audio train/test speaker-disjoint")


if __name__ == "__main__":
    run()
