"""Part 3 + 8 — dataset that reads the manifest ONLY (never raw data paths).

Provides log-mel audio tensors, 48x48 face tensors, scaled tabular vectors, and
the multi-task targets. Tabular scaler is fit on train only and persisted.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import joblib

from . import config as C

MEL_T = 188          # frames after 3.0s @ hop 256
N_MELS = 128
IMG_SIZE = 48


# ------------------------------------------------------------------ preprocessing
def load_logmel(path, sr=16000, dur=3.0, augment=False):
    import librosa
    from .data_utils import resolve
    y, _ = librosa.load(resolve(path), sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    n = int(sr * dur)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    if augment:
        y = _augment_wave(y, sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256,
                                         n_mels=N_MELS)
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    if logmel.shape[1] < MEL_T:
        logmel = np.pad(logmel, ((0, 0), (0, MEL_T - logmel.shape[1])))
    else:
        logmel = logmel[:, :MEL_T]
    if augment:
        logmel = _spec_augment(logmel)
    return logmel.astype(np.float32)


def _augment_wave(y, sr):
    """Waveform augmentation — pure NumPy only.

    NOTE: librosa.effects.pitch_shift (numba phase-vocoder) is deliberately NOT
    used here. It crashes the process natively (no Python traceback) on Windows
    when hammered in the training loop — an OpenMP/numba instability. Pitch/timbre
    invariance is instead approximated by SpecAugment frequency masking, which is
    pure NumPy and robust. Onset invariance via roll, robustness via noise/gain.
    """
    r = np.random.default_rng()
    y = np.roll(y, int(r.uniform(-0.3, 0.3) * sr))                 # time shift
    snr = r.uniform(15, 30)                                        # additive noise
    y = y + np.random.randn(len(y)) * (y.std() / (10 ** (snr / 20)) + 1e-9)
    y = y * (10 ** (r.uniform(-6, 6) / 20))                        # gain jitter
    return y.astype(np.float32)


def _spec_augment(mel, F=16, T=24, nf=2, nt=2):
    mel = mel.copy()
    r = np.random.default_rng()
    for _ in range(nf):
        f = r.integers(0, F); f0 = r.integers(0, max(1, mel.shape[0] - f))
        mel[f0:f0 + f, :] = 0
    for _ in range(nt):
        t = r.integers(0, T); t0 = r.integers(0, max(1, mel.shape[1] - t))
        mel[:, t0:t0 + t] = 0
    return mel


def load_face(path, augment=False):
    from PIL import Image
    from .data_utils import resolve
    arr = np.asarray(Image.open(resolve(path)).convert("L").resize((IMG_SIZE, IMG_SIZE)),
                     dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    if augment:
        r = np.random.default_rng()
        if r.random() < 0.5:
            arr = arr[:, ::-1].copy()                              # horizontal flip
    return arr.astype(np.float32)


# ------------------------------------------------------------------ scaler
def fit_scaler(manifest):
    from sklearn.preprocessing import StandardScaler
    tr = manifest[manifest.split == "train"]
    sc = StandardScaler().fit(tr[C.FEATURE_COLS].values)
    joblib.dump(sc, os.path.join(C.ARTIFACTS_DIR, "tab_scaler.joblib"))
    return sc


def load_scaler():
    return joblib.load(os.path.join(C.ARTIFACTS_DIR, "tab_scaler.joblib"))


# ------------------------------------------------------------------ dataset
class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, split, scaler=None, augment=False,
                 cache_audio=True):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df.split == split].reset_index(drop=True)
        self.split = split
        self.augment = augment and split == "train"
        self.scaler = scaler if scaler is not None else load_scaler()
        self.tab = self.scaler.transform(self.df[C.FEATURE_COLS].values).astype(np.float32)
        self._mel_cache = {} if cache_audio else None

    def __len__(self):
        return len(self.df)

    def _get_mel(self, path):
        # Cache the BASE (un-augmented) log-mel once per unique clip, so librosa
        # decodes each file only once across all epochs. Augmentation is applied
        # as cheap SpecAugment on the cached copy each call — big speedup.
        # (Needs persistent_workers=True when num_workers>0, else per-worker
        #  caches are discarded between epochs.)
        if self._mel_cache is not None and path in self._mel_cache:
            mel = self._mel_cache[path]
        else:
            mel = load_logmel(path, augment=False)
            if self._mel_cache is not None:
                self._mel_cache[path] = mel
        if self.augment:
            mel = _spec_augment(mel.copy())
        return mel

    def __getitem__(self, i):
        row = self.df.iloc[i]
        mel = self._get_mel(row.audio_path)
        face = load_face(row.image_path, augment=self.augment)
        reg = np.array([row.depression, row.anxiety, row.stress_score],
                       dtype=np.float32) / np.array(C.REG_MAXIMA, dtype=np.float32)
        return {
            "face": torch.from_numpy(face),
            "voice": torch.from_numpy(mel),
            "tab": torch.from_numpy(self.tab[i]),
            "y_cls": torch.tensor(int(row.stress_ordinal)),
            "y_reg": torch.from_numpy(reg),
            "conflict": torch.tensor(bool(row.mapping_conflict)),
        }
