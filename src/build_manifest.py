"""Part 2.6-2.7 — build participant manifests (random + matched).

Reads: data/audio_match.csv, data/image_match.csv, CSV, frozen splits.
Writes: data/manifest_random.csv, data/manifest_matched.csv.

Both carry the mapping_conflict flag. Matching is greedy nearest-neighbour in
z-scored measured-feature space, within stress class, with per-split capacity
caps. Match is within-split only (leakage firewall) — asserted.
Run: python -m src.build_manifest
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from . import config as C
from . import data_utils as D
from . import splits as S


def _attach_split(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind in {audio,image,csv}. Adds a 'split' column from frozen split files."""
    df = df.copy()
    df["split"] = None
    for sp in ["train", "val", "test"]:
        members = set(S.load_split(f"{kind}_{sp}"))
        if kind == "csv":
            members = set(int(x) for x in members)
            mask = df.index.isin(members)
        else:
            mask = df["path"].isin(members)
        df.loc[mask, "split"] = sp
    return df[df.split.notna()].copy()


def match(csv_df, pool_df, cols, split, rng, matched=True):
    """Return dict csv_index -> pool_index. If matched=False, random within class."""
    pairs = {}
    for cls in C.STRESS_CLASSES:
        C_ = csv_df[(csv_df[C.TARGET_CAT] == cls) & (csv_df.split == split)]
        P_ = pool_df[(pool_df.stress == cls) & (pool_df.split == split)]
        assert len(P_) > 0, f"empty pool: {cls}/{split}"
        cap = int(np.ceil(len(C_) / len(P_)))
        used = np.zeros(len(P_), dtype=int)

        if matched:
            mu = C_[cols].mean()
            sd = C_[cols].std().replace(0, 1)
            Cz = ((C_[cols] - mu) / sd).values
            Pz = ((P_[cols] - mu) / sd).values
            D_ = cdist(Cz, Pz, "euclidean")
            order_rows = np.argsort(D_.min(axis=1))     # hardest-to-match first
            for i in order_rows:
                order = np.argsort(D_[i])
                j = next(k for k in order if used[k] < cap)
                used[j] += 1
                pairs[C_.index[i]] = (P_.index[j], float(D_[i, j]))
        else:
            pool_idx = list(P_.index)
            for i in C_.index:
                # respect cap even in random baseline
                choices = [k for k in range(len(P_)) if used[k] < cap]
                k = rng.choice(choices)
                used[k] += 1
                pairs[i] = (P_.index[k], np.nan)
        assert used.max() <= cap, f"cap violated {cls}/{split}"
    return pairs


def build(matched: bool) -> pd.DataFrame:
    csv = _attach_split(D.load_csv(), "csv")
    audio = _attach_split(pd.read_csv(os.path.join(C.DATA_DIR, "audio_match.csv")), "audio")
    image = _attach_split(pd.read_csv(os.path.join(C.DATA_DIR, "image_match.csv")), "image")
    rng = np.random.default_rng(C.SEED)

    rows = []
    for split in ["train", "val", "test"]:
        a_pairs = match(csv, audio, C.AUDIO_MATCH_COLS, split, rng, matched)
        i_pairs = match(csv, image, C.IMAGE_MATCH_COLS, split, rng, matched)
        for ci in csv[csv.split == split].index:
            ai, ad = a_pairs[ci]
            ii, idi = i_pairs[ci]
            crow = csv.loc[ci]
            arow = audio.loc[ai]
            irow = image.loc[ii]
            rec = {
                "participant_id": ci,
                "csv_index": ci,
                "split": split,
                "audio_path": arow.path,
                "image_path": irow.path,
                "stress_class": crow[C.TARGET_CAT],
                "stress_ordinal": C.CLASS_TO_ORD[crow[C.TARGET_CAT]],
                "depression": crow.Depression_Score,
                "anxiety": crow.Anxiety_Score,
                "stress_score": crow.Stress_Score,
                "audio_emotion": arow.emotion,
                "image_emotion": irow.emotion,
                "audio_intensity": int(arow.intensity),
                "actor_id": int(arow.actor),
                "actor_sex": arow.sex,
                "match_dist_audio": ad,
                "match_dist_image": idi,
            }
            for col in C.FEATURE_COLS:
                rec[col] = crow[col]
            rows.append(rec)

    m = pd.DataFrame(rows)
    m["mapping_conflict"] = (
        m.audio_emotion.str.lower().isin(C.CONFLICT_EMOTIONS)
        | m.image_emotion.str.lower().isin(C.CONFLICT_EMOTIONS)
    )
    # leakage firewall: actors in train never appear in test
    tr_actors = set(m[m.split == "train"].actor_id)
    te_actors = set(m[m.split == "test"].actor_id)
    assert tr_actors.isdisjoint(te_actors), "AUDIO LEAKAGE across manifest splits"

    name = "matched" if matched else "random"
    out = os.path.join(C.DATA_DIR, f"manifest_{name}.csv")
    m.to_csv(out, index=False)
    print(f"wrote {out}  ({len(m)} rows, conflict={int(m.mapping_conflict.sum())})")
    return m


def run():
    build(matched=False)
    build(matched=True)
    print("manifests frozen. downstream reads these ONLY.")


if __name__ == "__main__":
    run()
