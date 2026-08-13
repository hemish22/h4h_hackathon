"""Part 1 — data audit. Writes reports/data_audit.md.

Verifies counts, post-mapping class distributions, the cross-modal mapping
conflict table, and CSV sanity (impossible values, monotonic score check).
Run: python -m src.audit
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import config as C
from . import data_utils as D

EXPECTED_IMAGE_COUNTS = {"Angry": 3995, "Disgust": 436, "Fear": 4097, "Happy": 7215,
                         "Neutral": 4965, "Sad": 4830, "Surprise": 3171}
EXPECTED_AUDIO_TOTAL = 1440
EXPECTED_IMAGE_TOTAL = 28709


def _md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(floatfmt)
    return df.to_markdown(index=True)


def run() -> str:
    lines: list[str] = ["# Data Audit\n"]
    warn: list[str] = []

    # ---- 1.1 counts
    audio = D.list_audio_files()
    img = D.list_image_files()
    csv = D.load_csv()

    lines.append("## 1.1 Counts\n")
    lines.append(f"- Audio wavs (Actor_01..24): **{len(audio)}** (expected {EXPECTED_AUDIO_TOTAL})")
    lines.append(f"- Images total: **{len(img)}** (expected {EXPECTED_IMAGE_TOTAL})")
    lines.append(f"- CSV shape: **{csv.shape}** (expected (4000, 22))\n")
    if len(audio) != EXPECTED_AUDIO_TOTAL:
        warn.append(f"audio count {len(audio)} != {EXPECTED_AUDIO_TOTAL}")
    if len(img) != EXPECTED_IMAGE_TOTAL:
        warn.append(f"image count {len(img)} != {EXPECTED_IMAGE_TOTAL}")

    img_counts = img.emotion.value_counts().to_dict()
    lines.append("Per-emotion image counts:\n")
    lines.append(_md_table(pd.DataFrame.from_dict(
        {k: [img_counts.get(k, 0), EXPECTED_IMAGE_COUNTS[k]] for k in C.IMAGE_EMOTIONS},
        orient="index", columns=["actual", "expected"])))
    lines.append("")

    # audio structure asserts
    per_actor = audio.groupby("actor").size()
    lines.append(f"\nAudio: {audio.actor.nunique()} actors, "
                 f"{per_actor.min()}–{per_actor.max()} clips/actor "
                 f"({'uniform 60' if (per_actor == 60).all() else 'NON-UNIFORM'}).\n")

    # ---- 1.2 post-mapping class distributions
    lines.append("## 1.2 Post-mapping stress-class distribution\n")
    dist = pd.DataFrame({
        "Audio": audio.stress.value_counts().reindex(C.STRESS_CLASSES).fillna(0).astype(int),
        "Images": img.stress.value_counts().reindex(C.STRESS_CLASSES).fillna(0).astype(int),
        "CSV": csv[C.TARGET_CAT].value_counts().reindex(C.STRESS_CLASSES).fillna(0).astype(int),
    })
    lines.append(_md_table(dist))
    severe_audio = int(dist.loc["Severe_Stress", "Audio"])
    lines.append(f"\n**Audio Severe = {severe_audio} clips** — binding constraint on the build.\n")

    # ---- 1.3 mapping conflict
    lines.append("## 1.3 Cross-modal mapping conflict\n")
    conf_rows = []
    all_emos = ["neutral", "calm", "happy", "sad", "surprised", "fearful", "angry", "disgust"]
    img_lookup = {k.lower(): v for k, v in C.IMAGE_EMO_TO_STRESS.items()}
    img_lookup["fearful"] = img_lookup.get("fear")   # name alignment
    img_lookup["surprised"] = img_lookup.get("surprise")
    for e in all_emos:
        a = C.AUDIO_EMO_TO_STRESS.get(e)
        i = img_lookup.get(e)
        status = "audio-only" if i is None else ("CONFLICT" if a != i else "agree")
        conf_rows.append([e, a, i or "—", status])
    lines.append(_md_table(pd.DataFrame(conf_rows,
                 columns=["emotion", "audio->", "image->", "status"]).set_index("emotion")))
    lines.append("\nConflict confined to angry & disgust (the two highest classes), exact inversion:")
    lines.append("- Severe: disgust voice + angry face")
    lines.append("- Moderate audio pool 50% conflict-sourced; image pool ~9.6%.\n")

    # ---- 1.4 CSV checks
    lines.append("## 1.4 CSV sanity\n")
    lines.append("Class balance (normalised):\n")
    lines.append(_md_table(csv[C.TARGET_CAT].value_counts(normalize=True)
                           .reindex(C.STRESS_CLASSES).to_frame("frac")))
    nulls = int(csv.isnull().sum().sum())
    lines.append(f"\nTotal nulls: {nulls}\n")

    # impossible values
    imp = {}
    for col, (lo, hi) in C.CLIP_RANGES.items():
        if col in csv.columns:
            n = int(((csv[col] < lo) | (csv[col] > hi)).sum())
            if n:
                imp[col] = n
    lines.append(f"Out-of-range values: {imp if imp else 'none'}\n")
    if imp:
        warn.append(f"out-of-range values present: {imp}")

    # monotonic score check -> ordinal justification
    lines.append("Mean scores per class (ordinal justification — should rise monotonically):\n")
    grp = csv.groupby(C.TARGET_CAT)[C.TARGET_REG].mean().reindex(C.STRESS_CLASSES)
    lines.append(_md_table(grp))
    mono = all(grp["Stress_Score"].diff().dropna() > 0)
    lines.append(f"\nStress_Score monotonic across Healthy->Severe: **{mono}** "
                 f"→ {'ordinal treatment justified' if mono else 'CHECK ordinal assumption'}\n")

    # feature-target correlations (top)
    corr = csv.corr(numeric_only=True)["Stress_Score"].drop(C.TARGET_REG).sort_values(key=abs, ascending=False)
    lines.append("Top feature↔Stress_Score correlations:\n")
    lines.append(_md_table(corr.head(8).to_frame("corr")))
    lines.append("")

    if warn:
        lines.insert(1, "> ⚠ WARNINGS: " + "; ".join(warn) + "\n")

    out = "\n".join(lines)
    path = os.path.join(C.REPORTS_DIR, "data_audit.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {path}")
    if warn:
        print("WARNINGS:", "; ".join(warn))
    return out


if __name__ == "__main__":
    run()
