"""Draw two presentation diagrams into figures/:
  alignment_diagram.png   — the feature-matched photo+audio+tabular joining
  architecture_diagram.png — encoders -> gated fusion -> multi-task heads
Run: python scripts/make_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

INK, SLATE, YELLOW, MIST, WHITE = "#222831", "#393E46", "#FFD369", "#EEEEEE", "#FFFFFF"


def box(ax, x, y, w, h, text, fc=WHITE, ec=SLATE, tc=INK, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=2, color=color))


def alignment():
    fig, ax = plt.subplots(figsize=(12, 6)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    fig.patch.set_facecolor(MIST); ax.set_facecolor(MIST)
    ax.text(6, 5.7, "The problem: three datasets, ZERO shared participants",
            ha="center", fontsize=15, weight="bold", color=INK)
    # three sources
    box(ax, 0.3, 3.7, 3.0, 1.1, "FER2013 faces\n28,709 images", fc=WHITE)
    box(ax, 0.3, 2.35, 3.0, 1.1, "RAVDESS voices\n1,440 clips", fc=WHITE)
    box(ax, 0.3, 1.0, 3.0, 1.1, "Behaviour/physiology CSV\n4,000 rows · 18 features", fc=WHITE)
    # matching engine
    box(ax, 4.4, 2.0, 3.2, 2.0,
        "MATCH in measured-\nfeature space\n(within stress class)", fc=INK, ec=INK, tc=MIST, bold=True, fs=12)
    ax.text(6.0, 1.55, "MFCC · Pitch · Speech_Rate=6/dur · HOG-variance",
            ha="center", fontsize=9.5, color=SLATE, style="italic")
    # output
    box(ax, 8.7, 2.15, 3.0, 1.7,
        "Synthetic participant\nface + voice + 18 feats\n+ 4-class + 3 scores",
        fc=YELLOW, ec=INK, bold=True, fs=11)
    for yy in (4.25, 2.9, 1.55):
        arrow(ax, 3.3, yy, 4.4, 3.0)
    arrow(ax, 7.6, 3.0, 8.7, 3.0)
    ax.text(6, 0.5, "The CSV already contains audio/face-derived columns — so we match each row "
            "to the nearest REAL voice & face, not a random one.",
            ha="center", fontsize=10.5, color=INK)
    plt.tight_layout(); p = os.path.join(FIG, "alignment_diagram.png")
    plt.savefig(p, dpi=140, facecolor=MIST); plt.close(); print("wrote", p)


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    fig.patch.set_facecolor(MIST); ax.set_facecolor(MIST)
    ax.text(6, 5.7, "Architecture — gated fusion, multi-task, uncertainty-aware",
            ha="center", fontsize=15, weight="bold", color=INK)
    box(ax, 0.3, 4.1, 2.6, 0.9, "Face → FaceCNN → 128", fc=WHITE)
    box(ax, 0.3, 2.9, 2.6, 0.9, "Voice → AudioCNN → 128", fc=WHITE)
    box(ax, 0.3, 1.7, 2.6, 0.9, "18 feats → MLP → 64", fc=WHITE)
    box(ax, 3.6, 2.6, 2.6, 1.7, "GATED FUSION\nper-participant\nmodality weights",
        fc=INK, ec=INK, tc=MIST, bold=True, fs=12)
    box(ax, 6.9, 3.0, 2.2, 1.1, "Trunk\n320→256→128", fc=WHITE)
    box(ax, 9.5, 4.0, 2.2, 0.95, "4-class stress\n(ordinal loss)", fc=YELLOW, ec=INK, bold=True, fs=10)
    box(ax, 9.5, 2.75, 2.2, 0.95, "3 severity scores\n(Dep/Anx/Str)", fc=YELLOW, ec=INK, bold=True, fs=10)
    box(ax, 6.9, 1.35, 2.2, 1.0, "Aux heads\n(per modality)", fc=WHITE, ec=SLATE, fs=10)
    for yy in (4.55, 3.35, 2.15):
        arrow(ax, 2.9, yy, 3.6, 3.45)
    arrow(ax, 6.2, 3.45, 6.9, 3.55)
    arrow(ax, 9.1, 3.55, 9.5, 4.47)
    arrow(ax, 9.1, 3.55, 9.5, 3.22)
    arrow(ax, 5.0, 2.6, 7.0, 1.85)
    ax.text(6, 0.6, "+ modality dropout (graceful degradation) · MC-dropout (uncertainty bands) · "
            "aux heads → free unimodal + decision-level fusion + concordance",
            ha="center", fontsize=10, color=INK)
    plt.tight_layout(); p = os.path.join(FIG, "architecture_diagram.png")
    plt.savefig(p, dpi=140, facecolor=MIST); plt.close(); print("wrote", p)


if __name__ == "__main__":
    alignment(); architecture()
