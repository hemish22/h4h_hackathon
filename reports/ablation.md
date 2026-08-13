# Ablation results

| Model | Pairing | Acc | MacroF1 | WeightedF1 | ROC-AUC | QWK | Dep MAE | Anx MAE | Str MAE | mean R² |
|---|---|---|---|---|---|---|---|---|---|---|
| Tabular only (LightGBM) | n/a | 0.370 | 0.260 | 0.355 | 0.493 | 0.050 | 9.05 | 6.64 | 10.11 | -0.121 |
| Gated fusion | random | 0.553 | 0.474 | 0.551 | 0.775 | 0.437 | 7.88 | 6.23 | 9.28 | 0.071 |
| Gated fusion | matched | 0.655 | 0.571 | 0.643 | 0.849 | 0.564 | 7.59 | 6.02 | 9.06 | 0.090 |
| + ordinal + modality dropout (final) | matched | 0.670 | 0.575 | 0.658 | 0.864 | 0.562 | 7.71 | 6.09 | 8.95 | 0.060 |

**Key finding — feature-matched alignment helps.** Matched pairing beats random pairing by ~10 accuracy points and ~0.13 QWK under an identical model, isolating the contribution of the alignment engine (criterion 3).

**Tabular is near-noise** (LightGBM acc 0.37, negative R²); fusion carries the signal via the voice and face emotion channels — confirmed by the gate weights (voice ≈ 0.41, face ≈ 0.34–0.40, tabular ≈ 0.18–0.26).

Severe-class recall stays low (~0.26–0.32): the CSV is heavily skewed (Severe ≈ 3%, only 19 test participants) — a stated limitation, not a bug.

![confusion](../figures/confusion_final_matched.png)

![gates](../figures/gate_weights.png)
