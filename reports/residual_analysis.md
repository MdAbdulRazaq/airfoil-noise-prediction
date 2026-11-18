# Phase 9 — Residual Analysis and Explainability

**Source:** Section 9.7 / 10.10 of the project specification.
**Configuration:** Experiment B (log10_frequency_hz + four unchanged inputs) + SVR(kernel="rbf", C=100, ε=0.2, γ="scale"), the Phase 7 selection (lowest mean GroupKFold CV MAE = 1.8947). Pipeline refit once on the frozen 1,188-row training block and applied once to the frozen 315-row holdout block — no retuning, no alternative models, no holdout-based changes.

## Holdout error summary

| Metric | Value |
|---|---|
| Holdout MAE | 2.067 dB |
| Holdout RMSE | 3.144 dB |
| Holdout R² | 0.770 |
| Holdout observations | 315 |
| Holdout operating groups | 22 |
| Training observations | 1,188 (84 groups) |

Generalization gap vs cross-validation: CV R² 0.856 → holdout R² 0.770; CV MAE 1.895 → holdout MAE 2.067. Honest gap consistent with an out-of-sample evaluation; nothing in this phase changes the frozen holdout predictions or metrics (`reports/holdout_predictions.csv`, `reports/holdout_evaluation.json`).

## 1. Permutation feature importance (holdout)

Computed with `sklearn.inspection.permutation_importance`, scoring=`neg_mean_absolute_error`, `n_repeats=30`, `random_state=42`, on the held-out predictions of the refit pipeline. Positive values mean permuting the feature increased holdout MAE. Reported as model behaviour only — **not** as physical causality.

| Rank | Feature | Importance (Δ MAE, dB) | Std (dB) |
|---|---|---|---|
| 1 | log10_frequency_hz | 4.890 | 0.262 |
| 2 | chord_length_m | 3.092 | 0.193 |
| 3 | angle_of_attack_deg | 2.600 | 0.157 |
| 4 | suction_side_displacement_thickness_m | 2.523 | 0.151 |
| 5 | free_stream_velocity_m_s | 0.655 | 0.081 |

**Interpretation:** log10 frequency is by far the most influential feature for this SVR (its removal raises holdout MAE by ~4.9 dB), consistent with Experiment B's log-transformation helping SVR in Phase 7 (CV MAE 1.89 vs 2.31 for raw frequency). Chord length, angle of attack, and thickness form a mid tier (~2.5–3.1 dB each, overlapping standard errors), while free-stream velocity has small, low-variance importance (~0.66 dB) — the model leans on it least of the five. Widths of the standard-error bars are narrow relative to means for all five features, so the ranking is stable across the 30 repeats.

**Limitation:** permutation importance measures what this fitted SVR uses on this holdout split; it does not rank physical drivers of noise, and correlated inputs can share importance. No causal claim is made about any feature.

Outputs: `reports/permutation_feature_importance.csv`, `reports/figures/permutation_feature_importance.png`.

## 2. Residuals by frequency

Scatter of residual = actual − predicted (dB) versus the original `frequency_hz` (200–20,000 Hz) on the 315 holdout rows (Figure: `reports/figures/residuals_by_frequency.png`).

Residuals sit scattered around the zero line across the full frequency range with no obvious gap at any band; absolute errors stay roughly in the same scale at low (200–1,000 Hz), mid, and high (12,500–20,000 Hz) frequencies. There is no systematic drift that would justify a frequency-dependent correction, but a scatter plot cannot rule out a weak trend — nothing in this phase models one, and no causal frequency effect is claimed.

## 3. Residuals by operating configuration

Each of the 22 holdout operating groups (exact `angle_of_attack || chord || velocity || thickness` combinations, frequency excluded from the key) is summarized in `reports/residuals_by_operating_group.csv`:

- Worst group: id `55` (α=22.2°, chord 0.0254 m, V=71.3 m/s, thickness 0.0214 m, n=15): all-15 residuals positive, mean residual +8.48 dB, max |error| 21.82 dB. The model systematically under-predicts this configuration on the holdout — the single largest error cluster, but one group among 22, not evidence of a global bias.
- Next largest |mean residual|: group `93` (−2.11 dB), `42` (+2.23 dB mean but max |error| 12.29 dB driven by one point), `47` (−1.60 dB).
- Remaining groups: mean residuals within ≈ ±1.7 dB, most within ±1 dB; mean absolute errors 0.39–3.56 dB; RMSE up to 3.73 dB (group 97).

Across groups residuals are split roughly evenly between under- and over-prediction (about split 10/12 positive vs negative means), so no overall directional bias is visible on this holdout. Small groups (n=11–17) make per-group means noisy — treat individual group values as descriptive, not as evidence of a systematic configuration effect.

## Limitations

- Single frozen holdout (315 rows / 22 groups); results are specific to this split and would shift on a different draw.
- Permutation importance and residual patterns describe model behaviour on held-out data — not physical causes of airfoil noise.
- Group sizes are small (11–17 observations), so per-group residual aggregates have wide sampling variability.
- No recommendation to adjust features, hyperparameters, or the model follows from this analysis; Phase 9 is read-only and the frozen configuration and holdout metrics remain unchanged.

Outputs: `reports/residuals_by_operating_group.csv`, plus the two figures above.
