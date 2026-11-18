# Feature Variant Analysis

## 1. Experiment definitions

### Experiment A — five original input features

Uses the five original input columns, order preserved exactly as configured:

| # | Column | Unit | Role |
|---|---|---|---|
| 1 | `frequency_hz` | Hz | Input feature |
| 2 | `angle_of_attack_deg` | degrees | Input feature |
| 3 | `chord_length_m` | m | Input feature |
| 4 | `free_stream_velocity_m_s` | m/s | Input feature |
| 5 | `suction_side_displacement_thickness_m` | m | Input feature |

Target (not part of the feature count): `scaled_sound_pressure_level_db` (dB).

### Experiment B — log-frequency representation

Replaces `frequency_hz` with `log10_frequency_hz` = `log10(frequency_hz)`. The remaining four features are unchanged. Order preserved:

| # | Column | Unit | Role |
|---|---|---|---|
| 1 | `log10_frequency_hz` | log10(Hz) | Input feature |
| 2 | `angle_of_attack_deg` | degrees | Input feature |
| 3 | `chord_length_m` | m | Input feature |
| 4 | `free_stream_velocity_m_s` | m/s | Input feature |
| 5 | `suction_side_displacement_thickness_m` | m | Input feature |

## 2. Descriptive statistics — frequency representations

Measured on the validated processed dataset (`data/processed/airfoil_self_noise.csv`, 1,503 rows):

| Statistic | `frequency_hz` (Experiment A) | `log10_frequency_hz` (Experiment B) |
|---|---|---|
| min | 200.00 | 2.3010 |
| max | 20,000.00 | 4.3010 |
| mean | 2,886.38 | 3.2249 |
| median | 1,600.00 | 3.2041 |
| skewness | +2.1371 | −0.0318 |
| unique values | 21 | 21 |

`log10` is a one-to-one monotonic transform, so all 21 distinct frequency values remain distinct (21 unique values in both columns).

## 3. Pearson correlations with the target

| Feature | Pearson r vs `scaled_sound_pressure_level_db` |
|---|---|
| `frequency_hz` (A) | −0.3907 |
| `log10_frequency_hz` (B) | −0.3547 |

All other four input features are unchanged between A and B, so their target correlations are identical in both variants:

| Feature | Pearson r vs target |
|---|---|
| `angle_of_attack_deg` | −0.1561 |
| `chord_length_m` | −0.2362 |
| `free_stream_velocity_m_s` | +0.1251 |
| `suction_side_displacement_thickness_m` | −0.3127 |

Input-input correlations involving frequency shift under the non-linear log10 transform (e.g., angle of attack vs frequency r = −0.2728 → vs log10 frequency r = −0.2921); correlations among the four unchanged features are identical in both variants.

## 4. Interpretation of skewness reduction

`frequency_hz` is strongly right-skewed (skewness +2.1371): most observations sit at low frequencies (median 1,600 Hz against a 200–20,000 Hz range), with a long high-frequency tail. Applying `log10` compresses that tail symmetrically, bringing skewness to approximately zero (−0.0318). The distribution becomes near-symmetric about its median (3.2249 mean vs 3.2041 median). This is a descriptive observation about distribution shape only; it does not by itself establish that the log-frequency representation will predict the target better.

## 5. Weaker Pearson correlation does not establish inferior predictive performance

Experiment B's `log10_frequency_hz` vs target correlation (−0.3547) is weaker in magnitude than Experiment A's raw-frequency correlation (−0.3907). Pearson correlation measures linear association only. A weaker linear correlation does not imply worse predictive performance, because:

- A subsequent model may capture non-linear relationships that Pearson r cannot detect.
- The log-frequency representation may improve conditioning, numerical stability, or linear-model behaviour even with a lower linear correlation.
- Predictive performance can only be assessed after training and evaluating models on held-out data, which is outside this phase.

## 6. Feature count and observation count confirmation

- Both variants retain exactly **five input features**.
- Both variants use all **1,503 observations**; no rows were removed or modified.
- The original processed dataset `data/processed/airfoil_self_noise.csv` is preserved unchanged.

## 7. No winning representation selected yet

**No winning representation has been selected.** Experiment A and Experiment B are two controlled, documented feature representations prepared for the subsequent modelling phase. Model training, selection, and comparison will occur in a later phase using held-out group-aware validation; no conclusion about which variant performs better is drawn here.
