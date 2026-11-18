# Group Validation Analysis (Phase 6)

## 1. Operating group definition

Operating groups are defined solely by the four experimental-condition columns:

| Column | Unit |
|---|---|
| `angle_of_attack_deg` | degrees |
| `chord_length_m` | m |
| `free_stream_velocity_m_s` | m/s |
| `suction_side_displacement_thickness_m` | m |

`frequency_hz` is explicitly **excluded** from the group key, so every frequency variant of the same operating condition shares one group id. This prevents group leakage between training and holdout data when comparing the two feature representations.

## 2. Group count (T-04)

The validated processed dataset (`data/processed/airfoil_self_noise.csv`) generates exactly **106 operating groups** from the four group columns.

- Rows in dataset: 1,503
- Unique group combinations: 106
- Group size range: 8–19 rows per group (mean ~14.2)
- Group ids are deterministic and cover all 1,503 rows (no missing group ids).

Group membership is identical for both Experiment A and Experiment B feature representations, since frequency is not part of the key.

### Group-ID generation and encoding method

Group ids are computed once from the four-condition key as follows:

1. Select the four group columns (`angle_of_attack_deg`, `chord_length_m`, `free_stream_velocity_m_s`, `suction_side_displacement_thickness_m`) from the validated processed dataset.
2. Convert each column to string and join the four values per row with the separator `||`, producing one condition-combination key per row.
3. Apply `pandas.factorize(keys, sort=True)` — this returns integer codes for each unique key. With `sort=True`, unique keys are sorted lexicographically and integer codes `0..n_groups-1` are assigned in **sorted key order**, so each group id is a deterministic function of its condition-combination key string, independent of row order.
4. Broadcast the per-row code back to all 1,503 rows as column `group_id`.

The input frame is never mutated; `attach_group_ids(df)` returns a copy with `group_id` appended. Group ids are deterministic: calling the generator twice on the same frame yields identical ids.

**Tooling:** scikit-learn version **1.9.1** (`GroupShuffleSplit`, `GroupKFold` from `sklearn.model_selection`), pandas, numpy. Random seed fixed at `random_state=42` for GroupShuffleSplit; GroupKFold has no randomness.

Group counts are exact: 106 unique groups; train 84 / holdout 22; train observations 1,188 / holdout observations 315.

### Environment note (preflight)

The approved technology specification targets Python 3.11 (`requires-python = ">=3.11"` in `pyproject.toml`). Python 3.11 is not installed on this machine; the project virtual environment runs Python 3.14.3, which satisfies `>=3.11` and imports `airfoil_noise` cleanly. No interpreter was installed or switched — the discrepancy is documented here rather than silently resolved, and all `pyproject.toml` dependencies (pandas, numpy, scikit-learn 1.9.1, ...) are already installed in the venv.

## 3. Train/holdout split (T-05)

Method: `sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`.

`test_size=0.20` is interpreted by scikit-learn at the **group level**: it rounds up 20% of the 106 unique groups (ceil(21.2) = 22 groups) for the holdout set. Because groups contain unequal numbers of rows (8–19), the resulting row-level holdout proportion is approximately 21%, not exactly 20%.

Actual verified split:

| Set | Groups | Observations | Share of rows |
|---|---|---|---|
| Training | 84 | 1,188 | ~79.04% |
| Holdout | 22 | 315 | ~20.96% |
| Total | 106 | 1,503 | 100.0% |

Isolation guarantees (verified by test and by construction):
- Training and holdout group id sets are **completely disjoint**.
- Their union covers all 106 groups — no group is dropped.
- No operating group appears in both training and holdout.

## 4. Five-fold GroupKFold on training groups

Method: `sklearn.model_selection.GroupKFold(n_splits=5)` applied to the 84 training groups.

Across the five folds:
- Each fold's training and validation group id sets are **completely disjoint**.
- Every one of the 84 training groups appears in **exactly one** validation fold (84 unique validation-group occurrences, one per group).
- No training group is validated more than once or omitted.

Per-fold distribution:

| Fold | Train groups | Validation groups | Train observations | Validation observations |
|---|---|---|---|---|
| 0 | 67 | 17 | 948 | 240 |
| 1 | 67 | 17 | 937 | 251 |
| 2 | 67 | 17 | 950 | 238 |
| 3 | 67 | 17 | 954 | 234 |
| 4 | 68 | 16 | 963 | 225 |
| Total | — | 84 (unique) | — | 1,188 |

Validation observation counts sum to 1,188, confirming every training-group observation is held out exactly once across the five folds. Fold train+validation observation counts each equal 1,188, confirming no leakage into or out of the training block.

## 5. No information leakage

- Group ids are computed once from the four-condition key and reused identically for both feature representations.
- The same train/holdout split and same five folds apply to Experiment A and Experiment B, so model comparisons in Phase 7 will be made on identical group partitions.
- No observation belongs to more than one role (train / validation-fold / final holdout).
- Frequency is not a grouping variable, so there is no path for same-condition frequency variants to leak across splits.

## 6. Determinism and reproducibility

- `random_state=42` fixed for GroupShuffleSplit.
- Group id assignment via `pd.factorize` over sorted unique condition combinations — deterministic given the same input ordering.
- No randomness in GroupKFold.
- Group assignments, split, random seed, and feature representations are frozen; no code change alters them.
