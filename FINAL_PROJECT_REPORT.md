# Airfoil Noise Predictor — Final Project Report

**Release 1.0** · B.Tech Data Science individual project · Baseline scope, 14 days
**Repository:** `airfoil-noise-prediction/` · **Frontend:** Streamlit · **Backend:** Python + scikit-learn
**Dataset:** UCI Airfoil Self-Noise, dataset 291, DOI `10.24432/C5VW2C`, CC BY 4.0

All performance figures below are measured by the executed implementation; none are promised in advance or fabricated.

---

## Abstract

Airfoil self-noise — broadband noise from turbulent boundary-layer interaction with an airfoil trailing edge — depends on several interacting wind-tunnel measurements. This project builds a reproducible, local-only regression prototype that estimates the **scaled sound pressure level (dB)** from five experimental inputs: frequency, angle of attack, chord length, free-stream velocity, and suction-side displacement thickness. Using the UCI Airfoil Self-Noise dataset (1,503 observations, no missing values), the workflow validates the data, defines 106 operating-configuration groups, splits 84/22 groups into train/holdout with no configuration appearing in both, compares five regression models under GroupKFold cross-validation, selects Experiment B + SVR(RBF, C=100, ε=0.2, γ="scale") by lowest mean CV MAE, evaluates the frozen selection once on the untouched holdout (MAE 2.0673 dB, RMSE 3.1444 dB, R² 0.7702), and explains it with permutation importance and residual analysis. The fitted pipeline is saved as `models/best_model.joblib` with metadata, and exposed through a single-page Streamlit app with five bounded inputs and one 2-decimal dB prediction. Release 1.0 contains no accounts, database, REST API, cloud component, or prediction history.

## 1. Introduction and problem statement

### 1.1 Problem

Airfoil noise measurements depend on frequency, angle of attack, chord length, free-stream velocity, and suction-side displacement thickness. Can a supervised regression model learn the relationship between these measured variables and the dataset target, scaled sound pressure level (dB)?

### 1.2 Objective

Build a reproducible end-to-end regression workflow and a small local prediction interface that estimates the dataset target for numeric inputs within the observed feature ranges. The model is an academic prototype trained on the supplied experimental dataset; it is **not** an aircraft certification, safety, design-release, or operational noise-compliance system.

### 1.3 Scope position

This is Release 1.0 of a locked baseline. Release 2.0 functionality (accounts, history, API, database, cloud, exports) is explicitly out of scope and was not implemented.

## 2. Objectives

1. Use the supplied original dataset as the source of truth; preserve field meanings and units.
2. Create a deterministic, reproducible training and evaluation workflow (fixed seed 42; package versions recorded).
3. Compare five regression approaches under the same group-aware validation design.
4. Select the model using training-only cross-validation; evaluate the frozen selection exactly once on an untouched holdout set.
5. Explain the selected model with permutation feature importance and residual/error inspection.
6. Provide a usable local Streamlit interface: five validated inputs, one prediction output in dB.
7. Deliver a documented, testable prototype: 78 automated tests all passing; README with setup, training, and app commands.

## 3. Literature and background

- **[R1]** UCI Machine Learning Repository. “Airfoil Self-Noise”, dataset 291 — NASA data from aerodynamic and acoustic tests of two- and three-dimensional airfoil blade sections in an anechoic wind tunnel; NACA 0012 airfoils of different sizes at several wind-tunnel speeds and angles of attack. DOI: `https://doi.org/10.24432/C5VW2C`. License: CC BY 4.0. Accessed 16 November 2026.
- **[R2]** Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise* [Dataset]. UCI Machine Learning Repository. DOI `10.24432/C5VW2C`.
- **[R3]** Project data audit against the supplied `airfoil_self_noise.dat`: 1,503 rows, six numeric columns, zero missing cells, zero exact duplicate rows, 106 operating configurations under the project group definition.

Source-metadata note documented rather than silently resolved: UCI labels the attack-angle feature as “Binary”, but the raw file contains 27 distinct numeric angle values (0–22.2°). The implementation treats `angle_of_attack_deg` as numeric.

## 4. Dataset and methodology

### 4.1 Data

| Property | Value |
|---|---|
| Canonical raw file | `data/raw/airfoil_self_noise.dat` (whitespace-separated, no header — never modified) |
| Observations | 1,503 |
| Columns | 6 (five inputs + target) |
| Missing values | 0 |
| Exact duplicate rows | 0 |
| Distinct operating configurations (group key) | 106 |
| Distinct frequencies | 21 |
| Target range | 103.380 – 140.987 dB (mean 124.836 dB, std 6.899 dB) |

### 4.2 Input variables and empirical bounds

| Input | Unit | Observed range (enforced in UI and backend) |
|---|---|---|
| `frequency_hz` | Hz | 200.0 – 20,000.0 |
| `angle_of_attack_deg` | degrees | 0.0 – 22.2 |
| `chord_length_m` | m | 0.0254 – 0.3048 |
| `free_stream_velocity_m_s` | m/s | 31.7 – 71.3 |
| `suction_side_displacement_thickness_m` | m | 0.000400682 – 0.0584113 |

Bounds are empirical safeguards from the dataset audit, stored once in `src/airfoil_noise/config.py` and shared by validation, tests, and the UI — never duplicated.

### 4.3 Methodology

1. **Ingest and validate** — raw `.dat` loaded with whitespace separation; schema, row count (1,503), dtypes, missing values, finiteness, duplicates, and input bounds checked (`src/airfoil_noise/data.py`, `validation.py`).
2. **Operating groups** — group id from `angle_of_attack_deg`, `chord_length_m`, `free_stream_velocity_m_s`, `suction_side_displacement_thickness_m` only; `frequency_hz` excluded so every frequency variant of one condition stays in one group (106 groups, deterministic via `pd.factorize` over sorted keys).
3. **Group-aware split** — `GroupShuffleSplit(test_size=0.20, random_state=42)` over groups: 84 training groups / 1,188 observations; 22 holdout groups / 315 observations; disjoint group sets; union covers all 106 groups.
4. **Feature experiments** — Experiment A (five original inputs) vs Experiment B (`log10_frequency_hz` plus the four unchanged inputs); exactly five features in both.
5. **Model comparison** — Linear Regression, Ridge, Random Forest, SVR(RBF), Gradient Boosting; each tuned with GridSearchCV inside GroupKFold(5) on training groups only; preprocessing inside pipelines (scalers fitted per fold, never on full data); selection = lowest mean GroupKFold CV MAE.
6. **Frozen artifact** — winning pipeline refit on all 1,188 training observations; staged to `models/*.tmp`, verified in a fresh Python process (finite prediction matching the in-memory pipeline on a real dataset row), then promoted atomically to `models/best_model.joblib` + `models/model_metadata.json`.
7. **Holdout evaluation** — exactly once, after selection is frozen; never used for tuning.
8. **Explainability** — permutation importance (`n_repeats=30`, `random_state=42`) and residual analysis on the holdout set; model behaviour only, no physical-causality claims.

## 5. Exploratory analysis

(Measured on the validated processed dataset; full figures in `reports/figures/` and findings in `reports/feature_variant_analysis.md`.)

- `frequency_hz` is strongly right-skewed (skewness +2.1371; median 1,600 Hz vs 200–20,000 Hz range). `log10` compression brings skewness to approximately zero (−0.0318) while preserving all 21 distinct values.
- Pearson correlation with target: raw frequency −0.3907 vs `log10_frequency_hz` −0.3547. Weaker linear correlation after the transform does **not** imply worse predictive performance — Pearson measures linear association only; performance is assessed on held-out data.
- Target correlations (identical in both experiments): angle of attack −0.1561; chord length −0.2362; free-stream velocity +0.1251; displacement thickness −0.3127.
- Input-input correlations involving frequency shift under the non-linear log10 transform; correlations among the four unchanged features are identical across experiments.

## 6. Feature engineering

Two controlled, exactly-five-feature representations:

| Experiment | Features |
|---|---|
| A | `frequency_hz`, `angle_of_attack_deg`, `chord_length_m`, `free_stream_velocity_m_s`, `suction_side_displacement_thickness_m` |
| B | `log10_frequency_hz`, `angle_of_attack_deg`, `chord_length_m`, `free_stream_velocity_m_s`, `suction_side_displacement_thickness_m` |

Experiment B won (Section 8). The `log10(frequency_hz)` transform is computed inside the fitted pipeline, so the app and backend always consume the raw five features in documented order; frequency is positive across the dataset so the log is defined everywhere.

## 7. Group-aware validation methodology

Group ids are computed once from the four-condition key and reused identically for both experiments. `GroupShuffleSplit` reserves ~20% of groups for holdout; `GroupKFold(5)` validates within the 84 training groups so each training group is held out exactly once.

| Set | Groups | Observations | Share of rows |
|---|---|---|---|
| Training | 84 | 1,188 | ~79.04% |
| Holdout | 22 | 315 | ~20.96% |
| Total | 106 | 1,503 | 100.0% |

Leakage control: train and holdout group sets are disjoint and cover all groups; every training group appears in exactly one validation fold; frequency is not a grouping variable, so no same-condition frequency variant can leak across splits.

## 8. Model experiments

Five candidates compared on identical group splits (full table: `reports/model_comparison.csv`; fold detail: `reports/model_comparison_folds.csv`):

| Variant | Model | Mean CV MAE | Std CV MAE | Mean CV RMSE | Mean CV R² |
|---|---|---:|---:|---:|---:|
| Experiment B | SVR (RBF) | 1.8947 | 0.3294 | 2.6272 | 0.8556 |
| Experiment A | Gradient Boosting | 1.8972 | 0.4811 | 2.6352 | 0.8506 |
| Experiment B | Gradient Boosting | 1.8972 | 0.4811 | 2.6352 | 0.8506 |
| Experiment A | Random Forest | 1.9740 | 0.4604 | 2.6602 | 0.8477 |
| Experiment B | Random Forest | 1.9742 | 0.4579 | 2.6614 | 0.8476 |
| Experiment A | SVR (RBF) | 2.3147 | 0.3901 | 3.1743 | 0.7888 |
| Experiment A | Linear Regression | 3.7102 | 0.3977 | 4.7582 | 0.5278 |
| Experiment B | Ridge | 3.7862 | 0.2615 | 4.8557 | 0.5078 |
| Experiment B | Linear Regression | 3.7861 | 0.2615 | 4.8557 | 0.5078 |
| Experiment A | Ridge | 3.7102 | 0.3977 | 4.7582 | 0.5278 |

**Selection:** Experiment B + SVR(kernel="rbf"), C=100.0, ε=0.2, γ="scale" — lowest mean CV MAE 1.8947. Selection rule (lowest mean GroupKFold CV MAE) was declared before holdout evaluation; RMSE/R² are supporting metrics only.

Hyperparameter grids were modest (training folds only): SVR tuned over C ∈ {1, 10, 100}, ε ∈ {0.1, 0.2}, γ ∈ {scale, auto}; Random Forest and Gradient Boosting over tree size/depth; Ridge over α; Linear Regression untuned.

## 9. Evaluation

Frozen metrics (`models/model_metadata.json`, `reports/holdout_evaluation.json`):

| Measure | Value |
|---|---|
| Mean GroupKFold CV MAE | 1.8947 dB (std 0.3294) |
| Mean GroupKFold CV RMSE | 2.6272 dB (std 0.5015) |
| Mean GroupKFold CV R² | 0.8556 (std 0.0422) |
| Holdout MAE (evaluated once) | 2.0673 dB |
| Holdout RMSE | 3.1444 dB |
| Holdout R² | 0.7702 |
| Holdout observations / groups | 315 / 22 |

Generalization gap (CV R² 0.856 → holdout R² 0.770; CV MAE 1.895 → holdout MAE 2.067) is an honest out-of-sample gap; holdout metrics are frozen and were never retuned. Holdout predictions (`reports/holdout_predictions.csv`) record actual, predicted, residual, and group id for all 315 rows and are byte-compared against regeneration in tests.

Sanity check through the approved artifact: the first dataset row (800 Hz, 0°, 0.3048 m, 71.3 m/s, 0.00266337 m) predicts **126.40 dB**, identical before and after joblib serialization (difference < 1e-9).

## 10. Explainability and findings

Permutation importance on holdout (scoring `neg_mean_absolute_error`, `n_repeats=30`, `random_state=42`) — model behaviour, not physical causality:

| Rank | Feature | Δ MAE (dB) | Std (dB) |
|---|---|---:|---:|
| 1 | `log10_frequency_hz` | 4.890 | 0.262 |
| 2 | `chord_length_m` | 3.092 | 0.193 |
| 3 | `angle_of_attack_deg` | 2.600 | 0.157 |
| 4 | `suction_side_displacement_thickness_m` | 2.523 | 0.151 |
| 5 | `free_stream_velocity_m_s` | 0.655 | 0.081 |

log10 frequency is by far the most influential feature (~4.9 dB MAE increase when permuted), consistent with Experiment B outperforming raw-frequency SVR (1.89 vs 2.31 CV MAE). Free-stream velocity is used least (~0.66 dB). Narrow standard errors make the ranking stable across repeats.

**Residuals** (`reports/residual_analysis.md`, `reports/figures/residuals_by_frequency.png`, `reports/residuals_by_operating_group.csv`):

- Residuals (actual − predicted) scatter around zero across 200–20,000 Hz with no systematic frequency-dependent drift; no frequency correction is warranted.
- Worst holdout group: id 55 (α=22.2°, chord 0.0254 m, V=71.3 m/s, thickness 0.0214 m, n=15) — all 15 residuals positive, mean +8.48 dB, max |error| 21.82 dB: under-prediction of one configuration, not global bias.
- Group means split roughly evenly between under- and over-prediction (about 10/12); remaining groups within ≈ ±2.2 dB mean residual. Small groups (n=11–17) make per-group means noisy — descriptive only.
- No adjustment to features, hyperparameters, or the model follows from this analysis; the frozen configuration and holdout metrics remain unchanged.

## 11. Technical architecture

Single local Python repository; offline training and local inference share one codebase; the trained preprocessing-plus-estimator pipeline is the single source of prediction behaviour; the UI calls a Python prediction function directly (no HTTP layer); raw data is immutable, processed data and artifacts are reproducible outputs.

```
user input (Streamlit) → validate_prediction_input (FEATURE_BOUNDS)
  → load_model (best_model.joblib + metadata compatibility check)
  → one-row DataFrame in FEATURE_COLUMNS order
  → pipeline: log10(frequency_hz) → StandardScaler → SVR(RBF)
  → finite float → "{:.2f} dB"
```

Metadata records dataset DOI, feature order, variant, model, hyperparameters, seed, CV/holdout metrics, group counts, and package versions (numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, joblib 1.6.0; Python 3.14.3).

## 12. Technology stack

| Layer | Technology |
|---|---|
| Language | Python >= 3.11 (validated on 3.14.3) |
| Data | pandas 3.1.1, numpy 2.5.5 |
| ML | scikit-learn 1.9.1 (pipelines, GroupKFold/GroupShuffleSplit, GridSearchCV, SVR, permutation importance) |
| Persistence | joblib 1.6.0 (trusted local artifact only) |
| Visualization | matplotlib 3.11.2, seaborn 0.13.2 (research/report figures only) |
| Frontend | Streamlit 1.67.0 (single page) |
| Testing | pytest 9.2.0 |
| Notebooks | Jupyter (dev) |

Explicitly not used: database, authentication, REST API, Docker, cloud hosting, XGBoost (sklearn `GradientBoostingRegressor` instead), SHAP (optional; permutation importance is the required explainability method). Exact resolved versions: `requirements-lock.txt`.

## 13. Installation

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd:        .venv\Scripts\activate.bat
# Unix/macOS:         source .venv/bin/activate
pip install -e .              # runtime dependencies from pyproject.toml
pip install -e ".[dev]"       # adds pytest, jupyter
```

Run tests: `pytest -q` (78 tests). Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MPLBACKEND=Agg` for stable threading/matplotlib behaviour.

Regenerate the Release 1.0 artifact: `python src/airfoil_noise/artifact.py` (deterministic given frozen data, group split, seed 42, recorded hyperparameters).

Start the app: `streamlit run app/streamlit_app.py` then open `http://localhost:8501`.

## 14. Project folder structure (repository root)

```
airfoil-noise-prediction/
├── APPROACH ... (none — see sections above)
├── FINAL_PROJECT_REPORT.md          # this report
├── Airfoil_Noise_Predictor_Presentation_Release_1.0.pptx
├── README.md                        # setup, run commands, dataset citation, results, limitations
├── requirements-lock.txt            # exact resolved package versions (pip freeze)
├── pyproject.toml                   # packaging: name airfoil-noise-prediction 1.0.0, Python >= 3.11
├── .gitignore                       # .venv, caches, data/processed, *.pyc, .env, logs
├── app/
│   └── streamlit_app.py             # Phase 12 frontend (five inputs, one Predict, one dB result)
├── data/
│   ├── raw/
│   │   └── airfoil_self_noise.dat   # canonical raw source — immutable, hash pinned in tests
│   └── processed/
│       └── airfoil_self_noise.csv   # reproducible processed copy
├── models/
│   ├── best_model.joblib            # frozen Release 1.0 pipeline artifact
│   └── model_metadata.json          # DOI, metrics, hyperparameters, seed, package versions
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_exploratory_analysis.ipynb
├── reports/
│   ├── model_comparison.csv         # five-model CV comparison (Phase 7)
│   ├── model_comparison_folds.csv   # fold-level detail
│   ├── holdout_evaluation.json      # frozen holdout metrics (Phase 8)
│   ├── holdout_predictions.csv      # 315 rows: actual, predicted, residual, group id
│   ├── permutation_feature_importance.csv
│   ├── residuals_by_operating_group.csv
│   ├── feature_variant_analysis.md  # EDA of frequency representations
│   ├── group_validation_analysis.md # group-aware validation methodology
│   ├── residual_analysis.md         # explainability + error analysis (Phase 9)
│   └── figures/                     # 11 PNG figures (histograms, boxplots, heatmap, correlations, residuals, importance, holdout)
├── src/airfoil_noise/               # backend: config, data, validation, features, groups,
│                                    # holdout, models, explainability, predict, artifact
├── tests/                           # 78 tests across 9 files (all passing)
└── apptest_check.py                 # Section 11.2 frontend acceptance via Streamlit AppTest
```

Generated/reproducible artifacts (`data/processed/`, `models/*`, `reports/*.csv`, `reports/figures/*.png`, logs, screenshots, caches) are produced by the documented scripts and are not hand-edited; `.gitignore` excludes environments, caches, bytecode, and transient files.

## 15. Limitations

- Single frozen holdout (315 rows / 22 operating groups) at `random_state=42`; results are specific to this split and would shift on another draw.
- 1,503 experimental records only; does not cover every airfoil geometry, operating condition, aircraft configuration, or real-world environment. No inference about universal aircraft or cabin noise.
- Valid only within observed input ranges; out-of-range inputs are blocked. In-range but unobserved combinations may still be unreliable.
- Empirical bounds do not guarantee every combination of five in-range values corresponds to a physically observed wind-tunnel setup.
- Small holdout groups (n=11–17) make per-group residual means noisy.
- Permutation importance and residuals describe model behaviour on held-out data — not physical causes of airfoil noise; no causal claims are made.
- Not validated for aviation safety, certification, regulatory compliance, or production engineering decisions.
- No Release 2.0 features: no history, export, uploads, accounts, API, database, or cloud components.

## 16. Conclusion and future scope

Release 1.0 is complete: the data-to-model workflow is reproducible, model comparison was performed without operating-configuration leakage, the selected model (Experiment B + SVR RBF, CV MAE 1.8947) is saved as one trusted pipeline with metadata, holdout evaluation was performed exactly once (MAE 2.0673 dB, RMSE 3.1444 dB, R² 0.7702), and the Streamlit frontend accepts the five documented inputs and returns one predicted dB value. All 78 automated tests pass and the frontend acceptance checks pass.

Future scope is explicitly **not part of Release 1.0** and would require a new scope approval: persistent prediction history/user accounts, REST API or cloud hosting, database persistence, additional algorithms, SHAP analysis, downloadable reports, or multi-page dashboards. None of these were implemented.

## 17. References

- [R1] UCI Machine Learning Repository. “Airfoil Self-Noise”, dataset 291. https://doi.org/10.24432/C5VW2C — Creative Commons Attribution 4.0 International (CC BY 4.0). Official page: https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
- [R2] Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW2C
- [R3] Project data audit of `airfoil_self_noise.dat` (this repository): 1,503 rows × 6 columns, zero missing/duplicates, 106 operating configurations.

**Attribution text for derived data:** This project uses the Airfoil Self-Noise dataset by Brooks, T., Pope, D., and Marcolini, M. (1989), distributed by the UCI Machine Learning Repository, https://doi.org/10.24432/C5VW2C. The dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). The project adds descriptive column headers to the downloaded raw data for analysis.
