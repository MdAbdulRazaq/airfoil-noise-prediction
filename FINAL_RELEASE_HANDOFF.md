# Final Release Handoff — Airfoil Noise Predictor

## 1. Project overview
Airfoil Noise Predictor — Release 1.0. A reproducible, local-only ML regression prototype that estimates the scaled sound pressure level (dB) of airfoil self-noise from five wind-tunnel test inputs (frequency, angle of attack, chord length, free-stream velocity, suction-side displacement thickness), trained on the UCI Airfoil Self-Noise dataset (1,503 observations). Backend: frozen scikit-learn pipeline (Experiment B + SVR RBF). Frontend: single-page Streamlit app with five bounded inputs and one 2-decimal dB prediction. No accounts, database, REST API, cloud component, or prediction history.

## 2. Release version
1.0.0 (`pyproject.toml` version; commit tag: root-commit)

## 3. Repository location
`D:\Project Aero\airfoil-noise-prediction`

## 4. Git initialization status
Local Git repository initialized in the project root (`.git/`). One commit on branch `master`. No remote configured, no push performed.

## 5. Commit hash
`c920ce8254d14b86aaa6a90960f0b68865b48d75` — `release: Airfoil Noise Prediction v1.0`

## 6. Number of tracked files
61 files (1 commit, 7,209 insertions, 0 deletions)

## 7. Main folder structure
```
airfoil-noise-prediction/              # Release 1.0 repository
├── .gitignore                         # excludes .venv, caches, pyc/log/temp, *.egg-info/, data/processed/
├── FINAL_PROJECT_REPORT.md            # academic project report
├── FINAL_RELEASE_HANDOFF.md           # this document
├── Airfoil_Noise_Predictor_Presentation_Release_1.0.pptx
├── README.md                          # setup, run commands, dataset citation, results, limitations
├── RELEASE_CHECKLIST.md               # release quality gates (7 PASS)
├── requirements-lock.txt              # pinned dependency versions
├── pyproject.toml                     # packaging: airfoil-noise-prediction 1.0.0, Python >= 3.11
├── run_tests.ps1                      # PowerShell test runner
├── apptest_check.py                   # Streamlit AppTest frontend acceptance
├── cdp_driver.py / clean_screenshot.py / screenshot_driver.py
├── data/raw/airfoil_self_noise.dat    # canonical raw source — immutable, hash pinned
├── data/processed/                    # gitignored reproducible processed copy
├── models/                            # tracked deliverables
│   ├── best_model.joblib              # frozen Release 1.0 pipeline artifact
│   └── model_metadata.json            # DOI, metrics, hyperparameters, seed, package versions
├── notebooks/                         # tracked deliverables
│   ├── 01_data_understanding.ipynb
│   └── 02_exploratory_analysis.ipynb
├── reports/                           # tracked deliverables
│   ├── evidence/ (5 acceptance screenshots)
│   ├── figures/ (11 PNG figures)
│   ├── model_comparison.csv
│   ├── model_comparison_folds.csv
│   ├── holdout_evaluation.json        # frozen holdout metrics
│   ├── holdout_predictions.csv        # 315 rows: actual/predicted/residual/group
│   ├── permutation_feature_importance.csv
│   ├── residuals_by_operating_group.csv
│   ├── feature_variant_analysis.md
│   ├── group_validation_analysis.md
│   ├── residual_analysis.md
│   └── group_validation_analysis.md
├── src/airfoil_noise/                 # backend package (11 modules)
│   ├── artifact.py config.py data.py explainability.py features.py
│   ├── groups.py holdout.py models.py predict.py validation.py
└── tests/                             # 9 test files, 78 tests — all tracked
```

## 8. Final model information
- Variant: Experiment B (`log10_frequency_hz` + four unchanged inputs)
- Model: SVR, kernel `rbf`
- Hyperparameters: C=100.0, ε=0.2, γ="scale"
- Preprocessing: log10(frequency_hz) → StandardScaler → SVR(RBF)
- Selection basis: lowest mean GroupKFold CV MAE (1.8947 ± 0.3294)
- Training: 84 groups / 1,188 observations; random seed 42
- Artifact: `models/best_model.joblib` + `models/model_metadata.json`
- Python version at artifact creation: 3.14.3

## 9. Actual holdout metrics (evaluated once, frozen)
| Measure | Value |
|---|---|
| Holdout MAE | 2.0673 dB |
| Holdout RMSE | 3.1444 dB |
| Holdout R² | 0.7702 |
| Holdout observations / groups | 315 / 22 |
| Mean GroupKFold CV MAE | 1.8947 (std 0.3294) |
| Mean GroupKFold CV RMSE | 2.6272 (std 0.5015) |
| Mean GroupKFold CV R² | 0.8556 (std 0.0422) |

## 10. Automated testing results
Suite: `pytest -q` — **78 passed, 0 failed** (9 test files), run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg`.
Coverage: data ingest/validation, operating groups, group-aware split isolation, five-model fit, evaluation metrics, save/reload model, prediction input validation, artifact reproducibility, source-dataset hash pin.

## 11. Frontend acceptance results
`apptest_check.py` (Streamlit AppTest) — **13/13 passed**. Live `streamlit run app/streamlit_app.py` returned HTTP 200. Five bounded inputs, one Predict button, one 2-decimal dB result; no Release 2.0 UI strings.

## 12. Documentation inventory
- `README.md` — overview, methodology, stack, installation, limitations, reproducibility
- `FINAL_PROJECT_REPORT.md` — full 17-section academic report
- `RELEASE_CHECKLIST.md` — 7 release quality gates, all PASS
- `reports/*.md` — feature variant analysis, group validation analysis, residual analysis
- `models/model_metadata.json` — artifact metadata + frozen metrics
- UCI DOI attribution: `10.24432/C5VW2C`; license: CC BY 4.0

## 13. Dependency / runtime information
Python >= 3.11 (validated on 3.14.3). Runtime: pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0, streamlit 1.67.0. Dev: pytest 9.2.0, jupyter. Exact versions in `requirements-lock.txt` and `models/model_metadata.json`.

## 14. Dataset attribution
UCI Machine Learning Repository, Airfoil Self-Noise dataset #291, DOI `10.24432/C5VW2C`, Creative Commons Attribution 4.0 International (CC BY 4.0). Canonical raw file `data/raw/airfoil_self_noise.dat` is immutable and hash-pinned (`74c75fd7…`).

## 15. Git exclusions
`.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.ipynb_checkpoints/`, `data/processed/`, `.env`, `*.log`, `*.swp`, `*~`, `Thumbs.db`, `.DS_Store`, `.vscode/`, `.idea/`, `*.egg-info/`. `models/`, `reports/`, `tests/`/`notebooks/`, schema, and deliverables remain version-controlled.

## 16. Known limitations
- Single frozen holdout (315 rows / 22 groups) at `random_state=42`; results split-specific.
- 1,503 experimental records only; not every airfoil geometry / operating condition covered.
- Valid only within observed input envelope; in-range but unobserved combinations may be unreliable.
- Small holdout groups (n=11–17) make per-group residual means noisy.
- Permutation importance / residual patterns describe model behaviour, not physical causality; no causal claims.
- Not validated for aviation safety, certification, regulatory compliance, or production engineering.
- No Release 2.0 features: no history, export, uploads, accounts, API, database, cloud.

## 17. Final working-tree status
Clean — `git status` reports "nothing to commit, working tree clean" on branch `master` at commit `c920ce8`.

## 18. GitHub publishing
Not performed. No remote created, no push, no visibility change. Remote publishing is a separate explicitly authorized task.
