# Release Acceptance Checklist — Airfoil Noise Predictor 1.0

Basis: project specification Sections 11.1 (test inventory), 11.2 (manual acceptance), 11.3 (release quality gates).
All results below were verified by executing the project's own tests and acceptance scripts (Phase 13, 2026-11-19).
Verdicts: PASS / FAIL / NOT APPLICABLE.

## 1. Section 11.3 release quality gates

| # | Gate | Requirement (spec) | Verdict | Evidence |
|---|---|---|---|---|
| G1 | Data gate | Schema, row count, missing values, duplicates and units documented | PASS | `tests/test_data.py`, `tests/test_validation.py` pass; 1,503 rows × 6 cols; 0 missing; 0 duplicates; units documented in README §3 and reports |
| G2 | Model gate | Group-aware CV + holdout methodology recorded; actual metrics generated | PASS | `reports/group_validation_analysis.md`; `reports/model_comparison.csv`; `reports/holdout_evaluation.json`; `models/model_metadata.json` (CV MAE 1.8947; holdout MAE 2.0673 / RMSE 3.1444 / R² 0.7702) |
| G3 | Artifact gate | Model and metadata load successfully in a clean process | PASS | `tests/test_artifact.py` (9/9 pass): joblib reload, metadata compatibility, prediction consistency pre/post serialization |
| G4 | Frontend gate | Five-input / one-output flow works locally | PASS | `apptest_check.py` 13/13 passed; live `streamlit run app/streamlit_app.py` returned HTTP 200 |
| G5 | Security gate | No untrusted serialized model loading; input validation enabled; no secrets present | PASS | `predict.py` loads only local trusted artifact with metadata verification; bounds + type/finiteness validation enabled; scan found no `.env`, `*.secret`, or credential files |
| G6 | Documentation gate | README, source citation, limitations, reproduction steps complete | PASS | `README.md`, `FINAL_PROJECT_REPORT.md` (DOI 10.24432/C5VW2C, CC BY 4.0 attribution, limitations §15, install/run/reproduce §13–17) |
| G7 | Scope gate | No unapproved functionality added | PASS | `apptest_check.py` asserts no Release 2.0 features; UI has exactly five inputs + one Predict button + one dB result; no auth, database, API, export, history, or cloud components |

**Release gates: 7 PASS, 0 FAIL, 0 NOT APPLICABLE.**

## 2. Section 11.1 automated test inventory

| ID | Test | Expected | Verdict |
|---|---|---|---|
| T-01 | Load raw .dat | 1,503 rows, six canonical columns | PASS (`tests/test_data.py`) |
| T-02 | Validate supplied dataset | No missing, non-finite, duplicate rows | PASS (`tests/test_validation.py`) |
| T-03 | Reject altered column order/name | Clear schema error | PASS (`tests/test_validation.py`) |
| T-04 | Create operating groups | 106 groups for baseline file | PASS (`tests/test_groups.py`) |
| T-05 | Check split isolation | No group id in both train and test | PASS (`tests/test_groups.py`) |
| T-06 | Fit each candidate estimator | All five complete, no leakage errors | PASS (`tests/test_models.py`) |
| T-07 | Run evaluation metrics | MAE, RMSE, R² finite | PASS (`tests/test_holdout.py`) |
| T-08 | Save and reload model | Same output within tolerance | PASS (`tests/test_artifact.py`) |
| T-09 | Valid prediction input | One finite float | PASS (`tests/test_prediction.py`) |
| T-10 | Missing/extra key | Clear validation error | PASS (`tests/test_prediction.py`) |
| T-11 | NaN/infinity/out-of-range | Rejected before `predict` | PASS (`tests/test_prediction.py`) |
| T-12 | Missing model artifact | Clear setup message | PASS (`tests/test_prediction.py` + `apptest_check.py`) |
| T-13 | Streamlit smoke test | Page starts; valid input → one dB value | PASS (`apptest_check.py` + live server smoke) |

**Full suite: 78 passed, 0 failed (pytest -q, 2026-11-19).** Source `.dat` hash matches the pinned value in `tests/test_holdout.py` (`74c75fd7…`).

## 3. Section 11.2 manual acceptance steps

| # | Step | Verdict |
|---|---|---|
| 1 | Open app from documented command in a fresh terminal | PASS (`streamlit run app/streamlit_app.py`, HTTP 200) |
| 2 | Each field label and unit matches the data dictionary | PASS (Frequency/Hz … Displacement thickness/m) |
| 3 | Valid dataset row → prediction | PASS (first row → 126.40 dB) |
| 4 | One field below minimum → validation blocks prediction | PASS (widget min_value + backend bounds check) |
| 5 | One field above maximum → validation blocks prediction | PASS (widget max_value + backend bounds check) |
| 6 | Displayed value labelled scaled sound pressure level (dB) | PASS ("Predicted scaled sound pressure level: 126.40 dB") |
| 7 | No prediction history, account, export, or unapproved UI | PASS (single page; apptest asserts no Release 2.0 strings) |

## 4. Scope statement (spec §13.3)

Release 1.0 contains no deferred Release 2.0 functionality: no accounts/authentication, no database, no REST API, no cloud hosting, no prediction history, no file uploads, no exports, no additional algorithms, no architecture redesign. Frozen holdout metrics unchanged; final model not retrained or replaced.

**Release 1.0 is accepted.**
