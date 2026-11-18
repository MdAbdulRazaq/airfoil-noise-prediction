"""Phase 10 — final model artifact and metadata (Sections 6.3, 7.5, 10.11).

Fits the frozen Experiment B + SVR pipeline (StandardScaler + SVR kernel rbf,
C=100, epsilon=0.2, gamma=scale) on all 1,188 training observations. The
artifact accepts the original five raw input features in the documented order;
the log10(frequency_hz) transformation happens inside the artifact. Stages to
models/*.tmp, verifies the staged files in a fresh Python process, then
promotes them to models/best_model.joblib and models/model_metadata.json.

joblib.load is used only on the temp file this process wrote moments earlier in
the local models directory — never on user-supplied or network paths.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

from .config import EXPERIMENT_B_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN
from .features import experiment_b_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
from .groups import GROUP_ID_COLUMN, make_group_ids, split_train_holdout_groups
from .holdout import BEST_PARAMS

REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
ARTIFACT_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
ARTIFACT_TMP = MODELS_DIR / ".best_model.joblib.tmp"
METADATA_TMP = MODELS_DIR / ".model_metadata.json.tmp"
PAYLOAD_TMP = MODELS_DIR / ".verify_payload.json.tmp"

SUMMARY_CSV = REPORTS_DIR / "model_comparison.csv"
EVALUATION_JSON = REPORTS_DIR / "holdout_evaluation.json"

N_TRAIN_OBSERVATIONS = 1188
N_TRAIN_GROUPS = 84
N_HOLDOUT_OBSERVATIONS = 315
N_HOLDOUT_GROUPS = 22
RANDOM_SEED = 42

# Embedded verification script run in a fresh Python process.
_VERIFY_SCRIPT = r"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

artifact_path, metadata_path, payload_path = sys.argv[1:4]

with open(metadata_path, encoding="utf-8") as fh:
    metadata = json.load(fh)

# Only the file this session staged moments earlier in the local models
# directory — never user-supplied or network input.
bundle = joblib.load(artifact_path)

with open(payload_path, encoding="utf-8") as fh:
    payload = json.load(fh)

row = pd.DataFrame([payload["row"]], columns=payload["columns"])
prediction = float(bundle.predict(row)[0])
expected = float(payload["expected_prediction"])
max_abs_diff = abs(prediction - expected)
matches = max_abs_diff <= 1e-6 * max(1.0, abs(expected))

report = {
    "prediction": prediction,
    "expected_prediction": expected,
    "max_abs_diff": max_abs_diff,
    "finite": bool(np.isfinite(prediction)),
    "matches_in_memory": bool(matches),
    "n_input_columns": int(len(metadata.get("input_columns", []))),
    "input_columns": list(metadata.get("input_columns", [])),
    "model_features": list(metadata.get("model_features", [])),
}
print(json.dumps(report))
sys.exit(0 if report["finite"] and report["matches_in_memory"] else 1)
"""


def _recorded_cv_metrics() -> dict:
    """Frozen Phase 7 results for the selected configuration (single source of truth)."""
    summary = pd.read_csv(SUMMARY_CSV)
    row = summary[(summary["variant"] == "Experiment B") & (summary["model"] == "SVR")]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one Experiment B/SVR row in {SUMMARY_CSV}")
    record = row.iloc[0]
    return {
        "mean_cv_mae": float(record["mean_cv_mae"]),
        "std_cv_mae": float(record["std_cv_mae"]),
        "mean_cv_rmse": float(record["mean_cv_rmse"]),
        "std_cv_rmse": float(record["std_cv_rmse"]),
        "mean_cv_r2": float(record["mean_cv_r2"]),
        "std_cv_r2": float(record["std_cv_r2"]),
    }


def _recorded_holdout_metrics() -> dict:
    with open(EVALUATION_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _build_artifact_pipeline() -> Pipeline:
    """Preprocessing + regressor exactly as frozen in Phases 7 and 8.

    Columns enter in the raw documented order; frequency_hz is log10-scaled
    inside the artifact, then all five internal features are standardized and
    passed to SVR(kernel="rbf", C=100, epsilon=0.2, gamma="scale").
    ColumnTransformer output order is [log10_frequency_hz] + the four
    passthrough columns, i.e. exactly EXPERIMENT_B_FEATURES.
    """
    passthrough_columns = [c for c in FEATURE_COLUMNS if c != "frequency_hz"]
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("log10_frequency", FunctionTransformer(np.log10), ["frequency_hz"]),
                        ("passthrough", "passthrough", passthrough_columns),
                    ]
                ),
            ),
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf", **BEST_PARAMS)),
        ]
    )


def build_artifact(df: pd.DataFrame):
    """Fit the frozen pipeline on the full 1188-row training block and assemble
    matching metadata. Returns (pipeline, metadata, smoke_row_frame, expected_prediction).
    Nothing is written to disk here."""
    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    train_mask = ids.isin(train_groups)
    train_df = df.loc[train_mask].reset_index(drop=True)
    if len(train_df) != N_TRAIN_OBSERVATIONS:
        raise ValueError(f"Expected {N_TRAIN_OBSERVATIONS} training observations, got {len(train_df)}")

    cv_metrics = _recorded_cv_metrics()
    holdout_metrics = _recorded_holdout_metrics()
    for key in ("holdout_mae", "holdout_rmse", "holdout_r2"):
        if key not in holdout_metrics:
            raise ValueError(f"Holdout evaluation missing {key}")

    X_train = train_df[FEATURE_COLUMNS].reset_index(drop=True)
    y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
    pipe = _build_artifact_pipeline()
    pipe.fit(X_train, y_train)

    smoke_row = train_df.iloc[:1][FEATURE_COLUMNS].reset_index(drop=True)
    expected_prediction = float(pipe.predict(smoke_row)[0])

    metadata = {
        "project_name": "airfoil-noise-prediction",
        "project_title": "Airfoil Noise Predictor",
        "dataset_source": "UCI Airfoil Self-Noise dataset #291, NASA wind tunnel experiment (airfoil self-noise)",
        "dataset_doi": "10.24432/C5VW2C",
        "raw_dataset_file": "data/raw/airfoil_self_noise.dat",
        "target_column": TARGET_COLUMN,
        "input_columns": list(FEATURE_COLUMNS),
        "model_features": list(EXPERIMENT_B_FEATURES),
        "feature_variant": "Experiment B",
        "frequency_transformation": "log10(frequency_hz)",
        "model": "SVR",
        "kernel": "rbf",
        "hyperparameters": dict(BEST_PARAMS),
        "preprocessing_steps": ["log10(frequency_hz)", "StandardScaler", "SVR(kernel=rbf)"],
        "validation": {
            "train_holdout_split": {
                "method": "GroupShuffleSplit",
                "test_size": 0.20,
                "random_state": RANDOM_SEED,
                "groups": "operating configurations (frequency excluded)",
            },
            "cross_validation": {
                "method": "GroupKFold",
                "n_splits": 5,
                "groups": "training groups only",
            },
            "selection_metric": "lowest mean GroupKFold CV MAE",
            "holdout_evaluated_once": True,
        },
        "selection_basis": "lowest mean GroupKFold cross-validation MAE",
        "random_seed": RANDOM_SEED,
        "n_train_observations": int(len(train_df)),
        "n_train_groups": int(len(train_groups)),
        "n_holdout_observations": int(len(df) - len(train_df)),
        "n_holdout_groups": int(len(holdout_groups)),
        "n_total_observations": int(len(df)),
        "cv_mae": cv_metrics["mean_cv_mae"],
        "cv_mae_std": cv_metrics["std_cv_mae"],
        "cv_rmse": cv_metrics["mean_cv_rmse"],
        "cv_rmse_std": cv_metrics["std_cv_rmse"],
        "cv_r2": cv_metrics["mean_cv_r2"],
        "cv_r2_std": cv_metrics["std_cv_r2"],
        "holdout_mae": float(holdout_metrics["holdout_mae"]),
        "holdout_rmse": float(holdout_metrics["holdout_rmse"]),
        "holdout_r2": float(holdout_metrics["holdout_r2"]),
        "python_version": sys.version,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
    }
    return pipe, metadata, smoke_row, expected_prediction


def finalize_artifact(df: pd.DataFrame) -> dict:
    """Fit, stage to .tmp, verify in a fresh Python process, then promote.

    The staged artifact is reloaded in a fresh process, checked against the
    metadata, used to predict one real dataset row (finite, matching the
    in-memory fitted pipeline), and only then promoted via os.replace.
    """
    pipe, metadata, smoke_row, expected_prediction = build_artifact(df)
    payload = {
        "columns": list(FEATURE_COLUMNS),
        "row": {col: float(value) for col, value in smoke_row.iloc[0].items()},
        "expected_prediction": expected_prediction,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACT_TMP)
    with open(METADATA_TMP, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    with open(PAYLOAD_TMP, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _VERIFY_SCRIPT,
            str(ARTIFACT_TMP),
            str(METADATA_TMP),
            str(PAYLOAD_TMP),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Staged artifact verification failed:\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    report = json.loads(completed.stdout)

    os.replace(ARTIFACT_TMP, ARTIFACT_PATH)
    os.replace(METADATA_TMP, METADATA_PATH)
    PAYLOAD_TMP.unlink(missing_ok=True)
    return {"artifact": ARTIFACT_PATH, "metadata": METADATA_PATH, "verification": report}


def main() -> None:
    from .data import load_raw_dataset

    result = finalize_artifact(load_raw_dataset())
    print("Final model artifact staged, verified and promoted:")
    print("  artifact:", result["artifact"])
    print("  metadata:", result["metadata"])
    print("  verification:", result["verification"])


if __name__ == "__main__":
    main()
