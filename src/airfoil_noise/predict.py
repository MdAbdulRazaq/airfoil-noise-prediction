"""Phase 11 — prediction backend over the frozen Phase 10 model artifact.

``load_model()`` loads and verifies ``models/best_model.joblib`` plus
``models/model_metadata.json``. ``validate_prediction_input()`` checks the
five raw features against ``config.FEATURE_BOUNDS``.
``predict_sound_pressure()`` builds a single-row DataFrame in
``FEATURE_COLUMNS`` order, runs the saved pipeline (the log10 frequency
transform lives inside it), and returns one finite Python float.

No Streamlit dependency; no retraining or retuning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifact import ARTIFACT_PATH, METADATA_PATH
from .config import (
    EXPERIMENT_B_FEATURES,
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from .holdout import BEST_PARAMS

EXPECTED_MODEL = "SVR"
EXPECTED_KERNEL = "rbf"
EXPECTED_VARIANT = "Experiment B"
EXPECTED_TRANSFORM = "log10(frequency_hz)"
EXPECTED_PREPROCESSING = ["log10(frequency_hz)", "StandardScaler", "SVR(kernel=rbf)"]


def load_model():
    """Load and verify the trusted model artifact; return the fitted pipeline.

    Raises FileNotFoundError when an artifact is missing and ValueError when
    the artifact or its metadata is invalid or incompatible with the
    approved project configuration.
    """
    missing = [str(path) for path in (ARTIFACT_PATH, METADATA_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(missing)
            + "\nRun Phase 10 (src/airfoil_noise/artifact.py) to produce them."
        )

    try:
        with open(METADATA_PATH, encoding="utf-8") as fh:
            metadata = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid model metadata {METADATA_PATH}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid model metadata {METADATA_PATH}: expected a JSON object")

    _verify_metadata(metadata)

    try:
        return joblib.load(ARTIFACT_PATH)
    except Exception as exc:  # corrupt pickle/joblib payloads raise many shapes
        raise ValueError(f"Failed to load model artifact {ARTIFACT_PATH}: {exc}") from exc


def _verify_metadata(metadata: dict) -> None:
    """Reject metadata that disagrees with the approved project configuration."""
    checks = [
        ("target_column", metadata.get("target_column"), TARGET_COLUMN),
        ("input_columns", metadata.get("input_columns"), list(FEATURE_COLUMNS)),
        ("model_features", metadata.get("model_features"), list(EXPERIMENT_B_FEATURES)),
        ("feature_variant", metadata.get("feature_variant"), EXPECTED_VARIANT),
        ("model", metadata.get("model"), EXPECTED_MODEL),
        ("kernel", metadata.get("kernel"), EXPECTED_KERNEL),
        ("frequency_transformation", metadata.get("frequency_transformation"), EXPECTED_TRANSFORM),
        ("preprocessing_steps", metadata.get("preprocessing_steps"), EXPECTED_PREPROCESSING),
        ("hyperparameters", metadata.get("hyperparameters"), dict(BEST_PARAMS)),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"Incompatible model metadata: {name}={actual!r}, expected {expected!r}"
            )


def validate_prediction_input(inputs: dict) -> dict:
    """Validate exactly the five named raw features and return them as floats
    in the documented FEATURE_COLUMNS order."""
    if not isinstance(inputs, dict):
        raise ValueError("Prediction input must be a dict of the five named features")

    keys = set(inputs.keys())
    expected = set(FEATURE_COLUMNS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise ValueError(
            "Exactly the five named inputs are required: "
            f"{FEATURE_COLUMNS}"
            + (f"; missing: {missing}" if missing else "")
            + (f"; extra: {extra}" if extra else "")
        )

    cleaned: dict = {}
    for column in FEATURE_COLUMNS:  # documented input order
        value = inputs[column]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"'{column}' must be a real number, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"'{column}' must be finite, got {value!r}")
        low, high = FEATURE_BOUNDS[column]
        if value < low or value > high:
            raise ValueError(
                f"'{column}'={value} is outside the observed bounds [{low}, {high}]"
            )
        cleaned[column] = value
    return cleaned


def predict_sound_pressure(inputs: dict) -> float:
    """Return one finite scaled sound pressure level (dB) prediction for the
    five raw input features. Raw values flow into the saved pipeline
    unchanged; log10(frequency_hz) is computed inside it."""
    clean = validate_prediction_input(inputs)
    model = load_model()
    frame = pd.DataFrame([clean], columns=list(FEATURE_COLUMNS))
    prediction = float(model.predict(frame)[0])
    if not math.isfinite(prediction):
        raise ValueError(f"Model produced a non-finite prediction: {prediction!r}")
    return prediction
