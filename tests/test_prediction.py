"""Phase 11 prediction backend tests (Section 10.12).

Covers model loading, metadata compatibility, raw feature ordering, a finite
prediction from a real dataset row, rejection of missing/extra/non-numeric/NaN/
infinite/out-of-bounds inputs, clear errors for missing artifacts, and
prediction consistency across repeated calls.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from airfoil_noise.config import FEATURE_BOUNDS, FEATURE_COLUMNS
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.predict import (
    ARTIFACT_PATH,
    METADATA_PATH,
    load_model,
    predict_sound_pressure,
    validate_prediction_input,
)

FIRST_ROW = load_raw_dataset()[FEATURE_COLUMNS].iloc[:1].copy()
VALID_INPUT = {col: float(FIRST_ROW.iloc[0][col]) for col in FEATURE_COLUMNS}


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_load_model_returns_fitted_pipeline(model):
    assert isinstance(model, Pipeline)


def test_metadata_compatibility_check_passes():
    # load_model() verifies metadata against config; no raise means compatible.
    model = load_model()
    metadata = json.load(open(METADATA_PATH, encoding="utf-8"))
    assert metadata["input_columns"] == list(FEATURE_COLUMNS)
    assert model.predict(pd.DataFrame([VALID_INPUT], columns=FEATURE_COLUMNS))[0] is not None


def test_real_dataset_row_produces_finite_prediction(model):
    prediction = float(model.predict(FIRST_ROW)[0])
    assert np.isfinite(prediction)


def test_predict_sound_pressure_returns_finite_float(model):
    prediction = predict_sound_pressure(VALID_INPUT)
    assert isinstance(prediction, float)
    assert np.isfinite(prediction)


def test_predict_uses_documented_feature_order():
    # Same values, scrambled key order: dict key order must not matter; the
    # DataFrame is rebuilt in FEATURE_COLUMNS order inside predict_sound_pressure.
    scrambled = {col: VALID_INPUT[col] for col in reversed(FEATURE_COLUMNS)}
    straight = predict_sound_pressure(VALID_INPUT)
    reordered = predict_sound_pressure(scrambled)
    assert straight == reordered


def test_missing_input_rejected():
    incomplete = dict(VALID_INPUT)
    del incomplete[FEATURE_COLUMNS[0]]
    with pytest.raises(ValueError, match="missing"):
        validate_prediction_input(incomplete)


def test_additional_input_rejected():
    extra = dict(VALID_INPUT, extra_feature=1.0)
    with pytest.raises(ValueError, match="extra"):
        validate_prediction_input(extra)


def test_non_numeric_input_rejected():
    bad = dict(VALID_INPUT)
    bad[FEATURE_COLUMNS[0]] = "800"
    with pytest.raises(ValueError, match="real number"):
        validate_prediction_input(bad)


def test_bool_input_rejected():
    bad = dict(VALID_INPUT)
    bad[FEATURE_COLUMNS[0]] = True
    with pytest.raises(ValueError, match="real number"):
        validate_prediction_input(bad)


def test_nan_rejected():
    bad = dict(VALID_INPUT)
    bad[FEATURE_COLUMNS[0]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_prediction_input(bad)


def test_infinity_rejected():
    bad = dict(VALID_INPUT)
    bad[FEATURE_COLUMNS[0]] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_prediction_input(bad)


def test_below_minimum_rejected():
    bad = dict(VALID_INPUT)
    bad["frequency_hz"] = FEATURE_BOUNDS["frequency_hz"][0] - 1e-6
    with pytest.raises(ValueError, match="outside the observed bounds"):
        validate_prediction_input(bad)


def test_above_maximum_rejected():
    bad = dict(VALID_INPUT)
    bad["frequency_hz"] = FEATURE_BOUNDS["frequency_hz"][1] + 1e-6
    with pytest.raises(ValueError, match="outside the observed bounds"):
        validate_prediction_input(bad)


def test_exact_bounds_accepted():
    at_bounds = dict(VALID_INPUT)
    at_bounds["frequency_hz"] = FEATURE_BOUNDS["frequency_hz"][0]
    at_bounds["angle_of_attack_deg"] = FEATURE_BOUNDS["angle_of_attack_deg"][1]
    validated = validate_prediction_input(at_bounds)
    assert validated["frequency_hz"] == FEATURE_BOUNDS["frequency_hz"][0]


def test_missing_model_artifact_produces_clear_error(monkeypatch):
    monkeypatch.setattr("airfoil_noise.predict.ARTIFACT_PATH", Path("no/such/model.joblib"))
    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        load_model()


def test_missing_metadata_produces_clear_error(monkeypatch):
    monkeypatch.setattr("airfoil_noise.predict.METADATA_PATH", Path("no/such/metadata.json"))
    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        load_model()


def test_prediction_consistent_across_repeated_calls(model):
    first = predict_sound_pressure(VALID_INPUT)
    second = predict_sound_pressure(VALID_INPUT)
    assert first == second
    # Reloaded artifact agrees too.
    fresh = joblib.load(ARTIFACT_PATH)
    assert abs(float(fresh.predict(pd.DataFrame([VALID_INPUT], columns=FEATURE_COLUMNS))[0]) - first) < 1e-12
