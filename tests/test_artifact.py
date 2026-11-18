"""Phase 10 final model artifact and metadata tests.

Covers artifact existence, joblib reload, exact raw feature order, internal
log10 frequency transformation, finite predictions, metadata schema and actual
values, and prediction consistency before/after serialization.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from airfoil_noise.artifact import ARTIFACT_PATH, METADATA_PATH, finalize_artifact
from airfoil_noise.config import FEATURE_COLUMNS, TARGET_COLUMN
from airfoil_noise.data import load_raw_dataset


@pytest.fixture(scope="module")
def artifact_result():
    return finalize_artifact(load_raw_dataset())


@pytest.fixture(scope="module")
def bundled_pipeline():
    return joblib.load(ARTIFACT_PATH)


@pytest.fixture(scope="module")
def metadata():
    return json.load(open(METADATA_PATH, encoding="utf-8"))


def test_artifact_exists():
    assert ARTIFACT_PATH.exists() and ARTIFACT_PATH.is_file()
    assert METADATA_PATH.exists() and METADATA_PATH.is_file()


def test_joblib_reload_successful(bundled_pipeline):
    from sklearn.pipeline import Pipeline
    assert isinstance(bundled_pipeline, Pipeline)


def test_artifact_feature_order_matches_metadata(metadata, bundled_pipeline):
    assert metadata["input_columns"] == list(FEATURE_COLUMNS)
    transformed = bundled_pipeline.named_steps["features"].transform(
        pd.DataFrame([{col: 1.0 for col in FEATURE_COLUMNS}])
    )
    assert transformed.shape[1] == len(FEATURE_COLUMNS)


def test_internal_log10_frequency_transformation(bundled_pipeline):
    row = pd.DataFrame([{col: 100.0 for col in FEATURE_COLUMNS}])
    transformed = bundled_pipeline.named_steps["features"].transform(row)
    frequency_index = list(FEATURE_COLUMNS).index("frequency_hz")
    expected_log10 = float(np.log10(100.0))
    assert abs(transformed[0, frequency_index] - expected_log10) < 1e-9


def test_prediction_is_finite(bundled_pipeline):
    df = load_raw_dataset()
    row = df[FEATURE_COLUMNS].iloc[:1].copy()
    prediction = float(bundled_pipeline.predict(row)[0])
    assert np.isfinite(prediction)


def test_metadata_schema_and_values(metadata):
    assert metadata["project_name"] == "airfoil-noise-prediction"
    assert metadata["dataset_doi"] == "10.24432/C5VW2C"
    assert metadata["target_column"] == TARGET_COLUMN
    assert metadata["feature_variant"] == "Experiment B"
    assert metadata["frequency_transformation"] == "log10(frequency_hz)"
    assert metadata["model"] == "SVR"
    assert metadata["kernel"] == "rbf"
    assert metadata["hyperparameters"] == {"C": 100.0, "epsilon": 0.2, "gamma": "scale"}
    assert metadata["model_features"] == [
        "log10_frequency_hz",
        "angle_of_attack_deg",
        "chord_length_m",
        "free_stream_velocity_m_s",
        "suction_side_displacement_thickness_m",
    ]
    assert metadata["n_train_observations"] == 1188
    assert metadata["n_train_groups"] == 84
    assert metadata["n_holdout_observations"] == 315
    assert metadata["n_holdout_groups"] == 22
    assert metadata["random_seed"] == 42
    assert metadata["cv_mae"] == pytest.approx(1.8947, abs=1e-4)
    assert metadata["holdout_mae"] == pytest.approx(2.0673, abs=1e-4)
    assert "GroupShuffleSplit" in metadata["validation"]["train_holdout_split"]["method"]
    assert metadata["validation"]["cross_validation"]["method"] == "GroupKFold"
    assert metadata["validation"]["selection_metric"] == "lowest mean GroupKFold CV MAE"
    assert metadata["validation"]["holdout_evaluated_once"] is True
    assert set(metadata["package_versions"]) >= {"numpy", "pandas", "scikit-learn", "joblib"}


def test_prediction_consistency_before_and_after_serialization(
    artifact_result, bundled_pipeline, metadata
):
    df = load_raw_dataset()
    row = df[FEATURE_COLUMNS].iloc[:1].copy()
    before = float(bundled_pipeline.predict(row)[0])
    assert np.isfinite(before)

    fresh = joblib.load(ARTIFACT_PATH)
    after = float(fresh.predict(row)[0])
    assert np.isfinite(after)
    assert abs(after - before) < 1e-9
    assert artifact_result["verification"]["finite"] is True
    assert artifact_result["verification"]["matches_in_memory"] is True
