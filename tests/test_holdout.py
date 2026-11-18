"""Phase 8 final holdout evaluation tests.

Verifies prediction count, finite predictions, metric validity, holdout group
isolation, output files, and that training/holdout assignments and source data
are unmodified.
"""

import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from airfoil_noise.config import TARGET_COLUMN
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.groups import GROUP_ID_COLUMN, make_group_ids, split_train_holdout_groups
from airfoil_noise.holdout import (
    BEST_PARAMS,
    EVALUATION_JSON,
    PREDICTIONS_CSV,
    SELECTED_MODEL,
    SELECTED_VARIANT,
    evaluate_holdout,
)


@pytest.fixture(scope="module")
def result():
    return evaluate_holdout(load_raw_dataset())


def test_prediction_count_matches_holdout(result):
    df = pd.read_csv(PREDICTIONS_CSV)
    assert len(df) == 315


def test_predictions_finite(result):
    df = pd.read_csv(PREDICTIONS_CSV)
    assert np.isfinite(df["actual"].values).all()
    assert np.isfinite(df["predicted"].values).all()
    assert np.isfinite(df["residual"].values).all()


def test_residuals_equal_actual_minus_predicted(result):
    df = pd.read_csv(PREDICTIONS_CSV)
    expected_resid = df["actual"].values - df["predicted"].values
    np.testing.assert_allclose(df["residual"].values, expected_resid)


def test_holdout_metrics_valid(result):
    assert result["mae"] > 0
    assert result["rmse"] > 0
    assert np.isfinite(result["mae"])
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["r2"])


def test_output_files_exist():
    assert PREDICTIONS_CSV.exists()
    assert EVALUATION_JSON.exists()


def test_evaluation_json_records_configuration():
    data = json.load(open(EVALUATION_JSON))
    assert data["variant"] == SELECTED_VARIANT
    assert data["model"] == SELECTED_MODEL
    assert data["best_params"] == BEST_PARAMS
    assert data["n_holdout_observations"] == 315
    assert data["n_train_observations"] == 1188
    assert data["n_holdout_groups"] == 22
    assert data["n_train_groups"] == 84
    assert isinstance(data["holdout_mae"], (int, float))
    assert isinstance(data["holdout_rmse"], (int, float))
    assert isinstance(data["holdout_r2"], (int, float))
    assert data["holdout_mae"] > 0


def test_holdout_group_isolation():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    assert not set(train_groups) & set(holdout_groups)


def test_holdout_predictions_use_only_holdout_groups():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    preds = pd.read_csv(PREDICTIONS_CSV)
    assert set(preds[GROUP_ID_COLUMN].unique()) <= set(holdout_groups)
    assert not set(preds[GROUP_ID_COLUMN].unique()) & set(train_groups)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_dataset_unmodified():
    expected_sha = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
    path = REPO_ROOT / "data" / "raw" / "airfoil_self_noise.dat"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected_sha


def test_processed_dataset_row_count_unmodified():
    df = pd.read_csv(REPO_ROOT / "data" / "processed" / "airfoil_self_noise.csv")
    assert len(df) == 1503
    assert list(df.columns) == [
        "frequency_hz",
        "angle_of_attack_deg",
        "chord_length_m",
        "free_stream_velocity_m_s",
        "suction_side_displacement_thickness_m",
        "scaled_sound_pressure_level_db",
    ]
