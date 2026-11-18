"""Feature engineering experiments (Phase 5).

Covers correct log10 transformation, invalid inputs, missing/NaN/infinity,
determinism, and the exact five-feature ordering of both experiments.
"""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.config import EXPECTED_ROW_COUNT
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.features import (
    EXPERIMENT_A_FEATURES,
    EXPERIMENT_B_FEATURES,
    LOG10_FREQUENCY_COLUMN,
    FeatureError,
    add_log10_frequency,
    experiment_a_frame,
    experiment_b_frame,
)


def test_experiment_a_has_five_original_features_in_order():
    assert EXPERIMENT_A_FEATURES == [
        "frequency_hz",
        "angle_of_attack_deg",
        "chord_length_m",
        "free_stream_velocity_m_s",
        "suction_side_displacement_thickness_m",
    ]


def test_experiment_b_has_five_features_log10_first_in_order():
    assert EXPERIMENT_B_FEATURES == [
        "log10_frequency_hz",
        "angle_of_attack_deg",
        "chord_length_m",
        "free_stream_velocity_m_s",
        "suction_side_displacement_thickness_m",
    ]


def test_add_log10_frequency_values_correct():
    df = pd.DataFrame({"frequency_hz": [1.0, 10.0, 100.0, 1000.0]})
    out = add_log10_frequency(df)
    expected = np.log10(df["frequency_hz"])
    pd.testing.assert_series_equal(
        out[LOG10_FREQUENCY_COLUMN], expected, check_names=False
    )
    # Original column retained, exactly one new column.
    assert list(out.columns) == ["frequency_hz", LOG10_FREQUENCY_COLUMN]


def test_transformation_does_not_mutate_input():
    df = pd.DataFrame({"frequency_hz": [100.0, 1000.0]})
    before = df.copy(deep=True)
    add_log10_frequency(df)
    pd.testing.assert_frame_equal(df, before)


def test_transformation_is_deterministic():
    df = load_raw_dataset()
    out1 = add_log10_frequency(df)
    out2 = add_log10_frequency(df)
    pd.testing.assert_frame_equal(out1, out2)


def test_negative_frequency_rejected():
    df = pd.DataFrame({"frequency_hz": [100.0, -50.0]})
    with pytest.raises(FeatureError, match="non-positive"):
        add_log10_frequency(df)


def test_zero_frequency_rejected():
    df = pd.DataFrame({"frequency_hz": [0.0]})
    with pytest.raises(FeatureError, match="non-positive"):
        add_log10_frequency(df)


def test_nan_frequency_rejected():
    df = pd.DataFrame({"frequency_hz": [100.0, np.nan]})
    with pytest.raises(FeatureError, match="missing"):
        add_log10_frequency(df)


def test_inf_frequency_rejected():
    df = pd.DataFrame({"frequency_hz": [100.0, np.inf, -np.inf]})
    with pytest.raises(FeatureError, match="non-finite"):
        add_log10_frequency(df)


def test_missing_frequency_column_rejected():
    df = pd.DataFrame({"angle_of_attack_deg": [1.0]})
    with pytest.raises(FeatureError, match="'frequency_hz'"):
        add_log10_frequency(df)


def test_experiment_b_frame_properties():
    df = load_raw_dataset()
    out = experiment_b_frame(df)
    assert list(out.columns) == EXPERIMENT_B_FEATURES
    assert out.shape[1] == 5
    assert len(out) == EXPECTED_ROW_COUNT
    assert "frequency_hz" not in out.columns
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    # All values finite and within plausible log10 frequency range.
    assert out[LOG10_FREQUENCY_COLUMN].min() >= 2.0  # log10(100)
    assert out[LOG10_FREQUENCY_COLUMN].max() <= 5.0  # log10(100000)


def test_experiment_a_frame_properties():
    df = load_raw_dataset()
    out = experiment_a_frame(df)
    assert list(out.columns) == EXPERIMENT_A_FEATURES
    assert out.shape[1] == 5
    assert len(out) == EXPECTED_ROW_COUNT
    assert np.isfinite(out.to_numpy(dtype=float)).all()
