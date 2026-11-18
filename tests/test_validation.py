"""T-02: Validate the supplied dataset (no missing, non-finite, duplicate rows).
T-03: Reject altered column order/name with a clear schema error.
"""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.config import ALL_COLUMNS, EXPECTED_ROW_COUNT
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.validation import ValidationError, validate_dataset


def test_t02_validate_supplied_dataset_passes():
    df = load_raw_dataset()
    validate_dataset(df)
    assert df.isna().sum().sum() == 0
    assert np.isfinite(df[ALL_COLUMNS].to_numpy(dtype=float)).all()
    assert not df.duplicated().any()
    assert len(df) == EXPECTED_ROW_COUNT


def test_t02_reject_missing_values():
    df = load_raw_dataset()
    df.iloc[0, 0] = np.nan
    with pytest.raises(ValidationError, match="missing cells"):
        validate_dataset(df)


def test_t02_reject_non_finite_values():
    df = load_raw_dataset()
    df.iloc[0, 1] = np.inf  # angle_of_attack_deg is float64
    with pytest.raises(ValidationError, match="non-finite"):
        validate_dataset(df)


def test_t02_reject_duplicate_rows():
    df = load_raw_dataset()
    df.iloc[1] = df.iloc[0].values
    with pytest.raises(ValidationError, match="duplicate"):
        validate_dataset(df)


def test_t02_reject_out_of_range_values():
    df = load_raw_dataset()
    df.at[0, "frequency_hz"] = 99999.0
    with pytest.raises(ValidationError, match="Range error"):
        validate_dataset(df)


def test_t03_reject_altered_column_names():
    df = load_raw_dataset().rename(columns={ALL_COLUMNS[0]: "hz"})
    with pytest.raises(ValidationError, match="Schema error: expected columns"):
        validate_dataset(df)


def test_t03_reject_altered_column_order():
    df = load_raw_dataset()[list(reversed(ALL_COLUMNS))]
    with pytest.raises(ValidationError, match="Schema error: expected columns"):
        validate_dataset(df)


def test_t02_reject_invalid_row_count():
    df = load_raw_dataset().iloc[:100]
    with pytest.raises(ValidationError, match="expected"):
        validate_dataset(df)


def test_t02_reject_non_numeric_column():
    df = load_raw_dataset()
    df["frequency_hz"] = df["frequency_hz"].astype(str)
    with pytest.raises(ValidationError, match="non-numeric"):
        validate_dataset(df)
