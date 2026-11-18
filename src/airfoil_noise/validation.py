"""Dataset validation: schema and quality checks with clear, actionable errors."""

import numpy as np
import pandas as pd

from .config import ALL_COLUMNS, EXPECTED_ROW_COUNT, FEATURE_BOUNDS

EXPECTED_COLUMNS = list(ALL_COLUMNS)


class ValidationError(ValueError):
    """Raised when the dataset fails a schema or quality check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_schema(df: pd.DataFrame) -> None:
    _require(
        list(df.columns) == EXPECTED_COLUMNS,
        f"Schema error: expected columns {EXPECTED_COLUMNS}, got {list(df.columns)}",
    )


def validate_row_count(df: pd.DataFrame) -> None:
    _require(
        len(df) == EXPECTED_ROW_COUNT,
        f"Schema error: expected {EXPECTED_ROW_COUNT} rows, got {len(df)}",
    )


def validate_dtypes(df: pd.DataFrame) -> None:
    non_numeric = [c for c in EXPECTED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    _require(not non_numeric, f"Dtype error: non-numeric columns {non_numeric}")


def validate_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    _require(
        missing.empty,
        f"Quality error: missing cells in {missing.to_dict()}",
    )


def validate_finite(df: pd.DataFrame) -> None:
    arr = df[EXPECTED_COLUMNS].to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not finite.all():
        rows, cols = np.where(~finite)
        bad = {EXPECTED_COLUMNS[c]: sorted({int(r) for r in rows[cols == i]}) for i, c in enumerate(cols)}
        raise ValidationError(f"Quality error: non-finite values at row indices {bad}")


def validate_duplicates(df: pd.DataFrame) -> None:
    n = int(df.duplicated().sum())
    _require(n == 0, f"Quality error: {n} exact duplicate row(s)")


def check_feature_bounds(df: pd.DataFrame) -> dict:
    """Return {column: [(row_index, value), ...]} for values outside FEATURE_BOUNDS."""
    violations = {}
    for col, (low, high) in FEATURE_BOUNDS.items():
        series = df[col]
        out = series.index[(series < low) | (series > high)]
        if len(out):
            violations[col] = [(int(i), float(series[i])) for i in out]
    return violations


def validate_dataset(df: pd.DataFrame, check_ranges: bool = True) -> pd.DataFrame:
    """Run every check and return the frame. Raises ValidationError on the first failure."""
    validate_schema(df)
    validate_row_count(df)
    validate_dtypes(df)
    validate_missing(df)
    validate_finite(df)
    validate_duplicates(df)
    if check_ranges:
        violations = check_feature_bounds(df)
        _require(not violations, f"Range error: out-of-bound values {violations}")
    return df
