"""Controlled feature-engineering experiments (Phase 5).

Two exactly-five-feature representations of the same validated dataset:

  Experiment A - the five original input columns.
  Experiment B - ``frequency_hz`` replaced by ``log10_frequency_hz``; the
  remaining four inputs unchanged.

The transformation is deterministic: pure NumPy on a copied frame, no random
state, no observation removal, no mutation of the source data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXPERIMENT_A_FEATURES, EXPERIMENT_B_FEATURES, LOG10_FREQUENCY_COLUMN


class FeatureError(ValueError):
    """Raised when a feature transformation cannot be applied safely."""


def _require_numeric(series: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(series):
        raise FeatureError(f"'{name}' must be numeric")


def add_log10_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``log10_frequency_hz`` column.

    ``frequency_hz`` must be present, numeric, finite and strictly positive
    (log10 is undefined at zero and below). The input frame is never mutated.
    """
    if "frequency_hz" not in df.columns:
        raise FeatureError("Input must contain 'frequency_hz'")

    freq = df["frequency_hz"]
    _require_numeric(freq, "frequency_hz")
    if freq.isna().any():
        raise FeatureError("'frequency_hz' contains missing values")
    if not np.isfinite(freq.to_numpy(dtype=float)).all():
        raise FeatureError("'frequency_hz' contains non-finite values")
    if (freq <= 0).any():
        raise FeatureError("'frequency_hz' contains non-positive values; log10 is undefined")

    out = df.copy()
    out[LOG10_FREQUENCY_COLUMN] = np.log10(freq.to_numpy(dtype=float))
    return out


def experiment_a_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Experiment A representation: the five original input features in the
    configured order. Five columns total. The input frame is never mutated."""
    missing = [c for c in EXPERIMENT_A_FEATURES if c not in df.columns]
    if missing:
        raise FeatureError(f"Missing required input columns: {missing}")
    return df[EXPERIMENT_A_FEATURES].copy()


def experiment_b_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Experiment B representation: ``log10_frequency_hz`` plus the four
    unchanged input features in the configured order. Five columns total."""
    augmented = add_log10_frequency(df)
    return augmented[EXPERIMENT_B_FEATURES].copy()


def compare_correlations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation matrices for Experiment A and Experiment B."""
    return experiment_a_frame(df).corr(), experiment_b_frame(df).corr()
