"""T-01: Load the raw .dat file. Returns 1,503 rows and six canonical columns."""

from airfoil_noise.config import ALL_COLUMNS, EXPECTED_ROW_COUNT
from airfoil_noise.data import load_raw_dataset


def test_t01_load_raw_returns_canonical_columns_and_row_count():
    df = load_raw_dataset()
    assert list(df.columns) == ALL_COLUMNS
    assert df.shape[1] == 6
    assert len(df) == EXPECTED_ROW_COUNT
