"""Dataset ingestion: load the raw .dat file and build the processed CSV copy."""

from pathlib import Path

import pandas as pd

from .config import ALL_COLUMNS, PROCESSED_DIR, PROCESSED_FILENAME, RAW_DIR, RAW_FILENAME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / RAW_DIR / RAW_FILENAME
PROCESSED_PATH = PROJECT_ROOT / PROCESSED_DIR / PROCESSED_FILENAME


def load_raw_dataset(path: str = str(RAW_PATH)) -> pd.DataFrame:
    """Load the canonical raw .dat file: whitespace-separated, no header,
    six columns mapped to ALL_COLUMNS in the documented order."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS, engine="python")
    if list(df.columns) != ALL_COLUMNS:
        raise ValueError(f"Loader mapped columns {list(df.columns)}; expected {ALL_COLUMNS}")
    return df


def save_processed_dataset(df: pd.DataFrame, output: str = str(PROCESSED_PATH)) -> Path:
    """Persist the frame as an index-free CSV; return the output path."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def ingest(path: str = str(RAW_PATH), output: str = str(PROCESSED_PATH)) -> pd.DataFrame:
    """Load the raw file, persist the processed CSV, return the frame."""
    df = load_raw_dataset(path)
    save_processed_dataset(df, output)
    return df


def main() -> None:
    df = ingest()
    print(f"Ingested {len(df)} rows x {df.shape[1]} columns -> {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
