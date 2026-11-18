"""Phase 12 — single-page Streamlit frontend over the approved Phase 11 backend.

Connects exclusively to ``src/airfoil_noise/predict.py`` (load_model /
validate_prediction_input / predict_sound_pressure). Never loads joblib
directly; never duplicates input bounds (read from config.FEATURE_BOUNDS).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# requirements.txt installs third-party dependencies only, so the local
# src-layout package is not importable on Streamlit Cloud. Add the repo's src
# directory to sys.path relative to this file (works on Linux and Windows).
# ponytail: src path bootstrap; ceiling: entrypoint must stay next to src/.
# Upgrade path: install the project via requirements.txt (-e .) and remove this.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from airfoil_noise.config import FEATURE_BOUNDS, FEATURE_COLUMNS
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.predict import (
    ARTIFACT_PATH,
    METADATA_PATH,
    load_model,
    predict_sound_pressure,
)

PAGE_TITLE = "Airfoil Noise Predictor"
PAGE_ICON = "⚡"  # high-voltage / engineering-flavoured icon
HEADING = "Airfoil Noise Predictor"
DESCRIPTION = "Estimate scaled sound pressure level from five airfoil test inputs."

TRAINING_HINT = (
    "Model artifacts are missing or incompatible. "
    "Run the training workflow to generate them: "
    "`python src/airfoil_noise/artifact.py`"
)

# Documented first-row example (a valid row from the supplied dataset). These
# are default starting values only — not a recommended or preferred
# aerodynamic configuration.
FIRST_ROW_DEFAULTS = {
    "frequency_hz": 800.0,
    "angle_of_attack_deg": 0.0,
    "chord_length_m": 0.3048,
    "free_stream_velocity_m_s": 71.3,
    "suction_side_displacement_thickness_m": 0.00266337,
}

FIELD_META = [
    {
        "label": "Frequency",
        "key": "frequency_hz",
        "unit": "Hz",
    },
    {
        "label": "Angle of attack",
        "key": "angle_of_attack_deg",
        "unit": "degrees",
    },
    {
        "label": "Chord length",
        "key": "chord_length_m",
        "unit": "m",
    },
    {
        "label": "Free-stream velocity",
        "key": "free_stream_velocity_m_s",
        "unit": "m/s",
    },
    {
        "label": "Suction-side displacement thickness",
        "key": "suction_side_displacement_thickness_m",
        "unit": "m",
    },
]


@st.cache_resource(show_spinner="Loading prediction model...")
def _load_cached_model():
    """Load the trusted model through the existing backend, cached per session.

    Returns (model, error_message). On missing/incompatible artifacts the
    model is None and the message instructs the maintainer.
    """
    if not ARTIFACT_PATH.exists() or not METADATA_PATH.exists():
        return None, TRAINING_HINT
    try:
        return load_model(), None
    except (FileNotFoundError, ValueError) as exc:  # incompatible artifact
        logging.getLogger("streamlit").exception("Model load failed")
        return None, TRAINING_HINT


def _dataset_first_row() -> dict:
    """Return the first real dataset row as floats, falling back to defaults."""
    try:
        df = load_raw_dataset()
        row = df[FEATURE_COLUMNS].iloc[0]
        return {col: float(row[col]) for col in FEATURE_COLUMNS}
    except Exception:  # dataset unavailable; use documented example
        return dict(FIRST_ROW_DEFAULTS)


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="centered",
    )
    st.title(HEADING)
    st.caption(DESCRIPTION)

    defaults = _dataset_first_row()

    values: dict = {}
    for field in FIELD_META:
        key = field["key"]
        low, high = FEATURE_BOUNDS[key]
        values[key] = st.number_input(
            label=f"{field['label']} ({field['unit']})",
            key=key,
            min_value=float(low),
            max_value=float(high),
            value=float(defaults[key]),
            step=_step_for(key),
            help=f"Allowed range: {low} to {high} {field['unit']}",
        )

    predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        inputs = {col: values[col] for col in FEATURE_COLUMNS}
        model, model_error = _load_cached_model()
        if model_error is not None:
            st.error(model_error)
            return
        try:
            prediction = predict_sound_pressure(inputs)
        except ValueError as exc:  # backend input validation failure
            st.error(f"Invalid input: {exc}")
            return
        except Exception:  # unexpected inference failure
            logging.getLogger("streamlit").exception("Prediction failed")
            st.error("Prediction could not be completed. Please try again.")
            return
        st.success("Prediction complete")
        st.markdown(
            f"**Predicted scaled sound pressure level:** {prediction:.2f} dB"
        )


def _step_for(key: str) -> float:
    """Sensible per-field increment; not part of the approved contract."""
    steps = {
        "frequency_hz": 10.0,
        "angle_of_attack_deg": 0.1,
        "chord_length_m": 0.001,
        "free_stream_velocity_m_s": 0.1,
        "suction_side_displacement_thickness_m": 0.000001,
    }
    return steps[key]


main()
