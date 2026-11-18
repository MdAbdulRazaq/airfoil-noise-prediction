"""Single source of truth for dataset schema, target and empirical input bounds."""

FEATURE_COLUMNS = [
    "frequency_hz",
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_m_s",
    "suction_side_displacement_thickness_m",
]

TARGET_COLUMN = "scaled_sound_pressure_level_db"

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

GROUP_COLUMNS = [
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_m_s",
    "suction_side_displacement_thickness_m",
]

# Experiment A: the five original input features, in documented order.
EXPERIMENT_A_FEATURES = list(FEATURE_COLUMNS)

# Experiment B: frequency_hz replaced by its base-10 logarithm. Exactly five
# features, order preserved: log-frequency first, then the four unchanged inputs.
LOG10_FREQUENCY_COLUMN = "log10_frequency_hz"
EXPERIMENT_B_FEATURES = [
    LOG10_FREQUENCY_COLUMN,
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_m_s",
    "suction_side_displacement_thickness_m",
]

# Observed ranges from the uploaded dataset audit (data dictionary, Section 2.3).
# These are empirical safeguards only; they are not code-generated at import time
# so the contract stays fixed regardless of the file on disk.
# Range is fixed at the exact observed extrema — never extended beyond them.
FEATURE_BOUNDS = {
    "frequency_hz": (200.0, 20000.0),
    "angle_of_attack_deg": (0.0, 22.2),
    "chord_length_m": (0.0254, 0.3048),
    "free_stream_velocity_m_s": (31.7, 71.3),
    "suction_side_displacement_thickness_m": (0.000400682, 0.0584113),
}

EXPECTED_ROW_COUNT = 1503

DATA_DIR = "data"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RAW_FILENAME = "airfoil_self_noise.dat"
PROCESSED_FILENAME = "airfoil_self_noise.csv"
