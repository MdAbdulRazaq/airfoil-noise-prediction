"""Phase 8 — Final holdout evaluation (one shot, no re-tuning).

Evaluates the Phase 7 selected configuration exactly once on the frozen
holdout set:

  Selected configuration (lowest mean GroupKFold CV MAE = 1.8947):
    Variant : Experiment B (log10_frequency_hz + four unchanged inputs)
    Model   : SVR(kernel="rbf"), C=100.0, epsilon=0.2, gamma="scale"

The complete preprocessing + model pipeline is refit once on all 1,188
training observations and used to predict the 315 holdout observations
exactly once. The holdout is never used for tuning or model selection.

No algorithms, dependencies or features beyond the frozen Phase 6 / Phase 7
specification are introduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import EXPERIMENT_B_FEATURES, GROUP_COLUMNS, TARGET_COLUMN
from .features import experiment_b_frame
from .groups import GROUP_ID_COLUMN, make_group_ids, split_train_holdout_groups

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
PREDICTIONS_CSV = REPORTS_DIR / "holdout_predictions.csv"
EVALUATION_JSON = REPORTS_DIR / "holdout_evaluation.json"

# Selected configuration from Phase 7 (frozen — not re-derived here).
SELECTED_VARIANT = "Experiment B (log-frequency)"
SELECTED_MODEL = "SVR(kernel=\"rbf\")"
BEST_PARAMS = {"C": 100.0, "epsilon": 0.2, "gamma": "scale"}
MEAN_CV_MAE = 1.8947

RANDOM_STATE = 42


def build_selected_pipeline() -> Pipeline:
    """Construct the selected SVR pipeline with frozen Phase 7 hyperparameters.

    StandardScaler inside the pipeline (per spec for SVR). Fit happens only on
    training data, so scaling never sees holdout information.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf", C=BEST_PARAMS["C"], epsilon=BEST_PARAMS["epsilon"], gamma=BEST_PARAMS["gamma"])),
        ]
    )


def evaluate_holdout(df: pd.DataFrame) -> dict:
    """Run the single holdout evaluation.

    Returns a dict with metrics and writes reports/holdout_predictions.csv
    and reports/holdout_evaluation.json.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Frozen Phase 6 split — never modified, never re-derived.
    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)

    train_mask = ids.isin(train_groups)
    holdout_mask = ids.isin(holdout_groups)

    train_df = df.loc[train_mask].reset_index(drop=True)
    holdout_df = df.loc[holdout_mask].reset_index(drop=True)
    train_ids = ids.loc[train_mask].reset_index(drop=True)
    holdout_ids = ids.loc[holdout_mask].reset_index(drop=True)

    # Isolation invariants (verified, never rebuilt from holdout).
    assert not set(train_groups) & set(holdout_groups)
    assert set(train_groups) | set(holdout_groups) == set(ids.unique())
    assert len(train_df) == 1188
    assert len(holdout_df) == 315
    assert len(train_groups) == 84
    assert len(holdout_groups) == 22

    # Experiment B feature frame (identical construction as Phase 7).
    X_train = experiment_b_frame(train_df)[EXPERIMENT_B_FEATURES].reset_index(drop=True)
    y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
    X_holdout = experiment_b_frame(holdout_df)[EXPERIMENT_B_FEATURES].reset_index(drop=True)
    y_holdout = holdout_df[TARGET_COLUMN].reset_index(drop=True)

    # Fit once on all training data; scale learned from training only.
    pipe = build_selected_pipeline()
    pipe.fit(X_train, y_train)

    # Predict the holdout exactly once.
    y_pred = pipe.predict(X_holdout)

    # Metrics.
    mae = float(mean_absolute_error(y_holdout, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_holdout, y_pred)))
    r2 = float(r2_score(y_holdout, y_pred))

    # Predictions CSV — actual, predicted, residual, group id.
    predictions = pd.DataFrame(
        {
            GROUP_ID_COLUMN: holdout_ids.astype(int),
            "actual": y_holdout.values.astype(float),
            "predicted": y_pred.astype(float),
            "residual": (y_holdout.values.astype(float) - y_pred).astype(float),
        }
    )
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    # Evaluation JSON — metrics + model configuration.
    evaluation = {
        "phase": 8,
        "variant": SELECTED_VARIANT,
        "model": SELECTED_MODEL,
        "best_params": BEST_PARAMS,
        "selection_basis": "lowest mean GroupKFold cross-validation MAE",
        "mean_cv_mae_selection": MEAN_CV_MAE,
        "n_train_groups": int(len(train_groups)),
        "n_holdout_groups": int(len(holdout_groups)),
        "n_train_observations": int(len(train_df)),
        "n_holdout_observations": int(len(holdout_df)),
        "holdout_mae": mae,
        "holdout_rmse": rmse,
        "holdout_r2": r2,
    }
    with open(EVALUATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, indent=2)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_holdout": int(len(holdout_df)),
        "predictions_csv": str(PREDICTIONS_CSV),
        "evaluation_json": str(EVALUATION_JSON),
    }


def main() -> None:
    from .data import load_raw_dataset

    df = load_raw_dataset()
    result = evaluate_holdout(df)
    print("Holdout evaluation (single pass, no re-tuning):")
    print(f"  MAE = {result['mae']:.4f}")
    print(f"  RMSE = {result['rmse']:.4f}")
    print(f"  R^2 = {result['r2']:.4f}")
    print(f"  Holdout observations = {result['n_holdout']}")
    print("Saved:", result["predictions_csv"])
    print("Saved:", result["evaluation_json"])


if __name__ == "__main__":
    main()
