"""Phase 9 — Explainability and error analysis (Section 9.7 / 10.10).

Deterministic, read-only analysis of the Phase 7 selected configuration
(Experiment B + SVR(RBF)) on the frozen holdout set:

  1. Permutation feature importance via sklearn.inspection.permutation_importance
     on the held-out data, scoring="neg_mean_absolute_error",
     n_repeats=30, random_state=42. Ranked results saved to CSV and a
     horizontal bar plot.
  2. Residual behaviour by frequency (scatter plot).
  3. Residual behaviour by operating configuration (per holdout group
     aggregates saved to CSV).

The selected pipeline is deterministically refit only on the frozen training
data with the frozen Phase 7 hyperparameters — no retuning, no alternative
models, no modification of the frozen holdout predictions or metrics.
Feature importance is reported as model behaviour, never as physical
causality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

from .config import EXPERIMENT_B_FEATURES, GROUP_COLUMNS, TARGET_COLUMN
from .features import experiment_b_frame
from .groups import GROUP_ID_COLUMN, make_group_ids, split_train_holdout_groups
from .holdout import build_selected_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
IMPORTANCE_CSV = REPORTS_DIR / "permutation_feature_importance.csv"
IMPORTANCE_PNG = FIG_DIR / "permutation_feature_importance.png"
RESID_FREQ_PNG = FIG_DIR / "residuals_by_frequency.png"
RESID_GROUP_CSV = REPORTS_DIR / "residuals_by_operating_group.csv"

SELECTED_VARIANT = "Experiment B (log-frequency)"
SELECTED_MODEL = "SVR(kernel=\"rbf\")"

N_REPEATS = 30
RANDOM_STATE = 42


def frozen_split(df: pd.DataFrame):
    """Return (train_df, holdout_df, train_ids, holdout_ids, ids) using the
    exact frozen Phase 6 split. Never modifies the input frame."""
    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    train_mask = ids.isin(train_groups)
    holdout_mask = ids.isin(holdout_groups)
    return (
        df.loc[train_mask].reset_index(drop=True),
        df.loc[holdout_mask].reset_index(drop=True),
        ids.loc[train_mask].reset_index(drop=True),
        ids.loc[holdout_mask].reset_index(drop=True),
        ids,
    )


def fit_selected_pipeline_on_train(train_df: pd.DataFrame) -> Pipeline:
    """Refit the selected pipeline on all training observations only.

    Deterministic given the frozen training data and frozen hyperparameters;
    no tuning, no estimator changes.
    """
    X = experiment_b_frame(train_df)[EXPERIMENT_B_FEATURES].reset_index(drop=True)
    y = train_df[TARGET_COLUMN].reset_index(drop=True)
    pipe = build_selected_pipeline()
    pipe.fit(X, y)
    return pipe


def compute_permutation_importance(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Ranked permutation importance on the holdout data.

    Positive mean importance: permuting the feature raises MAE (feature hurts
    performance when removed). Negative mean importance: permutation happened to
    lower MAE by chance — small magnitudes indicate negligible importance, not
    anti-importance. Model behaviour only, never physical causality.
    """
    result = permutation_importance(
        pipe,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    importance = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": [float(v) for v in result.importances_mean],
            "importance_std": [float(v) for v in result.importances_std],
        }
    )
    importance = importance.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    importance = importance[["rank", "feature", "importance_mean", "importance_std"]]
    return importance


def save_importance_plot(importance: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = importance.iloc[::-1]  # highest on top for horizontal bars
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=100)
    ax.barh(
        order["feature"],
        order["importance_mean"],
        xerr=order["importance_std"],
        color="steelblue",
        edgecolor="black",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation importance on holdout MAE (dB)")
    ax.set_ylabel("Feature")
    ax.set_title(
        "Permutation feature importance — Experiment B + SVR (RBF)\n"
        "Holdout set, scoring=neg_mean_absolute_error, 30 repeats, random_state=42"
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PNG, bbox_inches="tight")
    plt.close(fig)


def save_residuals_by_frequency_plot(holdout_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Scatter of residual (actual - predicted) vs original frequency_hz."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = holdout_df.copy().reset_index(drop=True)
    data["actual"] = predictions["actual"].to_numpy(dtype=float)
    data["predicted"] = predictions["predicted"].to_numpy(dtype=float)
    data["residual"] = data["actual"] - data["predicted"]

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=100)
    ax.scatter(
        data["frequency_hz"],
        data["residual"],
        s=24,
        alpha=0.7,
        edgecolor="none",
        color="darkgreen",
        label="Holdout residuals",
    )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual = actual - predicted (dB)")
    ax.set_title(
        "Holdout residuals by frequency — Experiment B + SVR (RBF)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESID_FREQ_PNG, bbox_inches="tight")
    plt.close(fig)


def save_residuals_by_operating_group(
    holdout_df: pd.DataFrame, predictions: pd.DataFrame, holdout_ids: pd.Series
) -> pd.DataFrame:
    """Per holdout operating group: obs count, mean residual, MAE, RMSE,
    max abs error, plus exact input/frequency measurements (no approximate
    matching — joined by deterministic position after the frozen split)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = holdout_df.copy().reset_index(drop=True)
    data[GROUP_ID_COLUMN] = holdout_ids.reset_index(drop=True)
    data["actual"] = predictions["actual"].to_numpy(dtype=float)
    data["predicted"] = predictions["predicted"].to_numpy(dtype=float)
    data["residual"] = data["actual"] - data["predicted"]
    data["abs_error"] = data["residual"].abs()

    # Exact group constant columns (frequency excluded from the group key, so
    # report its spread within the group plus the four grouping inputs).
    const_cols = GROUP_COLUMNS
    summary = (
        data.groupby(GROUP_ID_COLUMN, sort=True)
        .agg(
            n_observations=("residual", "size"),
            mean_residual=("residual", "mean"),
            mean_absolute_error=("abs_error", "mean"),
            rmse=("residual", lambda s: float(np.sqrt((s ** 2).mean()))),
            max_absolute_error=("abs_error", "max"),
            frequency_hz_min=("frequency_hz", "min"),
            frequency_hz_max=("frequency_hz", "max"),
            frequency_hz_mean=("frequency_hz", "mean"),
        )
    )
    group_const = data.groupby(GROUP_ID_COLUMN, sort=True)[const_cols].first()
    summary = summary.join(group_const)
    summary = summary.reset_index()
    col_order = [
        GROUP_ID_COLUMN,
        "n_observations",
        "frequency_hz_min",
        "frequency_hz_max",
        "frequency_hz_mean",
    ] + const_cols + [
        "mean_residual",
        "mean_absolute_error",
        "rmse",
        "max_absolute_error",
    ]
    summary = summary[col_order]
    summary.to_csv(RESID_GROUP_CSV, index=False)
    return summary


def run_all(df: pd.DataFrame) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, holdout_df, train_ids, holdout_ids, _ = frozen_split(df)
    pipe = fit_selected_pipeline_on_train(train_df)

    # Holdout predictions — reconstructed deterministically, compared against
    # the frozen file to confirm they are byte-aligned (never modified).
    X_holdout = experiment_b_frame(holdout_df)[EXPERIMENT_B_FEATURES].reset_index(drop=True)
    y_holdout = holdout_df[TARGET_COLUMN].reset_index(drop=True)
    y_pred = pipe.predict(X_holdout)
    holdout_predictions = pd.DataFrame(
        {
            GROUP_ID_COLUMN: holdout_ids.astype(int),
            "actual": y_holdout.values.astype(float),
            "predicted": y_pred.astype(float),
        }
    )
    holdout_predictions["residual"] = (
        holdout_predictions["actual"] - holdout_predictions["predicted"]
    )

    # Task 1: permutation importance on held-out data.
    importance = compute_permutation_importance(pipe, X_holdout, y_holdout)
    importance.to_csv(IMPORTANCE_CSV, index=False)
    save_importance_plot(importance)

    # Task 2: residuals by frequency.
    save_residuals_by_frequency_plot(holdout_df, holdout_predictions)

    # Task 3: residuals by operating configuration.
    group_summary = save_residuals_by_operating_group(
        holdout_df, holdout_predictions, holdout_ids
    )

    return {
        "importance": importance,
        "group_summary": group_summary,
        "n_holdout": len(holdout_df),
        "n_holdout_groups": holdout_predictions[GROUP_ID_COLUMN].nunique(),
        "importance_csv": str(IMPORTANCE_CSV),
        "importance_png": str(IMPORTANCE_PNG),
        "residual_freq_png": str(RESID_FREQ_PNG),
        "residual_group_csv": str(RESID_GROUP_CSV),
    }


def main() -> None:
    from .data import load_raw_dataset

    df = load_raw_dataset()
    result = run_all(df)
    pd.set_option("display.width", 200)
    print(result["importance"].to_string(index=False))
    print()
    print("Holdout groups:", result["n_holdout_groups"], "observations:", result["n_holdout"])
    print("Saved:", result["importance_csv"])
    print("Saved:", result["importance_png"])
    print("Saved:", result["residual_freq_png"])
    print("Saved:", result["residual_group_csv"])


if __name__ == "__main__":
    main()
