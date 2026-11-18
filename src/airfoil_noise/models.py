"""Phase 7 — train and compare five regressors across Experiment A and B.

Procedure (per the approved spec, Section 10.8):

  * Uses the frozen Phase 6 split: 84 training / 22 holdout operating groups.
  * The final holdout (315 observations) is never touched — no tuning, no
    evaluation against it.
  * Each candidate is tuned with GridSearchCV inside GroupKFold(5) on the
    training groups only; scalers live inside sklearn Pipelines so they are
    refit on every fold's training data and never leak from validation data.
  * Candidates are selected by lowest mean GroupKFold MAE. RMSE and R^2 are
    supporting metrics only.
  * Mean and standard deviation of MAE, RMSE and R^2 are recorded across the
    five folds, plus fold-level detail.

Five candidate estimators (exact, from the specification):

  LinearRegression        raw numeric inputs, no scaling
  Ridge                   StandardScaler inside the pipeline; tune alpha
  RandomForestRegressor   no scaling; random_state=42
  SVR(kernel="rbf")       StandardScaler inside the pipeline; tune C, epsilon, gamma
  GradientBoostingRegressor random_state=42; tune n_estimators, learning_rate, max_depth

Both Experiment A (five original features) and Experiment B (log10 frequency
for the first feature) are compared with identical group splits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import EXPERIMENT_A_FEATURES, EXPERIMENT_B_FEATURES, TARGET_COLUMN
from .features import experiment_a_frame, experiment_b_frame
from .groups import GROUP_ID_COLUMN, make_group_ids, split_train_holdout_groups

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
SUMMARY_CSV = REPORTS_DIR / "model_comparison.csv"
FOLDS_CSV = REPORTS_DIR / "model_comparison_folds.csv"

RANDOM_STATE = 42
N_SPLITS = 5
SELECTION_METRIC = "mean_cv_mae"  # lowest mean CV MAE wins

# One pipeline per candidate. Scaling only where the spec requires it
# (Ridge and SVR); Linear Regression gets raw inputs; tree models unscaled.
PIPELINES = {
    "LinearRegression": Pipeline([("regressor", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("regressor", Ridge())]),
    "RandomForestRegressor": Pipeline(
        [("regressor", RandomForestRegressor(random_state=RANDOM_STATE))]
    ),
    "SVR": Pipeline([("scaler", StandardScaler()), ("regressor", SVR(kernel="rbf"))]),
    "GradientBoostingRegressor": Pipeline(
        [("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE))]
    ),
}

# Modest, controlled hyperparameter grids (training folds only).
PARAM_GRIDS = {
    "LinearRegression": {},
    "Ridge": {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
    "RandomForestRegressor": {
        "regressor__n_estimators": [100, 300],
        "regressor__max_depth": [None, 10, 20],
    },
    "SVR": {
        "regressor__C": [1.0, 10.0, 100.0],
        "regressor__epsilon": [0.1, 0.2],
        "regressor__gamma": ["scale", "auto"],
    },
    "GradientBoostingRegressor": {
        "regressor__n_estimators": [100, 300],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5],
    },
}

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def tune_and_evaluate(pipe, param_grid, X, y, groups):
    """Tune with GridSearchCV(GroupKFold, MAE) then score the best pipeline
    across the same five folds. All preprocessing/tuning stays inside training
    folds via cross_validate cloning. Returns metrics and per-fold detail."""
    cv = GroupKFold(n_splits=N_SPLITS)
    search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=1,
    )
    search.fit(X, y, groups=groups)
    best = search.best_estimator_

    cv_results = cross_validate(
        clone(best),
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
    )
    mae = -cv_results["test_mae"]
    rmse = -cv_results["test_rmse"]
    r2 = cv_results["test_r2"]

    return {
        "best_params": search.best_params_,
        "mean_cv_mae": float(mae.mean()),
        "std_cv_mae": float(mae.std()),
        "mean_cv_rmse": float(rmse.mean()),
        "std_cv_rmse": float(rmse.std()),
        "mean_cv_r2": float(r2.mean()),
        "std_cv_r2": float(r2.std()),
        "fold_mae": [float(v) for v in mae],
        "fold_rmse": [float(v) for v in rmse],
        "fold_r2": [float(v) for v in r2],
    }


def compare_variants(df: pd.DataFrame) -> dict:
    """Run both feature variants through all five candidates on the frozen
    Phase 6 training groups. Save summary + fold-level CSVs to reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)

    # Leakage guard: holdout never enters tuning or scoring.
    assert not set(train_groups) & set(holdout_groups)
    assert set(train_groups) | set(holdout_groups) == set(ids.unique())

    train_mask = ids.isin(train_groups)
    train_df = df.loc[train_mask].reset_index(drop=True)
    train_groups_s = ids.loc[train_mask].reset_index(drop=True)

    variant_frames = {
        "Experiment A": experiment_a_frame(train_df)[EXPERIMENT_A_FEATURES],
        "Experiment B": experiment_b_frame(train_df)[EXPERIMENT_B_FEATURES],
    }

    summary_rows = []
    fold_rows = []
    for variant_name, X in variant_frames.items():
        y = train_df[TARGET_COLUMN].reset_index(drop=True)
        for model_name, pipe in PIPELINES.items():
            result = tune_and_evaluate(pipe, PARAM_GRIDS[model_name], X, y, train_groups_s)
            summary_rows.append(
                {
                    "variant": variant_name,
                    "model": model_name,
                    "mean_cv_mae": result["mean_cv_mae"],
                    "std_cv_mae": result["std_cv_mae"],
                    "mean_cv_rmse": result["mean_cv_rmse"],
                    "std_cv_rmse": result["std_cv_rmse"],
                    "mean_cv_r2": result["mean_cv_r2"],
                    "std_cv_r2": result["std_cv_r2"],
                    "best_params": str(result["best_params"]),
                }
            )
            for fold in range(N_SPLITS):
                fold_rows.append(
                    {
                        "variant": variant_name,
                        "model": model_name,
                        "fold": fold,
                        "mae": result["fold_mae"][fold],
                        "rmse": result["fold_rmse"][fold],
                        "r2": result["fold_r2"][fold],
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(SELECTION_METRIC).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    folds.to_csv(FOLDS_CSV, index=False)

    best = summary.iloc[0]
    return {
        "summary": summary,
        "folds": folds,
        "best_variant": best["variant"],
        "best_model": best["model"],
        "best_mean_cv_mae": best["mean_cv_mae"],
        "summary_csv": str(SUMMARY_CSV),
        "folds_csv": str(FOLDS_CSV),
        "n_train_groups": int(len(train_groups)),
        "n_holdout_groups": int(len(holdout_groups)),
        "n_train_observations": int(len(train_df)),
        "n_holdout_observations": int(len(df) - len(train_df)),
    }


def main() -> None:
    from .data import load_raw_dataset

    df = load_raw_dataset()
    result = compare_variants(df)
    pd.set_option("display.width", 200)
    print(result["summary"].to_string(index=False))
    print()
    print(
        "Best: variant={variant} model={model} mean CV MAE={mae:.4f}".format(
            variant=result["best_variant"],
            model=result["best_model"],
            mae=result["best_mean_cv_mae"],
        )
    )
    print("Saved:", result["summary_csv"])
    print("Saved:", result["folds_csv"])


if __name__ == "__main__":
    main()
