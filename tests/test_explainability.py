"""Phase 9 explainability/error analysis tests.

Cover deterministic reproducibility (same split, same hyperparameters =>
identical importance and group summaries on repeated runs), five
experiment-B features ranked exactly once, residual = actual - predicted,
group coverage (all 22 holdout groups, 315 observations), and that no
holdout outputs are modified.
"""

import filecmp
import numpy as np
import pandas as pd
import pytest

from airfoil_noise.config import EXPERIMENT_B_FEATURES
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.explainability import (
    GROUP_ID_COLUMN,
    IMPORTANCE_CSV,
    IMPORTANCE_PNG,
    RESID_FREQ_PNG,
    RESID_GROUP_CSV,
    run_all,
)


@pytest.fixture(scope="module")
def first_run():
    return run_all(load_raw_dataset())


@pytest.fixture(scope="module")
def second_run():
    return run_all(load_raw_dataset())


def test_output_files_exist(first_run):
    for path in (IMPORTANCE_CSV, IMPORTANCE_PNG, RESID_FREQ_PNG, RESID_GROUP_CSV):
        assert path.exists()


def test_importance_has_exactly_five_features(first_run):
    importance = pd.read_csv(IMPORTANCE_CSV)
    assert list(importance["rank"]) == list(range(1, 6))
    assert sorted(importance["feature"]) == sorted(EXPERIMENT_B_FEATURES)
    assert np.isfinite(importance["importance_mean"].values).all()
    assert np.isfinite(importance["importance_std"].values).all()


def test_importance_is_ranked_descending(first_run):
    importance = pd.read_csv(IMPORTANCE_CSV)
    means = importance["importance_mean"].values
    assert np.all(np.diff(means) <= 0)


def test_deterministic_on_repeat(first_run, second_run):
    imp1 = pd.read_csv(first_run["importance_csv"])
    imp2 = pd.read_csv(second_run["importance_csv"])
    pd.testing.assert_frame_equal(imp1, imp2)

    grp1 = pd.read_csv(first_run["residual_group_csv"])
    grp2 = pd.read_csv(second_run["residual_group_csv"])
    pd.testing.assert_frame_equal(grp1, grp2)


def test_holdout_predictions_not_modified(first_run):
    df = load_raw_dataset()
    from airfoil_noise.groups import make_group_ids, split_train_holdout_groups
    from airfoil_noise.holdout import PREDICTIONS_CSV

    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    holdout_mask = ids.isin(holdout_groups)
    holdout_df = df.loc[holdout_mask].reset_index(drop=True)
    holdout_ids = ids.loc[holdout_mask].reset_index(drop=True)

    from airfoil_noise.features import experiment_b_frame
    from airfoil_noise.holdout import build_selected_pipeline
    from airfoil_noise.config import TARGET_COLUMN

    pipe = build_selected_pipeline()
    X_tr = experiment_b_frame(df.loc[~holdout_mask].reset_index(drop=True))[
        EXPERIMENT_B_FEATURES
    ].reset_index(drop=True)
    y_tr = df.loc[~holdout_mask, TARGET_COLUMN].reset_index(drop=True)
    X_ho = experiment_b_frame(holdout_df)[EXPERIMENT_B_FEATURES].reset_index(drop=True)
    y_ho = holdout_df[TARGET_COLUMN].reset_index(drop=True)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_ho)

    regenerated = pd.DataFrame(
        {
            GROUP_ID_COLUMN: holdout_ids.astype(int),
            "actual": y_ho.values.astype(float),
            "predicted": pred.astype(float),
        }
    )
    regenerated["residual"] = regenerated["actual"] - regenerated["predicted"]
    existing = pd.read_csv(PREDICTIONS_CSV)
    pd.testing.assert_frame_equal(existing.sort_index().reset_index(drop=True), regenerated.sort_index().reset_index(drop=True))


def test_group_summary_coverage(first_run):
    summary = pd.read_csv(RESID_GROUP_CSV)
    assert len(summary) == 22
    assert summary["n_observations"].sum() == 315
    assert np.isfinite(summary["mean_residual"].values).all()
    assert (summary["mean_absolute_error"].values >= 0).all()
    assert (summary["rmse"].values >= 0).all()
    pred_line = sum(
        row["n_observations"] * abs(row["mean_residual"]) for _, row in summary.iterrows()
    )
    assert pred_line < 315 * 30  # sanity: residuals bounded, not garbage
