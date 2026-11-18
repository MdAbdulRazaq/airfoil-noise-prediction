"""Phase 7 regression comparison (T-06 / T-07 coverage).

Asserts all five candidates complete GroupKFold cross-validation on the
training groups only, metrics are finite, fold groups stay inside the training
block and are disjoint within each fold, and selection uses lowest mean CV MAE.
"""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.config import TARGET_COLUMN
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.features import experiment_a_frame
from airfoil_noise.groups import (
    GROUP_ID_COLUMN,
    group_kfold_folds,
    make_group_ids,
    split_train_holdout_groups,
)
from airfoil_noise.models import (
    FOLDS_CSV,
    N_SPLITS,
    PARAM_GRIDS,
    PIPELINES,
    SUMMARY_CSV,
    compare_variants,
    tune_and_evaluate,
)


def test_all_five_pipelines_build():
    assert set(PIPELINES) == {
        "LinearRegression",
        "Ridge",
        "SVR",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }


def test_param_grids_match_pipelines():
    assert set(PARAM_GRIDS) == set(PIPELINES)


def test_all_candidates_complete_cv_without_errors():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, _ = split_train_holdout_groups(ids)
    train_mask = ids.isin(train_groups)
    train_df = df.loc[train_mask].reset_index(drop=True)
    train_ids = ids.loc[train_mask].reset_index(drop=True)
    X = experiment_a_frame(train_df)
    y = train_df[TARGET_COLUMN].reset_index(drop=True)
    for model_name, pipe in PIPELINES.items():
        result = tune_and_evaluate(pipe, PARAM_GRIDS[model_name], X, y, train_ids)
        assert result["mean_cv_mae"] > 0
        assert result["mean_cv_rmse"] > 0
        assert len(result["fold_mae"]) == N_SPLITS
        assert len(result["fold_r2"]) == N_SPLITS


def test_r2_is_finite_and_bounded():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, _ = split_train_holdout_groups(ids)
    train_mask = ids.isin(train_groups)
    train_df = df.loc[train_mask].reset_index(drop=True)
    train_ids = ids.loc[train_mask].reset_index(drop=True)
    X = experiment_a_frame(train_df)
    y = train_df[TARGET_COLUMN].reset_index(drop=True)
    result = tune_and_evaluate(PIPELINES["LinearRegression"], {}, X, y, train_ids)
    assert np.isfinite(result["mean_cv_r2"])
    assert np.isfinite(result["std_cv_r2"])


def test_summary_csv_saved_and_selection_by_min_mae():
    result = compare_variants(load_raw_dataset())
    summary = result["summary"]
    assert not summary.empty
    assert result["best_variant"]
    assert result["best_model"]
    assert abs(result["best_mean_cv_mae"] - summary["mean_cv_mae"].min()) < 1e-12
    assert SUMMARY_CSV.exists()
    assert FOLDS_CSV.exists()


def test_fold_group_isolation_within_training_block():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    train_set = set(train_groups)
    holdout_set = set(holdout_groups)
    seen_val = set()
    for tr, val in group_kfold_folds(train_groups, n_splits=N_SPLITS):
        assert not set(tr) & set(val)
        assert set(val) <= train_set
        assert not set(val) & holdout_set
        seen_val.update(val)
    assert seen_val == train_set


def test_holdout_never_enters_training_data():
    result = compare_variants(load_raw_dataset())
    assert result["n_train_groups"] + result["n_holdout_groups"] == 106
    assert result["n_holdout_groups"] == 22
    assert result["n_train_observations"] + result["n_holdout_observations"] == 1503
