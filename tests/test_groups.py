"""Operating groups and group-aware splitting (Phase 6).

T-04: Create operating groups — 106 groups for the baseline file.
T-05: Check split isolation — no group ID occurs in both train and test.
"""

import pandas as pd
import pytest

from airfoil_noise.config import EXPECTED_ROW_COUNT, GROUP_COLUMNS
from airfoil_noise.data import load_raw_dataset
from airfoil_noise.groups import (
    GROUP_ID_COLUMN,
    attach_group_ids,
    expected_group_count,
    group_kfold_folds,
    make_group_ids,
    split_train_holdout_groups,
    verify_group_isolation,
)

GROUP_COLS = list(GROUP_COLUMNS)


def test_t04_baseline_generates_exactly_106_groups():
    df = load_raw_dataset()
    assert expected_group_count(df) == 106


def test_t04_group_ids_are_deterministic():
    df = load_raw_dataset()
    assert make_group_ids(df).equals(make_group_ids(df))


def test_t04_group_id_covers_all_rows():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    assert len(ids) == EXPECTED_ROW_COUNT
    assert not ids.isna().any()


def test_t04_frequency_is_not_part_of_group_key():
    base = dict(zip(GROUP_COLS, [5.0, 0.1, 40.0, 0.005]))
    row1 = {**base, "frequency_hz": 1000.0}
    row2 = {**base, "frequency_hz": 8000.0}
    row3 = {**dict(zip(GROUP_COLS, [10.0, 0.1, 40.0, 0.005])), "frequency_hz": 1000.0}
    df = pd.DataFrame([row1, row2, row3])
    ids = make_group_ids(df)
    # Same four group columns, different frequency -> one shared group id.
    assert ids.iloc[0] == ids.iloc[1]
    # Different on one group column -> different group id.
    assert ids.iloc[0] != ids.iloc[2]


def test_t05_train_holdout_groups_are_disjoint():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    assert train_groups and holdout_groups
    assert not set(train_groups) & set(holdout_groups)
    assert set(train_groups) | set(holdout_groups) == set(range(106))
    # test_size=0.20 applies to rows, so holdout groups are ~20% of rows,
    # not exactly 20% of groups. Assert a sensible band around it.
    holdout_frac = len(holdout_groups) / 106
    assert 0.10 < holdout_frac < 0.40


def test_t05_kfold_folds_have_disjoint_groups():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, _ = split_train_holdout_groups(ids)
    folds = group_kfold_folds(train_groups, n_splits=5)
    assert len(folds) == 5
    # Each training group appears in exactly one validation fold.
    val_occurrences = {g: 0 for g in train_groups}
    for _, val_groups in folds:
        for g in val_groups:
            val_occurrences[g] += 1
    assert all(v == 1 for v in val_occurrences.values())


def test_t05_kfold_each_fold_train_val_disjoint():
    df = load_raw_dataset()
    ids = make_group_ids(df)
    train_groups, _ = split_train_holdout_groups(ids)
    folds = group_kfold_folds(train_groups, n_splits=5)
    for train_groups_fold, val_groups in folds:
        assert not set(train_groups_fold) & set(val_groups)


def test_attach_group_ids_both_representations_same_groups():
    df = load_raw_dataset()
    exp_a = attach_group_ids(df[GROUP_COLS + ["frequency_hz"]])
    exp_b = attach_group_ids(df[GROUP_COLS])
    # Same row order, same group ids (groups exclude frequency).
    pd.testing.assert_series_equal(exp_a[GROUP_ID_COLUMN], exp_b[GROUP_ID_COLUMN])


def test_verify_group_isolation_end_to_end():
    df = load_raw_dataset()
    report = verify_group_isolation(df)
    assert report["n_groups"] == 106
    assert report["n_train_groups"] + report["n_holdout_groups"] == 106
    assert report["n_train_groups"] > 0 and report["n_holdout_groups"] > 0
    assert report["n_folds"] == 5
    assert report["val_occurrences_all_one"] is True
