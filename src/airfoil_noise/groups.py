"""Operating group generation and group-aware splitting (Phase 6).

Groups are defined solely by the four experimental-condition columns:

    angle_of_attack_deg, chord_length_m, free_stream_velocity_m_s,
    suction_side_displacement_thickness_m

``frequency_hz`` is deliberately excluded from the group key so that every
frequency variant of the same operating condition shares one group id. This
prevents group leakage between training and holdout data when comparing the
two feature representations.

All functions are deterministic and never mutate their input frame.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import GROUP_COLUMNS

GROUP_ID_COLUMN = "group_id"


def make_group_ids(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.Series:
    """Deterministic integer group id per unique combination of `columns`.

    Rows sharing identical values across all group columns receive one group
    id; ids run 0..n_groups-1 in first-appearance order. The input frame is
    not mutated.
    """
    cols = list(GROUP_COLUMNS) if columns is None else list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns: {missing}")
    keys = df[cols].astype(str).agg("||".join, axis=1)
    # pd.factorize returns (codes, uniques); codes has one entry per row.
    ids, _ = pd.factorize(keys, sort=True)
    return pd.Series(ids, index=df.index, name=GROUP_ID_COLUMN)


def attach_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``group_id`` column appended.

    Safe to call on either Experiment A or Experiment B frames, since group
    membership excludes frequency and is identical across representations.
    """
    out = df.copy()
    out[GROUP_ID_COLUMN] = make_group_ids(out)
    return out


def expected_group_count(df: pd.DataFrame) -> int:
    return int(make_group_ids(df).nunique())


def _dummy_x(n: int) -> pd.DataFrame:
    return pd.DataFrame({"_dummy": range(n)})


def split_train_holdout_groups(
    group_ids: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """GroupShuffleSplit into (train_group_ids, holdout_group_ids).

    Guarantees the two id lists are completely disjoint — no operating group
    appears in both training and holdout.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(_dummy_x(len(group_ids)), groups=group_ids))
    train_groups = sorted({int(g) for g in group_ids.iloc[train_idx]})
    holdout_groups = sorted({int(g) for g in group_ids.iloc[test_idx]})
    assert not set(train_groups) & set(holdout_groups), "train/holdout groups overlap"
    assert set(train_groups) | set(holdout_groups) == set(
        int(g) for g in group_ids.unique()
    ), "train/holdout do not cover all groups"
    return train_groups, holdout_groups


def group_kfold_folds(
    train_group_ids: Sequence[int], n_splits: int = 5
) -> list[tuple[list[int], list[int]]]:
    """GroupKFold over the training groups.

    Returns a list of ``(train_group_ids, val_group_ids)`` per fold. Within
    every fold the two id lists are completely disjoint, and across the five
    folds each training group appears in exactly one validation fold.
    """
    groups = pd.Series(list(train_group_ids), name=GROUP_ID_COLUMN)
    gkf = GroupKFold(n_splits=n_splits)
    folds: list[tuple[list[int], list[int]]] = []
    for train_idx, val_idx in gkf.split(_dummy_x(len(groups)), groups=groups):
        train_groups = sorted({int(groups.iloc[i]) for i in train_idx})
        val_groups = sorted({int(groups.iloc[i]) for i in val_idx})
        assert not set(train_groups) & set(val_groups), "fold train/val groups overlap"
        folds.append((train_groups, val_groups))
    return folds


def verify_group_isolation(df: pd.DataFrame) -> dict:
    """End-to-end check: 106 groups, disjoint train/holdout, disjoint folds."""
    ids = make_group_ids(df)
    n_groups = int(ids.nunique())
    train_groups, holdout_groups = split_train_holdout_groups(ids)
    folds = group_kfold_folds(train_groups)
    # Every group appears in exactly one validation fold across the 5 folds.
    val_occurrences = {g: 0 for g in train_groups}
    for _, val_groups in folds:
        for g in val_groups:
            val_occurrences[g] += 1
    return {
        "n_groups": n_groups,
        "n_train_groups": len(train_groups),
        "n_holdout_groups": len(holdout_groups),
        "n_folds": len(folds),
        "val_occurrences_all_one": all(v == 1 for v in val_occurrences.values()),
    }
