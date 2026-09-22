"""Leave-last-out splitting and the user-item matrix.

Two decisions here are the difference between a believable evaluation and a
meaningless one.

**The held-out item is the user's most recent play, not a random one.** A
random hold-out lets the model train on what the user did *after* the item it
is being asked to predict. Recommenders are deployed forwards in time, so the
evaluation has to run forwards too.

**Items the user already played are removed from the candidate list.** Scoring
them is the single most common way to inflate a recommender: the model's job is
to surface something new, and a system that recommends what someone has already
listened to twenty times will look excellent offline and get switched off in
production. `recommend` takes the training matrix precisely so it can mask
them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class Dataset:
    """A fitted train/test split, in the shapes the models want."""

    train_matrix: sparse.csr_matrix  # users x items, confidence-weighted counts
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    n_users: int
    n_items: int

    @property
    def test_users(self) -> np.ndarray:
        return np.sort(self.test_df["user_id"].unique())

    def ground_truth(self) -> dict[int, set]:
        """Held-out items per user, in the form the metrics consume."""
        grouped = self.test_df.groupby("user_id")["item_id"].apply(set)
        return {int(user): items for user, items in grouped.items()}


def leave_last_out(
    interactions: pd.DataFrame,
    n_items_catalogue: int,
    n_holdout: int = 2,
    min_train_interactions: int = 5,
) -> Dataset:
    """Hold out each user's `n_holdout` most recent plays.

    Users with too little history to leave anything behind are kept in training
    (they still inform the item factors) but excluded from the test set, rather
    than dropped. Dropping them quietly removes the cold users a recommender is
    worst at, and flatters every metric.
    """
    if n_holdout < 1:
        raise ValueError("n_holdout must be at least 1")

    df = interactions.sort_values(["user_id", "timestamp"], kind="mergesort")
    position_from_end = df.groupby("user_id", sort=False).cumcount(ascending=False)
    n_per_user = df.groupby("user_id", sort=False)["item_id"].transform("size")

    eligible = n_per_user - n_holdout >= min_train_interactions
    is_test = eligible & (position_from_end < n_holdout)

    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df = df.loc[is_test].reset_index(drop=True)

    if test_df.empty:
        raise ValueError(
            "no user had enough history to hold anything out; lower n_holdout "
            "or min_train_interactions"
        )

    n_users = int(interactions["user_id"].max()) + 1
    matrix = build_matrix(train_df, n_users, n_items_catalogue)

    return Dataset(
        train_matrix=matrix,
        train_df=train_df,
        test_df=test_df,
        n_users=n_users,
        n_items=n_items_catalogue,
    )


def build_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    """User-item sparse matrix of play counts.

    Counts, not binary flags: playing a track twenty times is a stronger signal
    than playing it once, and the ALS confidence weighting is built to use that.
    """
    matrix = sparse.csr_matrix(
        (
            df["play_count"].to_numpy(dtype=np.float32),
            (df["user_id"].to_numpy(), df["item_id"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
    matrix.sum_duplicates()
    return matrix


def assert_no_test_item_in_train(dataset: Dataset) -> None:
    """Verify that no (user, item) pair appears on both sides.

    A user can legitimately replay a track, so the same pair could appear in
    both windows -- and if it does, the model is being asked to predict
    something it was shown. This raises rather than warns, because the failure
    is silent and the metric it produces is wrong in the flattering direction.
    """
    train_pairs = set(zip(dataset.train_df["user_id"], dataset.train_df["item_id"]))
    test_pairs = set(zip(dataset.test_df["user_id"], dataset.test_df["item_id"]))
    overlap = train_pairs & test_pairs
    if overlap:
        raise AssertionError(
            f"{len(overlap):,} (user, item) pairs appear in both train and test; "
            "the held-out items are not actually unseen"
        )


def drop_repeat_consumption(interactions: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated plays of one item into a single row, keeping the last.

    Whether to do this is a product question, not a technical one: a music
    service wants to recommend tracks you will replay, a retailer does not want
    to recommend the washing machine you just bought. Collapsing first makes
    the leave-last-out split a genuine next-*new*-item prediction.
    """
    collapsed = (
        interactions.sort_values(["user_id", "timestamp"], kind="mergesort")
        .groupby(["user_id", "item_id"], as_index=False)
        .agg(timestamp=("timestamp", "max"), play_count=("play_count", "sum"))
    )
    return collapsed.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
