"""Split correctness and the exclusion of already-seen items.

Both failures these guard against inflate the metric rather than break the run,
which is why they are asserted rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import CatalogConfig, generate_interactions
from src.model import PopularityRecommender
from src.split import (
    assert_no_test_item_in_train,
    build_matrix,
    drop_repeat_consumption,
    leave_last_out,
)


@pytest.fixture(scope="module")
def interactions() -> pd.DataFrame:
    frame, _ = generate_interactions(
        CatalogConfig(n_users=300, n_items=400, seed=107)
    )
    return drop_repeat_consumption(frame)


def test_held_out_items_are_the_most_recent(interactions: pd.DataFrame) -> None:
    """Not a random hold-out: the model must not train on the user's future."""
    dataset = leave_last_out(interactions, 400, n_holdout=2)

    train_last = dataset.train_df.groupby("user_id")["timestamp"].max()
    test_first = dataset.test_df.groupby("user_id")["timestamp"].min()

    common = train_last.index.intersection(test_first.index)
    assert len(common) > 0
    assert (train_last.loc[common] <= test_first.loc[common]).all()


def test_no_pair_appears_on_both_sides(interactions: pd.DataFrame) -> None:
    dataset = leave_last_out(interactions, 400, n_holdout=2)
    assert_no_test_item_in_train(dataset)  # raises on failure


def test_every_test_user_holds_out_the_requested_number(
    interactions: pd.DataFrame,
) -> None:
    dataset = leave_last_out(interactions, 400, n_holdout=3)
    per_user = dataset.test_df.groupby("user_id").size()
    assert (per_user == 3).all()


def test_users_with_thin_history_stay_in_training(interactions: pd.DataFrame) -> None:
    """They inform the item factors even though they cannot be scored.

    Dropping them entirely would quietly remove the cold users a recommender is
    worst at, and flatter every metric in the table.
    """
    dataset = leave_last_out(interactions, 400, n_holdout=2, min_train_interactions=40)
    scored = set(dataset.test_df["user_id"])
    trained = set(dataset.train_df["user_id"])

    assert scored < trained, "no thin-history users were kept out of the test set"
    assert len(trained - scored) > 0


def test_recommendations_exclude_items_already_seen(interactions: pd.DataFrame) -> None:
    """The most common way to inflate a recommender's offline score."""
    dataset = leave_last_out(interactions, 400, n_holdout=2)
    model = PopularityRecommender().fit(dataset.train_matrix)

    users = dataset.test_users[:50]
    recommendations = model.recommend(users, k=10, train_matrix=dataset.train_matrix)

    train_sets = dataset.train_df.groupby("user_id")["item_id"].apply(set)
    for user, row in zip(users, recommendations):
        seen = train_sets.get(int(user), set())
        assert not (seen & set(row.tolist())), f"user {user} was recommended their own history"


def test_without_the_mask_a_trivial_model_looks_perfect(
    interactions: pd.DataFrame,
) -> None:
    """Demonstrates why the mask is not optional.

    Left unmasked, the popularity model happily returns items the user has
    already played -- and any model that re-ranks a user's own history would
    score near-perfectly while recommending nothing actionable.
    """
    dataset = leave_last_out(interactions, 400, n_holdout=2)
    model = PopularityRecommender().fit(dataset.train_matrix)

    users = dataset.test_users[:50]
    unmasked = model.recommend(users, k=10, train_matrix=None)

    train_sets = dataset.train_df.groupby("user_id")["item_id"].apply(set)
    overlap = sum(
        len(train_sets.get(int(u), set()) & set(row.tolist()))
        for u, row in zip(users, unmasked)
    )
    assert overlap > 0


def test_repeat_collapse_is_idempotent_and_preserves_volume(
    interactions: pd.DataFrame,
) -> None:
    once = drop_repeat_consumption(interactions)
    twice = drop_repeat_consumption(once)
    pd.testing.assert_frame_equal(once, twice)
    assert once["play_count"].sum() == interactions["play_count"].sum()
    assert not once.duplicated(subset=["user_id", "item_id"]).any()


def test_matrix_shape_and_counts(interactions: pd.DataFrame) -> None:
    matrix = build_matrix(interactions, n_users=300, n_items=400)
    assert matrix.shape == (300, 400)
    assert matrix.nnz == len(interactions)
    assert matrix.sum() == pytest.approx(interactions["play_count"].sum())


def test_invalid_holdout_is_rejected(interactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_holdout"):
        leave_last_out(interactions, 400, n_holdout=0)
    with pytest.raises(ValueError, match="enough history"):
        leave_last_out(interactions, 400, n_holdout=2, min_train_interactions=10_000)
