"""Ranking metrics checked against hand-computed values.

Recall and NDCG are easy to write down and easy to get subtly wrong -- an
off-by-one in the discount, a normaliser that ignores the hold-out size. Each
one here is checked on a case small enough to verify by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    catalogue_coverage,
    gini_at_k,
    hit_rate_at_k,
    map_at_k,
    mean_popularity_rank,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)

# Two users, five slots. User 0's held-out items are {1, 7}; user 1's is {3}.
RECOMMENDATIONS = np.array(
    [
        [1, 5, 7, 9, 2],  # hits at ranks 1 and 3
        [8, 4, 6, 3, 0],  # hit at rank 4
    ]
)
USERS = np.array([0, 1])
GROUND_TRUTH = {0: {1, 7}, 1: {3}}


def test_recall_counts_hits_over_held_out_items() -> None:
    # user 0 finds 2 of 2, user 1 finds 1 of 1
    assert recall_at_k(RECOMMENDATIONS, USERS, GROUND_TRUTH, k=5) == pytest.approx(1.0)
    # at k=2, user 0 finds 1 of 2 and user 1 finds none
    assert recall_at_k(RECOMMENDATIONS, USERS, GROUND_TRUTH, k=2) == pytest.approx(0.25)


def test_hit_rate_is_per_user_not_per_item() -> None:
    assert hit_rate_at_k(RECOMMENDATIONS, USERS, GROUND_TRUTH, k=5) == pytest.approx(1.0)
    assert hit_rate_at_k(RECOMMENDATIONS, USERS, GROUND_TRUTH, k=2) == pytest.approx(0.5)


def test_ndcg_matches_a_hand_computation() -> None:
    discounts = 1.0 / np.log2(np.arange(2, 7))
    user0 = (discounts[0] + discounts[2]) / (discounts[0] + discounts[1])
    user1 = discounts[3] / discounts[0]
    expected = (user0 + user1) / 2

    assert ndcg_at_k(RECOMMENDATIONS, USERS, GROUND_TRUTH, k=5) == pytest.approx(expected)


def test_ndcg_rewards_putting_the_hit_first() -> None:
    early = np.array([[3, 0, 0, 0, 0]])
    late = np.array([[0, 0, 0, 0, 3]])
    truth = {1: {3}}

    assert ndcg_at_k(early, np.array([1]), truth, k=5) > ndcg_at_k(
        late, np.array([1]), truth, k=5
    )


def test_map_matches_a_hand_computation() -> None:
    user0 = (1 / 1 + 2 / 3) / 2  # hits at ranks 1 and 3, two relevant items
    user1 = (1 / 4) / 1  # hit at rank 4, one relevant item
    expected = (user0 + user1) / 2

    assert map_at_k(RECOMMENDATIONS, USERS, GROUND_TRUTH, k=5) == pytest.approx(expected)


def test_perfect_and_empty_rankings_bracket_the_range() -> None:
    perfect = np.array([[1, 7, 0, 0, 0]])
    useless = np.array([[2, 3, 4, 5, 6]])
    truth = {0: {1, 7}}

    assert recall_at_k(perfect, np.array([0]), truth, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(perfect, np.array([0]), truth, k=5) == pytest.approx(1.0)
    assert recall_at_k(useless, np.array([0]), truth, k=5) == pytest.approx(0.0)
    assert ndcg_at_k(useless, np.array([0]), truth, k=5) == pytest.approx(0.0)


def test_asking_for_more_than_was_supplied_raises() -> None:
    with pytest.raises(ValueError, match="only 5 were given"):
        recall_at_k(RECOMMENDATIONS, USERS, GROUND_TRUTH, k=10)


def test_coverage_counts_distinct_items() -> None:
    # Eight distinct ids across the two rows, out of a catalogue of 20.
    assert catalogue_coverage(RECOMMENDATIONS, n_items=20, k=5) == pytest.approx(0.5)
    everyone_gets_the_same = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert catalogue_coverage(everyone_gets_the_same, n_items=100, k=3) == pytest.approx(0.03)


def test_gini_separates_uniform_from_winner_take_all() -> None:
    uniform = np.arange(100).reshape(10, 10)
    concentrated = np.tile(np.arange(10), (10, 1))

    assert gini_at_k(uniform, n_items=100, k=10) < 0.1
    assert gini_at_k(concentrated, n_items=100, k=10) > 0.8


def test_novelty_rewards_the_long_tail() -> None:
    popularity = np.array([1000.0, 500.0, 10.0, 1.0])
    head = np.array([[0, 1]])
    tail = np.array([[2, 3]])

    assert novelty_at_k(tail, popularity, k=2) > novelty_at_k(head, popularity, k=2)


def test_popularity_percentile_exposes_a_chart_recommender() -> None:
    popularity = np.array([5.0, 50.0, 500.0, 1.0])
    chart = np.array([[2, 1]])  # the two most popular
    obscure = np.array([[3, 0]])  # the two least popular

    assert mean_popularity_rank(chart, popularity, k=2) > 0.8
    assert mean_popularity_rank(obscure, popularity, k=2) < 0.3


def test_users_without_ground_truth_are_skipped() -> None:
    """A user with nothing held out must not count as a miss."""
    truth = {0: {1, 7}}  # user 1 absent
    assert recall_at_k(RECOMMENDATIONS, USERS, truth, k=5) == pytest.approx(1.0)


def test_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D array"):
        recall_at_k(np.array([1, 2, 3]), USERS, GROUND_TRUTH, k=2)
