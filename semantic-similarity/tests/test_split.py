"""Tests for the splitting protocols -- the part of this project that matters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import PairConfig, generate_pairs
from src.split import (
    assert_no_question_overlap,
    cluster_disjoint_split,
    connected_components,
    random_pair_split,
    split_report,
)


@pytest.fixture(scope="module")
def pairs() -> pd.DataFrame:
    return generate_pairs(PairConfig(n_pairs=6_000, n_intents=200, seed=71))


def test_cluster_disjoint_split_shares_no_question(pairs: pd.DataFrame) -> None:
    split = cluster_disjoint_split(pairs, test_size=0.25, seed=0)
    assert_no_question_overlap(pairs, split)  # raises if any question crosses


def test_random_split_leaks_questions(pairs: pd.DataFrame) -> None:
    """The failure this project is about, asserted rather than described."""
    split = random_pair_split(pairs, test_size=0.25, seed=0)
    with pytest.raises(AssertionError, match="appear in both train and test"):
        assert_no_question_overlap(pairs, split)


def test_no_cluster_appears_on_both_sides(pairs: pd.DataFrame) -> None:
    split = cluster_disjoint_split(pairs, test_size=0.25, seed=0)
    train_clusters = set(pairs.loc[split.train, "cluster1"]) | set(
        pairs.loc[split.train, "cluster2"]
    )
    test_clusters = set(pairs.loc[split.test, "cluster1"]) | set(
        pairs.loc[split.test, "cluster2"]
    )
    assert not (train_clusters & test_clusters)


def test_train_and_test_do_not_overlap(pairs: pd.DataFrame) -> None:
    for split in (
        random_pair_split(pairs, 0.25, seed=1),
        cluster_disjoint_split(pairs, 0.25, seed=1),
    ):
        assert not (split.train & split.test).any(), split.name


def test_disjoint_split_drops_boundary_pairs_and_says_so(pairs: pd.DataFrame) -> None:
    split = cluster_disjoint_split(pairs, test_size=0.25, seed=0)
    report = split_report(pairs, split)

    assert report["dropped_pairs"] > 0
    assert report["dropped_pairs"] == int(split.dropped.sum())
    assert "dropped" in split.summary(pairs)


def test_rebalancing_restores_the_original_duplicate_rate(pairs: pd.DataFrame) -> None:
    """Without it the two protocols would be scored at different base rates.

    PR-AUC and accuracy both move with the positive rate, so an unrebalanced
    comparison would confound the protocol effect with a base-rate effect.
    """
    rebalanced = split_report(pairs, cluster_disjoint_split(pairs, 0.25, seed=2))
    raw = split_report(
        pairs, cluster_disjoint_split(pairs, 0.25, seed=2, rebalance=False)
    )

    original = rebalanced["original_duplicate_rate"]
    assert rebalanced["test_duplicate_rate"] == pytest.approx(original, abs=0.02)
    assert raw["test_duplicate_rate"] > original + 0.10  # the distortion it corrects


def test_the_cluster_graph_is_one_component(pairs: pd.DataFrame) -> None:
    """Why whole-component assignment is not an option.

    Negative pairs join arbitrary clusters, so the graph collapses. If this
    ever returns many components the simpler splitter becomes viable.
    """
    assert connected_components(pairs).nunique() == 1


def test_invalid_test_size_is_rejected(pairs: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        cluster_disjoint_split(pairs, test_size=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        cluster_disjoint_split(pairs, test_size=1.0)


def test_splits_are_reproducible(pairs: pd.DataFrame) -> None:
    a = cluster_disjoint_split(pairs, 0.25, seed=5)
    b = cluster_disjoint_split(pairs, 0.25, seed=5)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)
