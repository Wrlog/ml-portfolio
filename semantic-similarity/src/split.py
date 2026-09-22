"""Splitting question pairs without leaking questions between train and test.

This is the part of the Quora task that quietly decides the result.

A question does not appear once. "How do I reset my password on Android?" is
paired with a dozen others across the dataset. Split the *rows* at random and
that question lands in training and in test, so the model is scored on text it
has already memorised. The published leaderboard scores for that dataset are
famously inflated by exactly this, plus "magic features" built from how often
each question id occurs -- which predict the label well and mean nothing,
because the frequency is an artefact of how the dataset was assembled.

The obvious fix -- partition the graph of clusters into connected components
and assign whole components -- does not work, and finding out why is the
instructive part. Negative pairs join arbitrary clusters, so the graph is one
giant component: 8,000 pairs over 400 clusters already collapse into a single
blob, and the real Quora question graph behaves the same way. There is no
partition that keeps every existing pair intact.

So the split assigns **clusters** to sides and *discards* the pairs that
straddle the boundary. That guarantees disjointness at a real cost, which is
reported rather than hidden:

* roughly a third of pairs are dropped at a 25% test size, almost all of them
  negatives (duplicate pairs join a cluster to itself and can never straddle);
* because the drops are one-sided, the duplicate rate rises on both sides.
  `split_report` surfaces the shift so it is not mistaken for a modelling
  result.

Both splitters live here on purpose. `train.py` runs the naive one too and
reports the gap, because the size of that gap is the argument for doing any of
this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class _UnionFind:
    """Minimal union-find over integer labels."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


@dataclass(frozen=True)
class SplitResult:
    """Row masks for one partition of the pair table.

    `train` and `test` need not cover every row: a cluster-disjoint split
    deliberately drops the pairs that cross the boundary.
    """

    train: np.ndarray
    test: np.ndarray
    name: str

    @property
    def dropped(self) -> np.ndarray:
        return ~(self.train | self.test)

    def summary(self, df: pd.DataFrame) -> str:
        parts = [
            f"{self.name}: train {self.train.sum():,} pairs "
            f"({df.loc[self.train, 'is_duplicate'].mean():.1%} duplicate)",
            f"test {self.test.sum():,} pairs "
            f"({df.loc[self.test, 'is_duplicate'].mean():.1%} duplicate)",
        ]
        n_dropped = int(self.dropped.sum())
        if n_dropped:
            parts.append(f"{n_dropped:,} dropped ({n_dropped / len(df):.1%}) as boundary-crossing")
        return ", ".join(parts)


def connected_components(df: pd.DataFrame) -> pd.Series:
    """Component id per pair, from the cluster graph induced by the pairs.

    Kept because it is the diagnostic that explains why the naive approach
    fails: if this returns one component, no pair-preserving partition exists.
    """
    uf = _UnionFind()
    for a, b in zip(df["cluster1"].to_numpy(), df["cluster2"].to_numpy()):
        uf.union(int(a), int(b))
    return df["cluster1"].map(lambda c: uf.find(int(c)))


def random_pair_split(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 0
) -> SplitResult:
    """Shuffle rows and cut. The default, and the wrong answer here."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_size
    return SplitResult(train=~is_test, test=is_test, name="random pair split (leaky)")


def _rebalance(
    df: pd.DataFrame, mask: np.ndarray, target_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Drop surplus duplicates from `mask` until its duplicate rate hits target.

    Discarding boundary-crossing pairs removes negatives almost exclusively, so
    both sides come out far more duplicate-heavy than the dataset they came
    from -- on a 25% cluster split the test side lands near 70% duplicates
    against an original 37%.

    That matters beyond aesthetics: the leaky and clean protocols would be
    scored on different base rates, and PR-AUC and accuracy both move with the
    base rate, so the comparison the whole project is built on would be
    confounded. Trimming positives back to the original rate costs some data
    and keeps the two estimates comparable.
    """
    indices = np.flatnonzero(mask)
    labels = df["is_duplicate"].to_numpy()[indices]
    positives, negatives = indices[labels == 1], indices[labels == 0]

    allowed = int(round(len(negatives) * target_rate / (1.0 - target_rate)))
    if len(positives) <= allowed:
        return mask  # already at or below the target rate

    keep = rng.choice(positives, size=max(allowed, 1), replace=False)
    out = np.zeros(len(df), dtype=bool)
    out[np.concatenate([keep, negatives])] = True
    return out


def cluster_disjoint_split(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 0, rebalance: bool = True
) -> SplitResult:
    """Assign clusters to sides; drop pairs whose two clusters land on opposite sides.

    The test *cluster* budget is `test_size`, but because cross-boundary pairs
    are discarded the surviving test set is a good deal smaller than that share
    of the original table. That is the price of an honest estimate, and
    `SplitResult.summary` reports it rather than hiding it.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    clusters = pd.unique(
        np.concatenate([df["cluster1"].to_numpy(), df["cluster2"].to_numpy()])
    )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(clusters)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    test_clusters = set(shuffled[:n_test].tolist())

    in_test_1 = df["cluster1"].isin(test_clusters).to_numpy()
    in_test_2 = df["cluster2"].isin(test_clusters).to_numpy()

    is_test = in_test_1 & in_test_2
    is_train = ~in_test_1 & ~in_test_2

    if not is_test.any() or not is_train.any():
        raise ValueError(
            f"test_size={test_size} left one side empty after dropping "
            "boundary-crossing pairs"
        )

    if rebalance:
        target = float(df["is_duplicate"].mean())
        is_train = _rebalance(df, is_train, target, rng)
        is_test = _rebalance(df, is_test, target, rng)

    return SplitResult(train=is_train, test=is_test, name="cluster-disjoint split")


def split_report(df: pd.DataFrame, split: SplitResult) -> dict:
    """Numbers worth recording about a split, including the base-rate shift."""
    return {
        "name": split.name,
        "train_pairs": int(split.train.sum()),
        "test_pairs": int(split.test.sum()),
        "dropped_pairs": int(split.dropped.sum()),
        "dropped_share": float(split.dropped.mean()),
        "original_duplicate_rate": float(df["is_duplicate"].mean()),
        "train_duplicate_rate": float(df.loc[split.train, "is_duplicate"].mean()),
        "test_duplicate_rate": float(df.loc[split.test, "is_duplicate"].mean()),
        "test_hard_negative_share": float(
            (df.loc[split.test, "pair_kind"] == "hard_negative").mean()
        ),
    }


def assert_no_question_overlap(df: pd.DataFrame, split: SplitResult) -> None:
    """Raise if any question text appears on both sides of the split.

    Cheap enough to run on every training run, and it is the assertion that
    would have caught the leak in the first place.
    """
    def questions(mask: np.ndarray) -> set:
        chunk = df.loc[mask]
        return set(chunk["question1"]) | set(chunk["question2"])

    overlap = questions(split.train) & questions(split.test)
    if overlap:
        example = sorted(overlap)[:3]
        raise AssertionError(
            f"{len(overlap):,} questions appear in both train and test "
            f"under '{split.name}'. Examples: {example}"
        )
