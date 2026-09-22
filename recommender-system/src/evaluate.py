"""Ranking metrics, plus the ones that stop a degenerate model passing.

RMSE does not appear here. It is the metric for rating prediction, and this is
not a rating-prediction problem: there are no negatives to regress against, and
the product question is "what are the ten best things to show" -- an ordering,
not a value. Recall@K, NDCG@K and MAP@K answer that question.

Accuracy metrics alone are not enough either, because they are all maximised by
a model that recommends the head of the catalogue to everybody. So three more
are reported alongside:

* **Catalogue coverage** -- how much of the catalogue ever gets recommended.
* **Novelty** -- mean self-information of recommended items. Low novelty means
  the model is recycling the chart.
* **Gini** -- how unequally recommendations are spread over items.

A model that wins on recall while losing badly on all three is not a better
recommender; it has just found the popularity prior.

All metrics rank over the **full catalogue**. Sampling a hundred negatives per
user is far cheaper and is known to be inconsistent -- it can reverse the order
of two models (Krichene & Rendle, KDD 2020) -- so with 2,000 items it is not
worth the risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate(recommendations: np.ndarray, k: int) -> np.ndarray:
    recommendations = np.asarray(recommendations)
    if recommendations.ndim != 2:
        raise ValueError("recommendations must be a 2-D array of item ids")
    if k > recommendations.shape[1]:
        raise ValueError(f"asked for k={k} but only {recommendations.shape[1]} were given")
    return recommendations[:, :k]


def recall_at_k(recommendations, user_ids, ground_truth: dict, k: int) -> float:
    """Share of each user's held-out items that appear in their top k."""
    top = _validate(recommendations, k)
    scores = []
    for row, user in zip(top, user_ids):
        relevant = ground_truth.get(int(user), set())
        if not relevant:
            continue
        scores.append(len(relevant & set(row.tolist())) / len(relevant))
    return float(np.mean(scores)) if scores else float("nan")


def hit_rate_at_k(recommendations, user_ids, ground_truth: dict, k: int) -> float:
    """Share of users with at least one correct recommendation."""
    top = _validate(recommendations, k)
    hits = []
    for row, user in zip(top, user_ids):
        relevant = ground_truth.get(int(user), set())
        if not relevant:
            continue
        hits.append(float(bool(relevant & set(row.tolist()))))
    return float(np.mean(hits)) if hits else float("nan")


def ndcg_at_k(recommendations, user_ids, ground_truth: dict, k: int) -> float:
    """Discounted gain, normalised by the best achievable given the hold-out size.

    Position matters: a correct item at rank 1 is worth more than the same item
    at rank 10, because almost nobody scrolls.
    """
    top = _validate(recommendations, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    scores = []
    for row, user in zip(top, user_ids):
        relevant = ground_truth.get(int(user), set())
        if not relevant:
            continue
        gains = np.fromiter((item in relevant for item in row.tolist()), float, count=k)
        ideal = discounts[: min(len(relevant), k)].sum()
        scores.append(float((gains * discounts).sum() / ideal) if ideal else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def map_at_k(recommendations, user_ids, ground_truth: dict, k: int) -> float:
    """Mean average precision -- rewards getting several hits high in the list."""
    top = _validate(recommendations, k)
    scores = []
    for row, user in zip(top, user_ids):
        relevant = ground_truth.get(int(user), set())
        if not relevant:
            continue
        hits, precision_sum = 0, 0.0
        for rank, item in enumerate(row.tolist(), start=1):
            if item in relevant:
                hits += 1
                precision_sum += hits / rank
        scores.append(precision_sum / min(len(relevant), k))
    return float(np.mean(scores)) if scores else float("nan")


def catalogue_coverage(recommendations, n_items: int, k: int) -> float:
    """Share of the catalogue that appears in anyone's top k."""
    top = _validate(recommendations, k)
    return float(len(np.unique(top)) / n_items)


def novelty_at_k(recommendations, train_popularity: np.ndarray, k: int) -> float:
    """Mean self-information, -log2(p(item)), of the recommended items.

    Higher means the model is surfacing things people have not already found.
    Recommending only the top of the chart drives this toward zero.
    """
    top = _validate(recommendations, k)
    total = train_popularity.sum()
    if total <= 0:
        return float("nan")
    probabilities = np.clip(train_popularity / total, 1e-12, None)
    self_information = -np.log2(probabilities)
    return float(self_information[top].mean())


def gini_at_k(recommendations, n_items: int, k: int) -> float:
    """Inequality of exposure across items. 0 is uniform, 1 is winner-take-all."""
    top = _validate(recommendations, k)
    counts = np.bincount(top.ravel(), minlength=n_items).astype(float)
    if counts.sum() == 0:
        return float("nan")
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_counts).sum()) / (n * sorted_counts.sum()) - (n + 1) / n)


def mean_popularity_rank(recommendations, train_popularity: np.ndarray, k: int) -> float:
    """Average popularity percentile of recommended items, 1.0 = the very head.

    The number that exposes a model whose "personalisation" is the chart.
    """
    top = _validate(recommendations, k)
    order = np.argsort(np.argsort(train_popularity))  # 0 = least popular
    percentile = order / max(len(order) - 1, 1)
    return float(percentile[top].mean())


def evaluate_recommender(
    recommendations: np.ndarray,
    user_ids: np.ndarray,
    ground_truth: dict,
    n_items: int,
    train_popularity: np.ndarray,
    k: int = 10,
) -> dict:
    """Every metric at one cut-off, accuracy and distribution together."""
    return {
        "k": k,
        "recall": recall_at_k(recommendations, user_ids, ground_truth, k),
        "ndcg": ndcg_at_k(recommendations, user_ids, ground_truth, k),
        "map": map_at_k(recommendations, user_ids, ground_truth, k),
        "hit_rate": hit_rate_at_k(recommendations, user_ids, ground_truth, k),
        "catalogue_coverage": catalogue_coverage(recommendations, n_items, k),
        "novelty": novelty_at_k(recommendations, train_popularity, k),
        "gini": gini_at_k(recommendations, n_items, k),
        "mean_popularity_percentile": mean_popularity_rank(
            recommendations, train_popularity, k
        ),
    }


def metrics_by_user_activity(
    recommendations: np.ndarray,
    user_ids: np.ndarray,
    ground_truth: dict,
    train_counts: np.ndarray,
    k: int = 10,
    n_buckets: int = 4,
) -> pd.DataFrame:
    """Accuracy split by how much history each user has.

    The aggregate is dominated by heavy users, who are also the easiest. Cold
    users are the ones a recommender is deployed to help, and they are where
    matrix factorisation degrades first.
    """
    activity = train_counts[user_ids]
    edges = np.quantile(activity, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bucket = np.digitize(activity, edges[1:-1])

    rows = []
    for b in range(n_buckets):
        mask = bucket == b
        if not mask.any():
            continue
        rows.append(
            {
                "activity_bucket": b,
                "n_users": int(mask.sum()),
                "median_train_items": float(np.median(activity[mask])),
                "recall": recall_at_k(
                    recommendations[mask], user_ids[mask], ground_truth, k
                ),
                "ndcg": ndcg_at_k(recommendations[mask], user_ids[mask], ground_truth, k),
            }
        )
    return pd.DataFrame(rows)
