"""Metrics for duplicate detection, sliced by how hard the negative was.

The aggregate number on this task is close to meaningless on its own, because
it is an average over two populations that behave completely differently:

* **random negatives** -- two unrelated questions. Any similarity measure
  separates them. A dataset made only of these produces a 0.95 AUC and no
  information.
* **hard negatives** -- questions that share most of their words and differ in
  the one that decides the answer. This is the population a deduplication
  system actually faces, because the candidates it scores have already been
  retrieved for being similar.

So every metric here can be computed per `pair_kind`, and `train.py` reports
the hard-negative slice next to the aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def classification_metrics(y_true, scores, threshold: float | None = None) -> dict:
    """Ranking quality, calibration, and accuracy at a chosen threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("need both classes present to score a classifier")

    if threshold is None:
        threshold = best_f1_threshold(y_true, scores)
    predictions = (scores >= threshold).astype(int)
    clipped = np.clip(scores, 1e-9, 1 - 1e-9)

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, clipped)),
        "brier_score": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
        "majority_class_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        "positive_rate": float(y_true.mean()),
    }


def best_f1_threshold(y_true, scores, n_steps: int = 200) -> float:
    """Threshold maximising F1 on the data it is given.

    Chosen on a validation split in `train.py`, never on the test set -- tuning
    the threshold where you report it is a quiet way of fitting to it.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, n_steps))
    best, best_score = 0.5, -1.0
    for threshold in np.unique(candidates):
        score = f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best, best_score = float(threshold), score
    return best


def metrics_by_pair_kind(
    df: pd.DataFrame, scores, threshold: float, label: str = "is_duplicate"
) -> pd.DataFrame:
    """Score each negative population against the duplicates separately.

    Hard negatives are evaluated against the same positives, so the reported
    AUC answers "can the model tell a paraphrase from a near-miss", which is
    the question the retrieval stage will actually ask it.
    """
    scores = np.asarray(scores, dtype=float)
    positives = df[label].to_numpy() == 1

    rows = []
    for kind in df.loc[~positives, "pair_kind"].unique():
        mask = positives | ((df["pair_kind"] == kind).to_numpy() & ~positives)
        subset_y = df.loc[mask, label].to_numpy()
        subset_scores = scores[mask]
        if len(np.unique(subset_y)) < 2:
            continue
        rows.append(
            {
                "negatives": str(kind),
                "n_pairs": int(mask.sum()),
                "roc_auc": float(roc_auc_score(subset_y, subset_scores)),
                "pr_auc": float(average_precision_score(subset_y, subset_scores)),
                "accuracy": float(
                    accuracy_score(subset_y, (subset_scores >= threshold).astype(int))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc")


def calibration_table(y_true, scores, n_bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed frequency, by score decile.

    A model used as a filter only needs the ranking. A model whose score is
    shown to a moderator as "87% likely duplicate" needs this table to be
    close to the diagonal, and most are not.
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(scores, edges[1:-1])

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_predicted": float(scores[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    return table


def worst_errors(
    df: pd.DataFrame, scores, n: int = 5, label: str = "is_duplicate"
) -> pd.DataFrame:
    """The most confident mistakes, for reading rather than for a metric.

    Looking at these is how you find out that a "modelling problem" is a
    labelling problem, which on real duplicate-question data it very often is.
    """
    scores = np.asarray(scores, dtype=float)
    frame = df.assign(score=scores)
    frame["error"] = np.abs(frame[label] - frame["score"])
    columns = ["question1", "question2", label, "score", "pair_kind"]
    return frame.nlargest(n, "error")[columns]
