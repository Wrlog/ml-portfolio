"""Evaluation built around the decision, not the classifier.

At a 0.17% base rate, accuracy is 99.83% for a model that approves every
transaction, ROC-AUC is dominated by the 99.8% of the curve nobody operates
on, and the 0.5 probability cut-off is arbitrary. So this module reports:

* **PR-AUC** as the headline ranking metric, with the positive rate as its
  floor -- a PR-AUC of 0.40 at a 0.17% base rate is a 235x lift, and saying
  so is more honest than quoting 0.98 ROC-AUC.
* **A cost curve.** A missed fraud costs the transaction amount plus a
  chargeback fee; an alert costs analyst time, and a false alert additionally
  costs a blocked customer. Those numbers are not equal, they are not equal
  *per transaction* either (amount varies by two orders of magnitude), and
  the threshold that minimises total cost is the deliverable.
* **Capacity metrics.** A review team handles a fixed number of alerts a day.
  Precision at that budget is what the team actually experiences.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class CostModel:
    """Unit economics of a single decision, in dollars.

    Defaults are order-of-magnitude figures for a mid-size card issuer. They
    are assumptions, not measurements -- the point of keeping them in one
    dataclass is that a fraud lead can change them and re-derive the operating
    point without touching the model.
    """

    loss_given_fraud: float = 1.0  # share of the amount the issuer eats
    chargeback_fee: float = 25.0  # fixed admin cost per confirmed fraud loss
    review_cost: float = 3.50  # analyst time per alert raised
    false_positive_friction: float = 12.00  # blocked good customer: support + churn risk

    def baseline_cost(self, y_true: np.ndarray, amounts: np.ndarray) -> float:
        """Cost of approving everything -- the do-nothing comparison."""
        y_true = np.asarray(y_true)
        amounts = np.asarray(amounts, dtype=float)
        fraud = y_true == 1
        return float(
            amounts[fraud].sum() * self.loss_given_fraud + fraud.sum() * self.chargeback_fee
        )


@dataclass(frozen=True)
class OperatingPoint:
    """What happens when the alert threshold is set to `threshold`."""

    threshold: float
    alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    alert_rate: float
    amount_recovered: float
    amount_missed: float
    total_cost: float
    net_savings: float
    savings_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def ranking_metrics(y_true, scores) -> dict:
    """Threshold-free quality of the score's ordering."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    base_rate = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, scores))
    return {
        "pr_auc": pr_auc,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "base_rate": base_rate,
        "pr_auc_lift_over_random": pr_auc / base_rate if base_rate > 0 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "log_loss": float(log_loss(y_true, np.clip(scores, 1e-9, 1 - 1e-9))),
    }


def _sorted_arrays(y_true, scores, amounts):
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if not (len(y_true) == len(scores) == len(amounts)):
        raise ValueError("y_true, scores and amounts must be the same length")
    order = np.argsort(-scores, kind="mergesort")  # stable: ties keep input order
    return y_true[order], scores[order], amounts[order]


def cost_curve(y_true, scores, amounts, costs: CostModel | None = None) -> pd.DataFrame:
    """Total cost at every possible alert volume, from zero alerts to all.

    Row `k` is the state of the world when the top `k` scoring transactions
    are alerted on.
    """
    costs = costs or CostModel()
    y_s, s_s, amt_s = _sorted_arrays(y_true, scores, amounts)
    n = len(y_s)
    total_pos = float(y_s.sum())
    total_fraud_amount = float((y_s * amt_s).sum())

    # Prepend the zero-alert state so the curve starts at the do-nothing cost.
    cum_tp = np.concatenate([[0.0], np.cumsum(y_s)])
    cum_caught = np.concatenate([[0.0], np.cumsum(y_s * amt_s)])
    k = np.arange(n + 1, dtype=float)

    fp = k - cum_tp
    fn = total_pos - cum_tp
    missed_amount = total_fraud_amount - cum_caught

    total_cost = (
        missed_amount * costs.loss_given_fraud
        + fn * costs.chargeback_fee
        + k * costs.review_cost
        + fp * costs.false_positive_friction
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(k > 0, cum_tp / np.maximum(k, 1), np.nan)

    # Threshold for k alerts is the k-th highest score; k=0 means "alert on
    # nothing", represented by a threshold above every observed score.
    thresholds = np.concatenate([[np.inf], s_s])

    baseline = costs.baseline_cost(np.asarray(y_true), np.asarray(amounts))
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "alerts": k.astype(int),
            "alert_rate": k / n,
            "true_positives": cum_tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": cum_tp / total_pos if total_pos else np.nan,
            "amount_recovered": cum_caught,
            "amount_missed": missed_amount,
            "total_cost": total_cost,
            "net_savings": baseline - total_cost,
        }
    )


def choose_threshold(
    y_true,
    scores,
    amounts,
    costs: CostModel | None = None,
    max_alert_rate: float | None = None,
) -> float:
    """Threshold that minimises total cost, optionally within review capacity.

    `max_alert_rate` is the review team's throughput as a share of traffic.
    Ignoring it produces a mathematically optimal threshold that floods a team
    which can only work a few hundred cases a day.
    """
    curve = cost_curve(y_true, scores, amounts, costs)
    if max_alert_rate is not None:
        curve = curve[curve["alert_rate"] <= max_alert_rate]
        if curve.empty:
            raise ValueError("max_alert_rate excludes every operating point")
    best = curve.loc[curve["total_cost"].idxmin()]
    return float(best["threshold"])


def operating_point(
    y_true, scores, amounts, threshold: float, costs: CostModel | None = None
) -> OperatingPoint:
    """Score a single threshold on a (held-out) set."""
    costs = costs or CostModel()
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    alerted = scores >= threshold
    tp = int(((y_true == 1) & alerted).sum())
    fp = int(((y_true == 0) & alerted).sum())
    fn = int(((y_true == 1) & ~alerted).sum())
    n_alerts = tp + fp

    recovered = float(amounts[(y_true == 1) & alerted].sum())
    missed = float(amounts[(y_true == 1) & ~alerted].sum())
    total_cost = (
        missed * costs.loss_given_fraud
        + fn * costs.chargeback_fee
        + n_alerts * costs.review_cost
        + fp * costs.false_positive_friction
    )
    baseline = costs.baseline_cost(y_true, amounts)

    return OperatingPoint(
        threshold=float(threshold),
        alerts=n_alerts,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=tp / n_alerts if n_alerts else 0.0,
        recall=tp / max(int((y_true == 1).sum()), 1),
        alert_rate=n_alerts / len(y_true),
        amount_recovered=recovered,
        amount_missed=missed,
        total_cost=total_cost,
        net_savings=baseline - total_cost,
        savings_rate=(baseline - total_cost) / baseline if baseline else 0.0,
    )


def precision_at_budget(y_true, scores, budget: int) -> dict:
    """Precision and recall when the team can only review `budget` cases.

    This is the number a fraud analyst feels day to day: out of the cases put
    in front of them, how many were real.
    """
    y_s, _, _ = _sorted_arrays(y_true, scores, np.ones(len(y_true)))
    budget = int(min(budget, len(y_s)))
    if budget <= 0:
        raise ValueError("budget must be positive")
    caught = float(y_s[:budget].sum())
    total_pos = float(y_s.sum())
    return {
        "budget": budget,
        "precision": caught / budget,
        "recall": caught / total_pos if total_pos else float("nan"),
    }


def recall_at_precision(y_true, scores, target_precision: float) -> float:
    """Highest recall reachable while holding precision at or above target.

    Returns 0.0 when the target is unreachable -- which is itself the useful
    answer, because it says the model cannot support that review workload.
    """
    curve = cost_curve(y_true, scores, np.ones(len(y_true)))
    feasible = curve[(curve["alerts"] > 0) & (curve["precision"] >= target_precision)]
    return float(feasible["recall"].max()) if not feasible.empty else 0.0
