"""Tests for the cost model and the threshold search.

The cost curve is the artefact a business decision gets made from, so it is
checked against hand-computed values rather than only for self-consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    CostModel,
    choose_threshold,
    cost_curve,
    operating_point,
    precision_at_budget,
    ranking_metrics,
    recall_at_precision,
)

COSTS = CostModel(
    loss_given_fraud=1.0, chargeback_fee=25.0, review_cost=3.5, false_positive_friction=12.0
)


def test_zero_alerts_costs_exactly_the_do_nothing_loss() -> None:
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    amounts = np.array([10.0, 100.0, 20.0, 50.0])

    curve = cost_curve(y, scores, amounts, COSTS)
    row = curve.iloc[0]

    expected = (100.0 + 50.0) + 2 * 25.0  # fraud amounts + two chargeback fees
    assert row["alerts"] == 0
    assert row["total_cost"] == pytest.approx(expected)
    assert row["net_savings"] == pytest.approx(0.0)
    assert COSTS.baseline_cost(y, amounts) == pytest.approx(expected)


def test_cost_at_a_known_operating_point() -> None:
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    amounts = np.array([10.0, 100.0, 20.0, 50.0])

    # Alert on scores >= 0.8: catches both frauds, no false positives.
    point = operating_point(y, scores, amounts, threshold=0.8, costs=COSTS)

    assert (point.true_positives, point.false_positives, point.false_negatives) == (2, 0, 0)
    assert point.precision == pytest.approx(1.0)
    assert point.recall == pytest.approx(1.0)
    assert point.total_cost == pytest.approx(2 * 3.5)  # two reviews, nothing lost
    assert point.net_savings == pytest.approx(200.0 - 7.0)


def test_operating_point_agrees_with_the_curve() -> None:
    rng = np.random.default_rng(0)
    n = 4_000
    y = (rng.random(n) < 0.02).astype(int)
    scores = np.clip(rng.beta(2, 20, n) + 0.35 * y, 0, 1)
    amounts = np.round(np.exp(rng.normal(3.2, 1.0, n)), 2)

    curve = cost_curve(y, scores, amounts, COSTS)
    row = curve.iloc[500]  # the state after alerting on the top 500
    point = operating_point(y, scores, amounts, row["threshold"], COSTS)

    assert point.alerts == pytest.approx(row["alerts"], abs=1)
    assert point.total_cost == pytest.approx(row["total_cost"], rel=1e-6)


def test_chosen_threshold_respects_review_capacity() -> None:
    rng = np.random.default_rng(1)
    n = 5_000
    y = (rng.random(n) < 0.02).astype(int)
    scores = np.clip(rng.beta(2, 20, n) + 0.25 * y, 0, 1)
    amounts = np.full(n, 500.0)  # expensive fraud: unconstrained optimum over-alerts

    unconstrained = choose_threshold(y, scores, amounts, COSTS)
    constrained = choose_threshold(y, scores, amounts, COSTS, max_alert_rate=0.01)

    assert constrained >= unconstrained
    assert (scores >= constrained).mean() <= 0.01 + 1e-9


def test_impossible_capacity_is_an_error() -> None:
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    amounts = np.ones(4)
    with pytest.raises(ValueError, match="excludes every operating point"):
        choose_threshold(y, scores, amounts, COSTS, max_alert_rate=-0.5)


def test_pr_auc_lift_is_relative_to_the_base_rate() -> None:
    rng = np.random.default_rng(2)
    n = 10_000
    y = (rng.random(n) < 0.01).astype(int)
    scores = np.clip(rng.beta(2, 20, n) + 0.3 * y, 0, 1)

    m = ranking_metrics(y, scores)
    assert m["base_rate"] == pytest.approx(y.mean())
    assert m["pr_auc_lift_over_random"] == pytest.approx(m["pr_auc"] / y.mean())
    assert m["pr_auc"] > m["base_rate"]


def test_precision_at_budget_uses_the_top_scores() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])

    assert precision_at_budget(y, scores, 2)["precision"] == pytest.approx(1.0)
    assert precision_at_budget(y, scores, 4)["precision"] == pytest.approx(0.5)
    assert precision_at_budget(y, scores, 4)["recall"] == pytest.approx(1.0)


def test_unreachable_precision_target_returns_zero_recall() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(2_000) < 0.01).astype(int)
    scores = rng.random(2_000)  # pure noise: no ordering to exploit
    assert recall_at_precision(y, scores, target_precision=0.9) == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        cost_curve(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]))
