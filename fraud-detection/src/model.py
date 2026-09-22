"""Model definitions, from the rule engine outward.

The comparison is deliberately laddered:

1. `RuleBaseline` -- the static if/else engine a fraud team already runs. If a
   model cannot beat this it has no business being deployed, and quoting a
   lift over "random" instead of over the incumbent is the most common way a
   portfolio project overstates its result.
2. Regularised logistic regression -- the interpretable, monotone reference.
3. LightGBM -- the production candidate.
4. LightGBM with `scale_pos_weight` -- included because it is the reflex
   answer to imbalance, and because on this data it is the wrong one. At a
   0.17% base rate the balancing weight is ~590x, which lets a few hundred
   positives dominate every split; the trees fit those specific frauds and
   the *ranking* gets worse, not just the calibration (PR-AUC 0.75 -> 0.40 in
   the run recorded in the README). The lesson the metric teaches: imbalance
   is a problem with the decision threshold, not with the loss function, and
   it is fixed in `evaluate.py` by pricing the two error types -- not by
   reweighting and leaving the cut-off at 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


class RuleBaseline(BaseEstimator, ClassifierMixin):
    """A hand-written rule engine, scored as a probability.

    Each rule contributes a fixed weight; the weights are passed through a
    logistic so the output can be ranked and thresholded like any other model.
    Nothing is learned from the data -- `fit` only records the classes -- which
    is the point: this is the incumbent, not a competitor.
    """

    RULES: dict[str, float] = {
        "is_foreign": 1.1,
        "card_not_present": 0.8,
        "large_vs_history": 1.3,
        "night": 0.4,
        "velocity": 1.5,
        "far_from_home": 0.9,
        "risky_category": 0.7,
    }
    RISKY_CATEGORIES = {"gift_card", "digital_goods", "electronics"}

    def __init__(self, amount_ratio_threshold: float = 3.0, distance_threshold_km: float = 300.0):
        self.amount_ratio_threshold = amount_ratio_threshold
        self.distance_threshold_km = distance_threshold_km

    def fit(self, X, y=None):  # noqa: D102 - nothing is learned, by design
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
        return self

    def _rule_hits(self, X: pd.DataFrame) -> pd.DataFrame:
        category = X["merchant_category"].astype(str)
        return pd.DataFrame(
            {
                "is_foreign": X["is_foreign"].astype(bool),
                "card_not_present": ~X["card_present"].astype(bool),
                "large_vs_history": X["amount_to_account_mean"] > self.amount_ratio_threshold,
                "night": X["is_night"].astype(bool),
                "velocity": X["txn_count_1h"] >= 2,
                "far_from_home": X["distance_from_home_km"] > self.distance_threshold_km,
                "risky_category": category.isin(self.RISKY_CATEGORIES),
            },
            index=X.index,
        )

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        hits = self._rule_hits(X)
        weights = pd.Series(self.RULES)
        score = hits.astype(float).mul(weights, axis=1).sum(axis=1)
        return (score - 2.2).to_numpy()  # intercept sets a sane base rate

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _lightgbm(random_state: int, scale_pos_weight: float | None = None) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_models(random_state: int = 7, pos_weight: float = 1.0) -> dict:
    """Return the model ladder, keyed by the name used in reports."""
    return {
        "rule_baseline": RuleBaseline(),
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        max_iter=2_000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "lightgbm": _lightgbm(random_state),
        "lightgbm_weighted": _lightgbm(random_state, scale_pos_weight=pos_weight),
    }


def positive_class_weight(y) -> float:
    """`scale_pos_weight` that balances the two classes: n_neg / n_pos."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("cannot weight classes: no positive examples")
    return float((y == 0).sum() / n_pos)
