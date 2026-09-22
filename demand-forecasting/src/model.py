"""Forecasting models, starting from the baselines that are hard to beat.

Seasonal naive is not a straw man. On daily retail data with strong weekly
seasonality it is genuinely competitive, it costs nothing to run, and it is
what the planning team is implicitly doing already. A forecasting project that
does not report it has not established that the model is worth its
maintenance cost.

Note that at a 28-day horizon seasonal naive is exactly `units_lag_28` -- 28 is
divisible by 7, so the most recent observation available at forecast time
happens to fall on the same weekday as the target. That is a convenience of
this horizon, not a general rule, and `SeasonalNaive` asserts the alignment
rather than assuming it.

For the learned model the loss function matters more than the hyperparameters.
Unit sales are non-negative, right-skewed counts with a variance that grows
with the mean. Squared error on that distribution pulls the forecast toward the
mean of a skewed distribution and produces systematic over-forecasting on the
long tail of slow movers, so a Tweedie objective is the default here and plain
L2 is kept for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from .features import YEARLY_LAG


class _ColumnBaseline(BaseEstimator, RegressorMixin):
    """Predicts by reading one already-computed feature column."""

    def __init__(self, column: str, fallback: float = 0.0):
        self.column = column
        self.fallback = fallback

    def fit(self, X: pd.DataFrame, y=None):
        if self.column not in X.columns:
            raise KeyError(f"baseline column {self.column!r} is not in the feature frame")
        # The fallback is the training mean, used only where the lag reaches
        # back before the series starts (new products).
        self.fallback_ = float(np.nanmean(y)) if y is not None else self.fallback
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.column].to_numpy(dtype=float)
        return np.nan_to_num(values, nan=getattr(self, "fallback_", self.fallback))


class SeasonalNaive(_ColumnBaseline):
    """Repeat the most recent observation on the same weekday."""

    def __init__(self, horizon: int):
        if horizon % 7 != 0:
            raise ValueError(
                f"horizon {horizon} is not a multiple of 7, so lag_{horizon} lands on a "
                "different weekday than the target; use a weekday-aligned lag instead"
            )
        self.horizon = horizon
        super().__init__(column=f"units_lag_{horizon}")


class MovingAverage(_ColumnBaseline):
    """Average of the last 28 observable days."""

    def __init__(self, horizon: int):
        self.horizon = horizon
        super().__init__(column=f"units_mean_28d_lag_{horizon}")


class LastYear(_ColumnBaseline):
    """Same day last year -- the planner's instinct for seasonal categories."""

    def __init__(self, horizon: int):
        self.horizon = horizon
        super().__init__(column=f"units_lag_{YEARLY_LAG}")


def _lightgbm(objective: str, random_state: int) -> LGBMRegressor:
    params = dict(
        objective=objective,
        n_estimators=900,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    if objective == "tweedie":
        # 1.0 is Poisson, 2.0 is gamma. Retail unit sales sit between the two:
        # a point mass at zero plus a continuous-looking right tail.
        params["tweedie_variance_power"] = 1.2
    return LGBMRegressor(**params)


def build_models(horizon: int, random_state: int = 19) -> dict:
    """The model ladder, keyed by the name used in reports."""
    return {
        "seasonal_naive": SeasonalNaive(horizon),
        "moving_average_28d": MovingAverage(horizon),
        "last_year": LastYear(horizon),
        "lightgbm_l2": _lightgbm("regression", random_state),
        "lightgbm_tweedie": _lightgbm("tweedie", random_state),
    }


def clip_forecasts(predictions: np.ndarray) -> np.ndarray:
    """Demand cannot be negative; an L2 model will happily predict it anyway."""
    return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
