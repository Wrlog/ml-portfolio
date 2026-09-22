"""Rolling-origin backtesting.

A single train/test split on a time series gives you one sample of the error
and no idea of its variance. Demand in December behaves nothing like demand in
March, so a model evaluated on one 28-day window is being scored on the weather
as much as on its own quality.

Rolling-origin evaluation re-fits the model at several successive cut-off dates
and forecasts the following `horizon` days from each. It answers the question
the planning team actually has -- "if we had run this every month for the last
half year, how would it have done?" -- and it exposes variance across folds,
which is usually the difference between a model that is better and a model that
got a good month.

The invariant enforced here: **at every fold, the model sees only rows dated on
or before the forecast origin.** `tests/test_backtest.py` asserts it by
checking fold boundaries directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import feature_columns, training_frame
from .model import clip_forecasts


@dataclass(frozen=True)
class Fold:
    """One forecast origin and the window it is scored on."""

    index: int
    origin: pd.Timestamp  # last date whose label the model may see
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"fold {self.index}: train <= {self.origin:%Y-%m-%d}, "
            f"score {self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d}"
        )


def rolling_origin_folds(
    dates: pd.Series, horizon: int, n_folds: int, min_train_days: int = 420
) -> list[Fold]:
    """Successive non-overlapping test windows, most recent last.

    `min_train_days` defaults to more than a year so that the yearly lag and
    the annual seasonality features have something to work with in the very
    first fold.
    """
    if n_folds < 1:
        raise ValueError("need at least one fold")
    dates = pd.to_datetime(pd.Series(dates))
    first, last = dates.min(), dates.max()

    folds: list[Fold] = []
    for i in range(n_folds):
        test_end = last - pd.Timedelta(days=i * horizon)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        origin = test_start - pd.Timedelta(days=1)
        if (origin - first).days < min_train_days:
            break
        folds.append(Fold(index=n_folds - 1 - i, origin=origin,
                          test_start=test_start, test_end=test_end))

    if not folds:
        raise ValueError(
            f"history is too short for {n_folds} folds at horizon {horizon} "
            f"with min_train_days={min_train_days}"
        )
    return sorted(folds, key=lambda f: f.origin)


def backtest(
    features: pd.DataFrame,
    horizon: int,
    models: dict,
    n_folds: int = 6,
    min_train_days: int = 420,
    drop_censored_targets: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-fit every model at every fold and return tidy per-row predictions.

    Returns one row per (model, fold, SKU, date) with the actual, the forecast
    and how many days ahead of the origin the target fell.
    """
    cols = feature_columns(horizon)
    dates = pd.to_datetime(features["date"])
    folds = rolling_origin_folds(dates, horizon, n_folds, min_train_days)

    records: list[pd.DataFrame] = []
    for fold in folds:
        if verbose:
            print(f"  {fold.describe()}")

        train_mask = dates <= fold.origin
        test_mask = (dates >= fold.test_start) & (dates <= fold.test_end)

        train_df = features[train_mask]
        X_train, y_train = training_frame(train_df, horizon, drop_censored_targets)

        test_df = features[test_mask]
        X_test = test_df[cols]
        # Stockout days stay in the *test* set on purpose: a forecast is judged
        # on what it was asked to predict, and hiding the hard days would
        # flatter every model equally.
        y_test = test_df["units"].to_numpy(dtype=float)
        days_ahead = (pd.to_datetime(test_df["date"]) - fold.origin).dt.days.to_numpy()

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = clip_forecasts(model.predict(X_test))
            records.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold.index,
                        "origin": fold.origin,
                        "date": test_df["date"].to_numpy(),
                        "sku_id": test_df["sku_id"].to_numpy(),
                        "category": test_df["category"].to_numpy(),
                        "days_ahead": days_ahead,
                        "units": y_test,
                        "forecast": preds,
                    }
                )
            )

    return pd.concat(records, ignore_index=True)


def aggregate(predictions: pd.DataFrame, metric_fn) -> pd.DataFrame:
    """Per-model metrics, plus the spread across folds.

    The fold standard deviation is reported alongside the mean because a model
    that wins on average but swings wildly between months is harder to plan
    around than a slightly worse, steadier one.
    """
    rows = []
    for name, chunk in predictions.groupby("model", observed=True):
        overall = metric_fn(chunk["units"], chunk["forecast"])
        per_fold = [
            metric_fn(f["units"], f["forecast"])[  # noqa: E231
                "wape"
            ]
            for _, f in chunk.groupby("fold", observed=True)
        ]
        rows.append(
            {
                "model": name,
                **overall,
                "wape_fold_std": float(np.std(per_fold)),
                "wape_worst_fold": float(np.max(per_fold)),
                "folds": len(per_fold),
            }
        )
    return pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)
