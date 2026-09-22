"""Forecast accuracy metrics, chosen for a catalogue with mixed velocity.

MAPE is the metric everyone asks for and it is the wrong one here. It divides
by the actual, so a SKU selling 2 units a day dominates the average over one
selling 400, and on any day with zero sales it is undefined or infinite. A
60-SKU catalogue with slow movers and stockouts will produce both problems
within the first week.

So the headline metric is **WAPE** -- total absolute error divided by total
actual units. It is interpretable in the units the business cares about
("we are off by 23% of volume"), it is well defined at zero, and it weights
each SKU by how much it actually sells, which is what an inventory budget
does too.

Two other things get reported that accuracy metrics hide:

* **Bias.** A forecast that is 20% low every single day and one that alternates
  ±20% have identical WAPE and completely different consequences: the first
  empties the shelf, the second just churns the warehouse. Bias separates them.
* **Forecast value add** over seasonal naive. The absolute error number is
  meaningless without knowing what doing nothing would have cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot score an empty forecast")
    return y_true, y_pred


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|error| / sum(actual)."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denominator)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, reported only because stakeholders ask for a percentage.

    Still ill-behaved when actual and forecast are both near zero; the
    denominator guard keeps it finite rather than making it meaningful.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)[mask] / denominator[mask]))


def bias(y_true, y_pred) -> float:
    """Signed error as a share of actual volume. Positive means over-forecast."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float((y_pred - y_true).sum() / denominator)


def forecast_metrics(y_true, y_pred) -> dict:
    return {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def forecast_value_add(model_wape: float, baseline_wape: float) -> float:
    """Share of the baseline's error the model removes. Negative means worse."""
    if baseline_wape <= 0:
        return float("nan")
    return (baseline_wape - model_wape) / baseline_wape


def metrics_by_group(
    frame: pd.DataFrame, group: str, actual: str = "units", pred: str = "forecast"
) -> pd.DataFrame:
    """Per-group WAPE and bias, sorted by volume.

    The aggregate hides the failure modes. A model can look fine overall and be
    unusable on the slow-moving third of the catalogue, which is where
    inventory money gets stuck.
    """
    rows = []
    for key, chunk in frame.groupby(group, observed=True):
        rows.append(
            {
                group: key,
                "actual_units": float(chunk[actual].sum()),
                "wape": wape(chunk[actual], chunk[pred]),
                "bias": bias(chunk[actual], chunk[pred]),
                "n": int(len(chunk)),
            }
        )
    return pd.DataFrame(rows).sort_values("actual_units", ascending=False)


def metrics_by_horizon_day(
    frame: pd.DataFrame, actual: str = "units", pred: str = "forecast"
) -> pd.DataFrame:
    """Error as a function of how far ahead the forecast reaches.

    How to read this curve depends on the design. Here a *single* model is fit
    with every feature lagged by the full horizon, so day 1 and day 28 of the
    window are predicted from exactly the same information. The curve is
    therefore expected to be roughly **flat**, and what remains is drift: the
    world moving away from the forecast origin.

    A steep decay would say the horizon is too long for the features. A curve
    that gets *better* further out is the alarming one -- it means something in
    the later window is informed by data from inside it.

    The alternative design, one model per horizon step so that day 1 may use
    `lag_1`, buys real accuracy at the near end and costs 28 models to train
    and monitor. See the README for why this project did not take it.
    """
    if "days_ahead" not in frame.columns:
        raise KeyError("frame needs a 'days_ahead' column from the backtest")
    rows = []
    for days_ahead, chunk in frame.groupby("days_ahead", observed=True):
        rows.append(
            {
                "days_ahead": int(days_ahead),
                "wape": wape(chunk[actual], chunk[pred]),
                "bias": bias(chunk[actual], chunk[pred]),
                "n": int(len(chunk)),
            }
        )
    return pd.DataFrame(rows).sort_values("days_ahead")
