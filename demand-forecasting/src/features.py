"""Horizon-aware feature engineering.

The rule this module exists to enforce: **when forecasting `h` days ahead,
every feature derived from the target series must be lagged by at least `h`.**

`units_lag_1` is the strongest feature you can build for a sales forecaster and
it is unusable at a 28-day horizon, because on the day you produce the forecast
yesterday's sales for the target date have not happened yet. Building it anyway
is the single most common bug in retail forecasting projects: cross-validation
looks excellent, the model ships, and it is beaten by a seasonal naive
baseline, because in production that column arrives full of nulls or, worse,
stale values.

So `build_features` takes the horizon as an argument and derives every lag from
it. There is no way to ask this module for `lag_1` at a 28-day horizon.

Covariates are split by what is knowable at forecast time:

* **Past covariates** -- sales history. Must be lagged by `horizon`.
* **Future covariates** -- price, promotion, calendar. These are *planned* by a
  human weeks ahead or are deterministic, so their values on the target date
  are known when the forecast is made and they enter unlagged.

Treating the second group as if it were the first throws away most of the
signal about promotions, which is where the forecast error concentrates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Offsets added to the horizon. Lag `h` is the freshest usable observation;
# the others give the model a short trend and the same weekday a week back.
LAG_OFFSETS = [0, 1, 2, 7, 14, 28]
ROLLING_WINDOWS = [7, 28, 91]
YEARLY_LAG = 364  # a whole number of weeks, so weekday alignment is preserved

FUTURE_COVARIATES = [
    "price",
    "discount_pct",
    "on_promo",
    "is_holiday",
    "day_of_week",
    "is_weekend",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "days_since_start",
]

STATIC_FEATURES = ["category"]


def lag_columns(horizon: int) -> list[str]:
    names = [f"units_lag_{horizon + off}" for off in LAG_OFFSETS]
    if YEARLY_LAG >= horizon:
        names.append(f"units_lag_{YEARLY_LAG}")
    return names


def rolling_columns(horizon: int) -> list[str]:
    cols: list[str] = []
    for window in ROLLING_WINDOWS:
        cols += [
            f"units_mean_{window}d_lag_{horizon}",
            f"units_std_{window}d_lag_{horizon}",
        ]
    cols += [
        f"units_nonzero_rate_28d_lag_{horizon}",
        f"promo_days_28d_lag_{horizon}",
        f"units_trend_ratio_lag_{horizon}",
    ]
    return cols


def feature_columns(horizon: int) -> list[str]:
    return lag_columns(horizon) + rolling_columns(horizon) + FUTURE_COVARIATES + STATIC_FEATURES


def _validate(panel: pd.DataFrame, horizon: int) -> None:
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    required = {"date", "sku_id", "units", "price", "base_price", "on_promo"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"sales panel is missing columns: {sorted(missing)}")


def build_features(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return the panel with features appended, all safe at `horizon` days out.

    Rows early in each SKU's history will carry NaNs where a lag reaches back
    before the series starts. They are kept rather than dropped: LightGBM
    handles missing values natively, and dropping them would silently remove
    every new product from the training set -- which is exactly the population
    a forecaster struggles with most.
    """
    _validate(panel, horizon)
    df = panel.sort_values(["sku_id", "date"], kind="mergesort").copy()
    grp = df.groupby("sku_id", sort=False, observed=True)["units"]

    for offset in LAG_OFFSETS:
        lag = horizon + offset
        df[f"units_lag_{lag}"] = grp.shift(lag)
    if YEARLY_LAG >= horizon:
        df[f"units_lag_{YEARLY_LAG}"] = grp.shift(YEARLY_LAG)

    # Every rolling statistic is computed on the series *already shifted* by the
    # horizon, so the window can only contain observations that exist at
    # forecast time.
    shifted = grp.shift(horizon)
    shifted_by_sku = shifted.groupby(df["sku_id"], sort=False, observed=True)

    for window in ROLLING_WINDOWS:
        roll = shifted_by_sku.rolling(window, min_periods=max(2, window // 4))
        df[f"units_mean_{window}d_lag_{horizon}"] = roll.mean().to_numpy()
        df[f"units_std_{window}d_lag_{horizon}"] = roll.std().to_numpy()

    # Share of recent days with any sale: separates slow movers from a SKU that
    # is simply out of distribution this month. Computed as a rolling mean of
    # an indicator rather than `rolling().apply()`, which would drop into a
    # Python loop per window and dominate the runtime of the whole pipeline.
    nonzero = (shifted > 0).astype(float).where(shifted.notna())
    df[f"units_nonzero_rate_28d_lag_{horizon}"] = (
        nonzero.groupby(df["sku_id"], sort=False, observed=True)
        .rolling(28, min_periods=7)
        .mean()
        .to_numpy()
    )

    promo_shifted = (
        df.groupby("sku_id", sort=False, observed=True)["on_promo"]
        .shift(horizon)
        .astype(float)
    )
    df[f"promo_days_28d_lag_{horizon}"] = (
        promo_shifted.groupby(df["sku_id"], sort=False, observed=True)
        .rolling(28, min_periods=7)
        .sum()
        .to_numpy()
    )

    # Short window over long window: is this SKU accelerating or decaying?
    short = df[f"units_mean_{ROLLING_WINDOWS[0]}d_lag_{horizon}"]
    long = df[f"units_mean_{ROLLING_WINDOWS[-1]}d_lag_{horizon}"]
    df[f"units_trend_ratio_lag_{horizon}"] = np.where(
        long.to_numpy() > 0, short.to_numpy() / long.to_numpy(), np.nan
    )

    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear.to_numpy()
    df["day_of_week"] = dates.dt.dayofweek.astype("int16")
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype("int8")
    df["month"] = dates.dt.month.astype("int16")
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["days_since_start"] = (dates - dates.min()).dt.days.astype("int32")
    df["discount_pct"] = 1.0 - df["price"] / df["base_price"]
    df["on_promo"] = df["on_promo"].astype("int8")
    df["is_holiday"] = df["is_holiday"].astype("int8")

    return df


def training_frame(
    df: pd.DataFrame, horizon: int, drop_censored: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the feature frame into X and y for fitting.

    `drop_censored` removes days the SKU was out of stock. A zero on a stockout
    day is not demand of zero, it is *unobserved* demand, and training on it
    teaches the model to forecast the warehouse's failures instead of the
    customer's intent. Most real datasets do not come with a stock flag, which
    is why this is usually a data-collection fight rather than a modelling one.
    """
    mask = pd.Series(True, index=df.index)
    if drop_censored and "was_in_stock" in df.columns:
        mask &= df["was_in_stock"].astype(bool)
    cols = feature_columns(horizon)
    return df.loc[mask, cols], df.loc[mask, "units"]
