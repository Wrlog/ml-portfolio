"""The horizon-safety tests.

A forecasting feature bug does not crash anything. It produces a
cross-validation score that looks excellent and a model that loses to seasonal
naive in production. These tests assert the horizon constraint mechanically
rather than trusting the naming convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import DemandConfig, generate_sales
from src.features import (
    LAG_OFFSETS,
    YEARLY_LAG,
    build_features,
    feature_columns,
    lag_columns,
    training_frame,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return generate_sales(DemandConfig(n_skus=8, days=800, seed=31))


def test_no_lag_is_shorter_than_the_horizon() -> None:
    """The whole point of the module: at horizon h there is no lag < h."""
    for horizon in (1, 7, 14, 28, 56):
        for name in lag_columns(horizon):
            lag = int(name.rsplit("_", 1)[1])
            assert lag >= horizon, f"{name} is unusable at a {horizon}-day horizon"


def test_lag_values_match_a_manual_shift(panel: pd.DataFrame) -> None:
    horizon = 28
    feats = build_features(panel, horizon)

    one_sku = feats[feats["sku_id"] == "SKU-003"].sort_values("date")
    expected = one_sku["units"].shift(horizon)
    pd.testing.assert_series_equal(
        one_sku[f"units_lag_{horizon}"], expected, check_names=False
    )


def test_rolling_windows_only_see_observable_history(panel: pd.DataFrame) -> None:
    """A 7-day mean at horizon 28 must cover days t-34 .. t-28, not t-6 .. t."""
    horizon = 28
    feats = build_features(panel, horizon).sort_values(["sku_id", "date"])
    one_sku = feats[feats["sku_id"] == "SKU-002"].reset_index(drop=True)

    units = one_sku["units"].to_numpy(dtype=float)
    computed = one_sku[f"units_mean_7d_lag_{horizon}"].to_numpy()

    for i in (300, 450, 600):
        window = units[i - horizon - 6 : i - horizon + 1]
        assert np.isclose(computed[i], window.mean()), f"row {i} window is misaligned"


def test_features_do_not_change_when_the_future_is_removed(panel: pd.DataFrame) -> None:
    """Truncating the panel must not alter features for the rows that remain."""
    horizon = 28
    cutoff = panel["date"].quantile(0.7)

    full = build_features(panel, horizon)
    truncated = build_features(panel[panel["date"] <= cutoff].copy(), horizon)

    cols = [c for c in feature_columns(horizon) if c != "category"]
    left = full[full["date"] <= cutoff].sort_values(["sku_id", "date"])[cols]
    right = truncated.sort_values(["sku_id", "date"])[cols]

    pd.testing.assert_frame_equal(
        left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False
    )


def test_future_covariates_are_not_lagged(panel: pd.DataFrame) -> None:
    """Price and promotion are planned ahead, so they enter at their own date.

    Lagging them would throw away the promotional signal, which is where the
    error concentrates.
    """
    feats = build_features(panel, 28)
    pd.testing.assert_series_equal(
        feats["price"].reset_index(drop=True),
        panel.sort_values(["sku_id", "date"])["price"].reset_index(drop=True),
        check_names=False,
    )


def test_yearly_lag_is_weekday_aligned() -> None:
    assert YEARLY_LAG % 7 == 0, "a 365-day lag lands on a different weekday each year"


def test_short_horizon_drops_the_yearly_lag_only_when_impossible() -> None:
    assert f"units_lag_{YEARLY_LAG}" in lag_columns(28)
    assert f"units_lag_{YEARLY_LAG}" not in lag_columns(400)


def test_training_frame_drops_stockout_days(panel: pd.DataFrame) -> None:
    """A zero on a stockout day is unobserved demand, not demand of zero."""
    feats = build_features(panel, 28)
    X_all, y_all = training_frame(feats, 28, drop_censored=False)
    X_clean, y_clean = training_frame(feats, 28, drop_censored=True)

    assert len(X_clean) < len(X_all)
    assert len(X_clean) == int(feats["was_in_stock"].sum())
    assert y_clean.mean() > y_all.mean()  # the censored zeros drag the mean down


def test_invalid_inputs_are_rejected(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="horizon"):
        build_features(panel, 0)
    with pytest.raises(KeyError, match="missing columns"):
        build_features(panel.drop(columns=["price"]), 28)


def test_all_declared_features_exist(panel: pd.DataFrame) -> None:
    horizon = 14
    feats = build_features(panel, horizon)
    missing = set(feature_columns(horizon)) - set(feats.columns)
    assert not missing, f"declared but not built: {sorted(missing)}"
    assert len(LAG_OFFSETS) == len(set(LAG_OFFSETS)), "duplicate lag offsets"
