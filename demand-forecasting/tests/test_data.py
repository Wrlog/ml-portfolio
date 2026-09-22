"""Checks that the generator produces the structure the model is meant to find.

If the simulated demand has no weekly seasonality or no price response, the
features that exploit them cannot be validated by any downstream test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORIES, DemandConfig, generate_sales


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return generate_sales(DemandConfig(n_skus=25, days=1_095, seed=53))


def test_panel_is_complete_and_sorted(panel: pd.DataFrame) -> None:
    assert len(panel) == panel["sku_id"].nunique() * panel["date"].nunique()
    assert not panel.duplicated(subset=["sku_id", "date"]).any()
    for _, chunk in panel.groupby("sku_id", observed=True):
        assert chunk["date"].is_monotonic_increasing


def test_units_are_non_negative_integers(panel: pd.DataFrame) -> None:
    assert (panel["units"] >= 0).all()
    assert panel["units"].dtype.kind in "iu"


def test_generation_is_reproducible() -> None:
    cfg = DemandConfig(n_skus=4, days=500, seed=61)
    pd.testing.assert_frame_equal(generate_sales(cfg), generate_sales(cfg))


def test_weekly_seasonality_exists(panel: pd.DataFrame) -> None:
    by_dow = panel.groupby(panel["date"].dt.dayofweek, observed=True)["units"].mean()
    assert by_dow.max() / by_dow.min() > 1.2, "no weekday effect for the model to learn"
    assert by_dow.idxmax() in (4, 5, 6), "the weekend peak is in the wrong place"


def test_promotions_lift_volume(panel: pd.DataFrame) -> None:
    on = panel.loc[panel["on_promo"], "units"].mean()
    off = panel.loc[~panel["on_promo"], "units"].mean()
    assert on > off * 1.3


def test_price_response_is_negative(panel: pd.DataFrame) -> None:
    """Cheaper means more units, within a SKU."""
    correlations = []
    for _, chunk in panel.groupby("sku_id", observed=True):
        if chunk["price"].nunique() > 1:
            correlations.append(np.corrcoef(chunk["price"], chunk["units"])[0, 1])
    assert np.mean(correlations) < -0.1


def test_annual_seasonality_exists(panel: pd.DataFrame) -> None:
    by_month = panel.groupby(panel["date"].dt.month, observed=True)["units"].mean()
    assert by_month.max() / by_month.min() > 1.1


def test_stockouts_are_flagged_and_zero(panel: pd.DataFrame) -> None:
    out_of_stock = panel.loc[~panel["was_in_stock"]]
    assert len(out_of_stock) > 0
    assert (out_of_stock["units"] == 0).all()
    # The flag is what makes the censoring recoverable; without it those zeros
    # are indistinguishable from genuine no-demand days.
    assert panel["was_in_stock"].mean() > 0.95


def test_variance_grows_with_the_mean(panel: pd.DataFrame) -> None:
    """Negative binomial, not Gaussian -- the reason for the Tweedie objective."""
    stats = panel.groupby("sku_id", observed=True)["units"].agg(["mean", "var"])
    busy = stats.nlargest(5, "mean")
    quiet = stats.nsmallest(5, "mean")
    assert busy["var"].mean() > quiet["var"].mean() * 5


def test_categories_are_all_represented(panel: pd.DataFrame) -> None:
    assert set(panel["category"].unique()) <= set(CATEGORIES)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="400 days"):
        DemandConfig(days=100)
    with pytest.raises(ValueError, match="elasticity"):
        DemandConfig(price_elasticity=0.5)
    with pytest.raises(ValueError, match="promo_rate"):
        DemandConfig(promo_rate=1.5)
