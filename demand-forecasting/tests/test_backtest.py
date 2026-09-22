"""Backtest boundaries, metric behaviour, and an end-to-end run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import aggregate, backtest, rolling_origin_folds
from src.data import DemandConfig, generate_sales
from src.evaluate import (
    bias,
    forecast_metrics,
    forecast_value_add,
    metrics_by_horizon_day,
    smape,
    wape,
)
from src.features import build_features
from src.model import MovingAverage, SeasonalNaive, build_models, clip_forecasts
from src.train import run


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    panel = generate_sales(DemandConfig(n_skus=6, days=900, seed=37))
    return build_features(panel, 28)


def test_folds_do_not_overlap_and_move_forward(features: pd.DataFrame) -> None:
    folds = rolling_origin_folds(features["date"], horizon=28, n_folds=4)

    assert len(folds) == 4
    for fold in folds:
        assert (fold.test_end - fold.test_start).days == 27
        assert fold.origin < fold.test_start
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end < later.test_start
        assert earlier.origin < later.origin


def test_folds_respect_the_minimum_training_history(features: pd.DataFrame) -> None:
    folds = rolling_origin_folds(
        features["date"], horizon=28, n_folds=20, min_train_days=420
    )
    first_date = pd.to_datetime(features["date"]).min()
    for fold in folds:
        assert (fold.origin - first_date).days >= 420
    # Twenty folds do not fit in 900 days with that much warm-up.
    assert len(folds) < 20


def test_impossible_fold_request_raises(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="too short"):
        rolling_origin_folds(features["date"], horizon=28, n_folds=3, min_train_days=5_000)


def test_backtest_never_scores_a_row_it_trained_on(features: pd.DataFrame) -> None:
    """The invariant the whole evaluation rests on."""
    models = {"moving_average_28d": MovingAverage(28)}
    preds = backtest(features, 28, models, n_folds=3, verbose=False)

    for (fold, origin), chunk in preds.groupby(["fold", "origin"], observed=True):
        assert pd.to_datetime(chunk["date"]).min() > pd.to_datetime(origin), (
            f"fold {fold} scored a date at or before its training cut-off"
        )


def test_backtest_covers_every_scored_day_once_per_model(features: pd.DataFrame) -> None:
    models = {"seasonal_naive": SeasonalNaive(28), "moving_average_28d": MovingAverage(28)}
    preds = backtest(features, 28, models, n_folds=3, verbose=False)

    for name, chunk in preds.groupby("model", observed=True):
        assert not chunk.duplicated(subset=["sku_id", "date"]).any(), name
        assert set(chunk["days_ahead"]) <= set(range(1, 29))


def test_seasonal_naive_rejects_a_misaligned_horizon() -> None:
    """At horizon 30 the lag lands on a different weekday than the target."""
    with pytest.raises(ValueError, match="multiple of 7"):
        SeasonalNaive(30)
    SeasonalNaive(28)  # fine: 28 is four whole weeks


def test_learned_models_beat_seasonal_naive(features: pd.DataFrame) -> None:
    preds = backtest(features, 28, build_models(28), n_folds=2, verbose=False)
    table = aggregate(preds, forecast_metrics).set_index("model")

    assert table.loc["lightgbm_tweedie", "wape"] < table.loc["seasonal_naive", "wape"]


def test_wape_is_volume_weighted_not_row_weighted() -> None:
    """The reason WAPE is used instead of MAPE.

    One fast SKU forecast perfectly and one slow SKU forecast badly: MAPE is
    dominated by the slow one, WAPE by the volume that actually moves.
    """
    y_true = np.array([1000.0, 2.0])
    y_pred = np.array([1000.0, 4.0])

    assert wape(y_true, y_pred) == pytest.approx(2.0 / 1002.0)
    mape = np.mean(np.abs(y_true - y_pred) / y_true)
    assert mape == pytest.approx(0.5)  # 50% error from a two-unit miss


def test_wape_is_defined_when_actuals_are_zero() -> None:
    y_true = np.array([0.0, 0.0, 10.0])
    y_pred = np.array([1.0, 1.0, 10.0])
    assert wape(y_true, y_pred) == pytest.approx(0.2)
    assert np.isfinite(smape(y_true, y_pred))


def test_bias_has_a_sign_that_means_something() -> None:
    y_true = np.array([10.0, 10.0, 10.0])
    assert bias(y_true, np.array([12.0, 12.0, 12.0])) == pytest.approx(0.2)
    assert bias(y_true, np.array([8.0, 8.0, 8.0])) == pytest.approx(-0.2)
    # Alternating errors cancel in bias but not in WAPE -- the distinction the
    # metric exists to make.
    alternating = np.array([12.0, 8.0, 10.0])
    assert bias(y_true, alternating) == pytest.approx(0.0)
    assert wape(y_true, alternating) > 0


def test_forecast_value_add_is_signed() -> None:
    assert forecast_value_add(0.4, 0.5) == pytest.approx(0.2)
    assert forecast_value_add(0.6, 0.5) == pytest.approx(-0.2)


def test_forecasts_are_never_negative(features: pd.DataFrame) -> None:
    assert (clip_forecasts(np.array([-5.0, 0.0, 3.2])) >= 0).all()
    preds = backtest(features, 28, build_models(28), n_folds=1, verbose=False)
    assert (preds["forecast"] >= 0).all()


def test_error_does_not_improve_further_from_the_origin(features: pd.DataFrame) -> None:
    """Later days must not be *easier* than nearer ones.

    Every feature is lagged by the full horizon, so the curve should be flat
    within noise. A pronounced downward slope would mean the far end of the
    window is informed by data from inside it.
    """
    preds = backtest(features, 28, {"lightgbm_tweedie": build_models(28)["lightgbm_tweedie"]},
                     n_folds=2, verbose=False)
    by_day = metrics_by_horizon_day(preds)
    first_week = by_day[by_day["days_ahead"] <= 7]["wape"].mean()
    last_week = by_day[by_day["days_ahead"] > 21]["wape"].mean()

    assert last_week >= first_week - 0.10, (
        "the far end of the horizon is suspiciously accurate"
    )


def test_run_produces_a_metrics_file(tmp_path) -> None:
    summary = run(
        cfg=DemandConfig(n_skus=5, days=760, seed=41),
        horizon=28,
        n_folds=2,
        output_dir=tmp_path,
        make_plots=False,
        verbose=False,
    )

    assert (tmp_path / "metrics.json").exists()
    assert summary["best_model"] in {m["model"] for m in summary["models"]}
    assert summary["n_folds"] == 2
