"""Synthetic daily SKU demand.

Demand is built from the components a merchandising team would actually name:
a per-SKU base level, a slow trend, day-of-week and annual seasonality, price
elasticity, promotion uplift, holiday effects, and stockouts. Counts are drawn
from a negative binomial rather than a normal, because unit sales are
non-negative integers whose variance grows with the mean -- assuming
homoscedastic Gaussian noise is what makes a forecaster quietly terrible on
the slow-moving half of the catalogue.

Two properties of this generator drive design decisions downstream:

* **Stockouts censor the label.** A zero on a stockout day is not demand of
  zero, it is unobserved demand. `units` records the sale; `was_in_stock`
  records whether the number means anything. Real retail data usually does not
  hand you that second column, which is why forecasts trained naively on sales
  learn to predict the stockouts.
* **Price and promotion are known in advance.** They are planned by a human
  weeks ahead, so at forecast time their future values are available. That
  makes them *future covariates*, and unlike sales history they do not need to
  be lagged. `features.py` treats the two kinds differently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES = ["beverages", "snacks", "household", "personal_care", "frozen"]

# Category-level day-of-week multipliers. Household goods move at the weekend;
# frozen food is flatter.
_DOW_PROFILE = {
    "beverages": [0.92, 0.88, 0.90, 0.98, 1.15, 1.32, 1.08],
    "snacks": [0.90, 0.86, 0.89, 0.97, 1.18, 1.35, 1.10],
    "household": [0.85, 0.82, 0.86, 0.94, 1.12, 1.45, 1.20],
    "personal_care": [0.95, 0.93, 0.95, 1.00, 1.08, 1.18, 0.98],
    "frozen": [0.97, 0.95, 0.97, 1.01, 1.06, 1.12, 1.00],
}

# Fixed-date holidays with a demand multiplier and a lead-in window, so the
# model has to learn a ramp rather than a single spike.
_HOLIDAYS = [
    ((1, 1), 0.55, 0),    # New Year's Day: stores quiet
    ((7, 4), 1.35, 3),
    ((11, 28), 1.80, 5),  # approximated Thanksgiving week
    ((12, 24), 1.95, 10),
    ((12, 25), 0.30, 0),  # closed
]


@dataclass
class DemandConfig:
    n_skus: int = 60
    start: str = "2021-01-01"
    days: int = 1_095  # three years
    base_demand_log_mean: float = 2.6  # ~13 units/day median
    base_demand_log_sd: float = 0.9
    price_elasticity: float = -1.8  # a 10% price cut lifts units ~19%
    promo_rate: float = 0.12  # share of SKU-weeks on promotion
    stockout_rate: float = 0.012
    dispersion: float = 6.0  # negative binomial k; lower means burstier
    trend_sd: float = 0.25  # spread of per-SKU annual growth
    seed: int = 19

    def __post_init__(self) -> None:
        if self.days < 400:
            raise ValueError("need at least ~400 days for annual seasonality to be learnable")
        if self.price_elasticity >= 0:
            raise ValueError("price elasticity should be negative")
        if not 0 <= self.promo_rate < 1:
            raise ValueError("promo_rate must be a proportion")


def _holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Multiplier per date, ramping in over each holiday's lead window."""
    mult = np.ones(len(dates))
    month = dates.month.to_numpy()
    day = dates.day.to_numpy()
    doy = dates.dayofyear.to_numpy()

    for (h_month, h_day), peak, lead in _HOLIDAYS:
        target_doy = pd.Timestamp(year=2021, month=h_month, day=h_day).dayofyear
        exact = (month == h_month) & (day == h_day)
        mult = np.where(exact, mult * peak, mult)
        if lead:
            # Linear ramp from 1.0 at `lead` days out to the peak the day before.
            distance = target_doy - doy
            in_lead = (distance > 0) & (distance <= lead)
            ramp = 1.0 + (peak - 1.0) * (lead - distance + 1) / (lead + 1)
            mult = np.where(in_lead, mult * np.maximum(ramp, 0.2), mult)
    return mult


def _sku_table(rng: np.random.Generator, cfg: DemandConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": [f"SKU-{i:03d}" for i in range(cfg.n_skus)],
            "category": rng.choice(CATEGORIES, size=cfg.n_skus),
            "base_log_demand": rng.normal(
                cfg.base_demand_log_mean, cfg.base_demand_log_sd, size=cfg.n_skus
            ),
            "base_price": np.round(np.exp(rng.normal(1.6, 0.5, size=cfg.n_skus)), 2),
            "annual_growth": rng.normal(0.03, cfg.trend_sd, size=cfg.n_skus),
            "yearly_amp": rng.uniform(0.05, 0.35, size=cfg.n_skus),
            "yearly_phase": rng.uniform(0, 2 * np.pi, size=cfg.n_skus),
        }
    )


def _promotion_calendar(
    rng: np.random.Generator, cfg: DemandConfig, dates: pd.DatetimeIndex
) -> np.ndarray:
    """Promotions run for a whole week, planned per SKU.

    Drawing them per *day* would let the model exploit a pattern that no
    merchandising calendar produces, and would make the promo feature far
    easier than it is in practice.
    """
    n_weeks = int(np.ceil(len(dates) / 7)) + 1
    weekly = rng.random((cfg.n_skus, n_weeks)) < cfg.promo_rate
    week_index = ((dates - dates[0]).days // 7).to_numpy()
    return weekly[:, week_index]


def generate_sales(cfg: DemandConfig | None = None) -> pd.DataFrame:
    """Return a long-format daily sales panel: one row per SKU per date."""
    cfg = cfg or DemandConfig()
    rng = np.random.default_rng(cfg.seed)

    dates = pd.date_range(cfg.start, periods=cfg.days, freq="D")
    skus = _sku_table(rng, cfg)
    promo = _promotion_calendar(rng, cfg, dates)

    t = np.arange(cfg.days) / 365.25
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    holiday_mult = _holiday_multiplier(dates)

    frames = []
    for i, sku in skus.iterrows():
        dow_profile = np.array(_DOW_PROFILE[sku["category"]])[dow]
        yearly = 1.0 + sku["yearly_amp"] * np.sin(
            2 * np.pi * doy / 365.25 + sku["yearly_phase"]
        )
        trend = np.exp(sku["annual_growth"] * t)

        on_promo = promo[i]
        # Promotions cut price; the elasticity term then converts that into
        # units, so price and promo are not two encodings of one effect.
        discount = np.where(on_promo, rng.uniform(0.10, 0.35, size=cfg.days), 0.0)
        price = np.round(sku["base_price"] * (1.0 - discount), 2)
        price_effect = (price / sku["base_price"]) ** cfg.price_elasticity
        # Display and feature space during a promo lift units beyond the price
        # cut alone -- the reason promo needs its own flag.
        promo_uplift = np.where(on_promo, 1.25, 1.0)

        mu = (
            np.exp(sku["base_log_demand"])
            * trend
            * dow_profile
            * yearly
            * price_effect
            * promo_uplift
            * holiday_mult
        )
        mu = np.clip(mu, 0.05, None)

        k = cfg.dispersion
        units = rng.negative_binomial(k, k / (k + mu))

        in_stock = rng.random(cfg.days) > cfg.stockout_rate
        observed = np.where(in_stock, units, 0)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku_id": sku["sku_id"],
                    "category": sku["category"],
                    "price": price,
                    "base_price": sku["base_price"],
                    "on_promo": on_promo,
                    "is_holiday": holiday_mult != 1.0,
                    "was_in_stock": in_stock,
                    "units": observed.astype(int),
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["sku_id", "date"], kind="mergesort").reset_index(drop=True)
    panel["category"] = panel["category"].astype("category")
    return panel


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    frame = generate_sales()
    print(frame.head(10).to_string(index=False))
    print(f"\nrows={len(frame):,}  skus={frame['sku_id'].nunique()}  "
          f"dates={frame['date'].nunique()}")
    print(frame.groupby("on_promo")["units"].mean())
