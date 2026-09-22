# 28-day SKU demand forecasting

Daily unit forecasts for a 60-SKU grocery catalogue, four weeks out, evaluated
by rolling-origin backtest against the baselines a planning team is already
running.

> **The data is synthetic.** `src/data.py` simulates daily demand from named
> components — base level, trend, weekly and annual seasonality, price
> elasticity, promotions, holidays, stockouts. Every number below is
> reproducible by running the code; none of it is evidence about a real
> retailer.

## The problem

A replenishment team commits purchase orders about a month before the stock is
needed. So the forecast that matters is not "what will we sell tomorrow" — it
is **"what will we sell on each of the next 28 days, decided today."**

That lead time is the entire difficulty, and it is what most forecasting
projects quietly ignore.

## The bug this project is built to avoid

`units_lag_1` — yesterday's sales — is the strongest single feature you can
build for a sales forecaster. At a 28-day horizon it is **unusable**, because
on the day the forecast is produced, the day before the target date has not
happened yet.

Build it anyway and cross-validation looks excellent. Ship it, and the column
arrives full of nulls or stale values, and the model loses to a baseline that
costs nothing to run.

So `build_features` takes the horizon as an argument and derives every lag from
it. There is no way to ask this module for `lag_1` at a 28-day horizon:

```python
for offset in LAG_OFFSETS:          # [0, 1, 2, 7, 14, 28]
    lag = horizon + offset          # never less than the horizon
    df[f"units_lag_{lag}"] = grp.shift(lag)

shifted = grp.shift(horizon)        # rolling windows are computed on the
...                                 # already-shifted series
```

`tests/test_features.py` asserts this mechanically rather than trusting the
naming convention, including a check that the 7-day rolling mean at horizon 28
covers days *t−34 … t−28* and not *t−6 … t*.

### Past covariates vs future covariates

Not everything needs lagging, and treating all covariates the same throws away
most of the promotional signal:

| Kind | Examples | Available at forecast time? |
|---|---|---|
| **Past covariates** | sales history | No — must be lagged by ≥ horizon |
| **Future covariates** | price, promotion flag, calendar, holidays | **Yes** — a human planned them weeks ago |

Price and promo enter unlagged, at their own target date. That is not leakage;
it is the merchandising calendar, which is a real input a real forecaster gets.

## Results

60 SKUs × 1,095 days. Six monthly forecast origins from 2023-07-16 to
2023-12-03; at each one the model is re-fit on everything up to that date and
scored on the following 28 days. Reproduce with `python -m src.train`.

| Model | WAPE | Bias | MAE | RMSE | Fold σ | Worst fold | FVA vs seasonal naive |
|---|---|---|---|---|---|---|---|
| **LightGBM (Tweedie)** | **0.387** | −0.021 | 11.91 | 22.06 | 0.011 | 0.404 | **+36.1%** |
| LightGBM (L2) | 0.397 | +0.019 | 12.23 | 22.10 | 0.015 | 0.421 | +34.4% |
| Moving average, 28d | 0.476 | −0.047 | 14.64 | 27.88 | 0.021 | 0.511 | +21.5% |
| Same day last year | 0.580 | −0.113 | 17.84 | 32.74 | 0.020 | 0.614 | +4.3% |
| Seasonal naive | 0.606 | −0.031 | 18.64 | 34.38 | 0.017 | 0.623 | — |

The learned model removes **36% of the seasonal-naive error**, and does so in
every one of the six folds — the worst fold (0.404) is still better than the
best baseline fold. A single train/test split would have told you none of that.

### Why WAPE and not MAPE

MAPE is what stakeholders ask for and it is the wrong metric on a mixed
catalogue. It divides by the actual, so a SKU selling 2 units a day outweighs
one selling 400, and on a zero-sales day it is undefined. This catalogue has
both. WAPE — total absolute error over total actual units — is well defined at
zero, is interpretable ("we are off by 39% of volume"), and weights each SKU by
the volume it actually moves, which is what an inventory budget does too.

### Bias is reported separately on purpose

A forecast that is 20% low every day and one that alternates ±20% have
identical WAPE and completely different consequences: the first empties the
shelf, the second just churns the warehouse.

This matters for the model choice. Tweedie wins on WAPE, but it under-forecasts
(−2.1%) while L2 over-forecasts (+1.9%). For a replenishment system, a
systematically low forecast turns into stockouts, so the 1-point WAPE win is
not automatically the right trade — that call belongs to whoever owns the
service level, and the code reports both numbers so they can make it.

### Where it fails

| Slice | WAPE | Bias |
|---|---|---|
| 20 fastest-moving SKUs (210k units) | 0.372 | −4.1% |
| 20 slowest-moving SKUs (21k units) | 0.476 | **+4.9%** |

The aggregate hides this. Slow movers are both harder *and* biased the other
way — the model over-forecasts them, which is how inventory money gets stuck in
the long tail of a catalogue.

### The error curve is flat, and that is expected

| Days ahead | WAPE |
|---|---|
| 1–7 | 0.387 |
| 8–14 | 0.376 |
| 15–21 | 0.389 |
| 22–28 | 0.400 |

One model is fit with every feature lagged by the full 28 days, so day 1 and
day 28 are predicted from *exactly the same information*. A flat curve is the
correct outcome; the slight upward drift is the world moving away from the
origin.

The alternative — a separate model per horizon step, so day 1 may use `lag_1` —
would be meaningfully more accurate at the near end. It also means 28 models to
train, monitor and retrain. For a replenishment decision that commits the whole
28-day window at once, the near-end accuracy has nowhere to go, so this project
takes the single-model design deliberately rather than by default.

## Stockouts censor the label

A zero on a day the SKU was out of stock is not demand of zero — it is
**unobserved** demand. Training on those rows teaches the model to forecast the
warehouse's failures instead of the customer's intent, so `training_frame`
drops them:

```python
X, y = training_frame(df, horizon, drop_censored=True)
```

They stay in the *test* set, because a forecast is judged on what it was asked
to predict. In real data the stock flag usually does not exist, which makes
this a data-collection fight rather than a modelling one — worth knowing before
promising an accuracy number.

## Running it

```bash
pip install -r ../requirements.txt

python -m src.train                          # full backtest, ~2m10s on a laptop CPU
python -m src.train --n-skus 15 --n-folds 2  # quick version, ~50s
python -m pytest tests -q                    # 35 tests, ~100s
```

| Flag | Default | Effect |
|---|---|---|
| `--horizon` | 28 | Forecast horizon in days. Every lag rederives from it. |
| `--n-folds` | 6 | Rolling forecast origins. |
| `--n-skus` / `--days` | 60 / 1095 | Catalogue and history size. |

Outputs land in `artifacts/`: `metrics.json`, per-row `predictions.csv`, and
`backtest.png` (accuracy by model, error decay over the horizon, and one SKU's
actuals against the forecast).

## What this does not do

- **No prediction intervals.** Replenishment needs a service level, not a point
  forecast — the order quantity depends on the 90th percentile of demand, not
  the mean. LightGBM quantile regression at a few levels is the natural next
  step and is the single biggest gap here.
- **No hierarchical reconciliation.** SKU forecasts summed to category level
  will not match a forecast made directly at category level. Real planning
  systems reconcile the two.
- **No cold start.** A new SKU has no history and gets NaN lags. The model
  survives it (LightGBM handles missing values) but it has nothing better than
  the category prior, and there is no explicit handling.
- **Promotions are simple here.** The generator has no cannibalisation between
  SKUs and no pull-forward — a real promotion steals demand from the following
  weeks and from competing products, and that is where forecast error
  concentrates in practice.
