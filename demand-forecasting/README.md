# 28-day SKU demand forecasting

Daily unit forecasts for a 60-SKU grocery catalogue, four weeks out, evaluated
with a rolling-origin backtest against the baselines a planning team already
runs.

The data is synthetic. `src/data.py` builds daily demand from named
components: base level, trend, weekly and annual seasonality, price
elasticity, promotions, holidays and stockouts. The numbers are reproducible
from the code but say nothing about a real retailer.

## The problem

A replenishment team places purchase orders about a month before the stock is
needed, so what matters is the forecast for each of the next 28 days, made
today. Most of the difficulty is in that lead time, and most forecasting
projects ignore it.

## Lags and the horizon

Yesterday's sales, `units_lag_1`, is the strongest single feature for a sales
forecaster. At a 28-day horizon you can't have it: when the forecast is made,
the day before the target date hasn't happened yet. Build it anyway and
cross-validation looks excellent, then in production the column is full of
nulls or stale values and the model loses to a baseline that costs nothing.

So `build_features` takes the horizon as an argument and derives every lag
from it. There's no way to get `lag_1` at a 28-day horizon:

```python
for offset in LAG_OFFSETS:          # [0, 1, 2, 7, 14, 28]
    lag = horizon + offset          # never less than the horizon
    df[f"units_lag_{lag}"] = grp.shift(lag)

shifted = grp.shift(horizon)        # rolling windows are computed on the
...                                 # already-shifted series
```

`tests/test_features.py` checks this directly rather than trusting the names,
including that the 7-day rolling mean at horizon 28 covers days
*t−34 … t−28* and not *t−6 … t*.

Not everything needs lagging, though. Lagging all covariates the same way
throws away most of the promotional signal:

| Kind | Examples | Available at forecast time? |
|---|---|---|
| Past covariates | sales history | No, must be lagged by ≥ horizon |
| Future covariates | price, promotion flag, calendar, holidays | Yes, a human planned them weeks ago |

Price and promo go in unlagged, at their own target date. That's the
merchandising calendar, a real input a forecaster gets, so it isn't leakage.

## Results

60 SKUs × 1,095 days. Six monthly forecast origins from 2023-07-16 to
2023-12-03; at each one the model is re-fit on everything up to that date and
scored on the next 28 days. Reproduce with `python -m src.train`.

| Model | WAPE | Bias | MAE | RMSE | Fold σ | Worst fold | FVA vs seasonal naive |
|---|---|---|---|---|---|---|---|
| **LightGBM (Tweedie)** | **0.387** | −0.021 | 11.91 | 22.06 | 0.011 | 0.404 | **+36.1%** |
| LightGBM (L2) | 0.397 | +0.019 | 12.23 | 22.10 | 0.015 | 0.421 | +34.4% |
| Moving average, 28d | 0.476 | −0.047 | 14.64 | 27.88 | 0.021 | 0.511 | +21.5% |
| Same day last year | 0.580 | −0.113 | 17.84 | 32.74 | 0.020 | 0.614 | +4.3% |
| Seasonal naive | 0.606 | −0.031 | 18.64 | 34.38 | 0.017 | 0.623 | — |

The model removes 36% of the seasonal-naive error in every one of the six
folds. Its worst fold (0.404) beats the best baseline fold, which a single
train/test split couldn't have shown.

WAPE is used instead of MAPE. MAPE divides by the actual, so a SKU selling 2
units a day outweighs one selling 400, and it's undefined on zero-sales days.
This catalogue has both. WAPE (total absolute error over total actual units)
works at zero, reads easily ("we are off by 39% of volume"), and weights SKUs
by volume, as an inventory budget does.

Bias is reported separately. A forecast that's 20% low every day and one that
swings ±20% have the same WAPE, but the first empties the shelf and the
second just churns the warehouse. It matters for the choice here: Tweedie
wins on WAPE but under-forecasts (−2.1%), while L2 over-forecasts (+1.9%). A
consistently low forecast means stockouts, so the 1-point WAPE win isn't
automatically the better pick. That's for whoever owns the service level to
decide, and the code reports both numbers.

### Where it does worst

| Slice | WAPE | Bias |
|---|---|---|
| 20 fastest-moving SKUs (210k units) | 0.372 | −4.1% |
| 20 slowest-moving SKUs (21k units) | 0.476 | **+4.9%** |

The aggregate hides this. Slow movers are harder and biased the other way;
over-forecasting them is how inventory money gets stuck in the long tail.

### Error by horizon

| Days ahead | WAPE |
|---|---|
| 1–7 | 0.387 |
| 8–14 | 0.376 |
| 15–21 | 0.389 |
| 22–28 | 0.400 |

One model is fit with every feature lagged by the full 28 days, so day 1 and
day 28 see exactly the same information. A flat curve is expected, and the
slight rise is the world drifting from the origin.

A separate model per horizon step would let day 1 use `lag_1` and be more
accurate at the near end, at the cost of 28 models to train, monitor and
retrain. Replenishment commits the whole 28-day window at once, so that
near-end accuracy wouldn't change the order. I chose the single model for
that reason.

## Stockouts

A zero on an out-of-stock day is unobserved demand, not zero demand. Training
on those rows teaches the model to forecast the warehouse's failures, so
`training_frame` drops them:

```python
X, y = training_frame(df, horizon, drop_censored=True)
```

They stay in the test set, since a forecast is judged on what it was asked to
predict. Real data usually has no stock flag, which makes this a
data-collection problem before it's a modelling one. Good to know before
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

Outputs go to `artifacts/`: `metrics.json`, per-row `predictions.csv`, and
`backtest.png` (accuracy by model, error over the horizon, and one SKU's
actuals against the forecast).

## Limitations

- No prediction intervals. Order quantity depends on something like the 90th
  percentile of demand, not the mean. LightGBM quantile regression at a few
  levels is the obvious next step and the biggest gap here.
- No hierarchical reconciliation. SKU forecasts summed to category level won't
  match a direct category forecast; real planning systems reconcile them.
- No cold start. A new SKU gets NaN lags. LightGBM tolerates missing values,
  but the model has nothing better than the category prior, and there's no
  explicit handling.
- Simple promotions. The generator has no cannibalisation between SKUs and no
  pull-forward. Real promotions steal demand from later weeks and competing
  products, which is where forecast error concentrates in practice.
