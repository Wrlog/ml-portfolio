"""End-to-end forecasting run: generate, engineer, backtest, report.

    python -m src.train

Every model in the ladder is re-fit at each of six monthly forecast origins and
scored on the 28 days that follow. Nothing is scored on data any model was
fitted on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import aggregate, backtest
from .data import DemandConfig, generate_sales
from .evaluate import (
    forecast_metrics,
    forecast_value_add,
    metrics_by_group,
    metrics_by_horizon_day,
)
from .features import build_features
from .model import build_models


def run(
    cfg: DemandConfig,
    horizon: int,
    n_folds: int,
    output_dir: Path,
    baseline: str = "seasonal_naive",
    make_plots: bool = True,
    verbose: bool = True,
) -> dict:
    print(f"generating {cfg.n_skus} SKUs x {cfg.days} days ...")
    panel = generate_sales(cfg)
    print(f"  {len(panel):,} rows, {panel['units'].sum():,} units sold")

    print(f"building features for a {horizon}-day horizon ...")
    features = build_features(panel, horizon)

    print(f"backtesting {n_folds} rolling origins ...")
    models = build_models(horizon, random_state=cfg.seed)
    predictions = backtest(features, horizon, models, n_folds=n_folds, verbose=verbose)

    summary_table = aggregate(predictions, forecast_metrics)
    baseline_wape = float(
        summary_table.loc[summary_table["model"] == baseline, "wape"].iloc[0]
    )
    summary_table["fva_vs_" + baseline] = summary_table["wape"].map(
        lambda w: forecast_value_add(w, baseline_wape)
    )

    print("\naccuracy across all folds (lower WAPE is better):\n")
    display = summary_table.copy()
    for col in ("wape", "bias", "smape", "wape_fold_std", "wape_worst_fold",
                f"fva_vs_{baseline}"):
        display[col] = display[col].map(lambda v: f"{v:+.3f}" if "bias" in col
                                        or "fva" in col else f"{v:.3f}")
    display["mae"] = display["mae"].map(lambda v: f"{v:.2f}")
    display["rmse"] = display["rmse"].map(lambda v: f"{v:.2f}")
    print(display.to_string(index=False))

    best = summary_table.iloc[0]["model"]
    print(f"\nbest model by WAPE: {best}")

    best_preds = predictions[predictions["model"] == best]
    by_horizon = metrics_by_horizon_day(best_preds)
    by_category = metrics_by_group(best_preds, "category")

    print("\nerror by distance from the forecast origin:")
    weekly_error = (
        by_horizon.assign(week=((by_horizon["days_ahead"] - 1) // 7 + 1))
        .groupby("week", observed=True)["wape"]
        .mean()
        .round(3)
    )
    for week, value in weekly_error.items():
        print(f"  days {(week - 1) * 7 + 1:>2}-{week * 7:>2} ahead: WAPE {value:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "n_skus": cfg.n_skus,
            "days": cfg.days,
            "price_elasticity": cfg.price_elasticity,
            "promo_rate": cfg.promo_rate,
            "stockout_rate": cfg.stockout_rate,
            "seed": cfg.seed,
        },
        "horizon_days": horizon,
        "n_folds": int(predictions["fold"].nunique()),
        "baseline": baseline,
        "models": summary_table.to_dict(orient="records"),
        "best_model": best,
        "best_model_by_horizon_week": by_horizon.to_dict(orient="records"),
        "best_model_by_category": by_category.to_dict(orient="records"),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {metrics_path}")

    predictions.to_parquet(output_dir / "predictions.parquet", index=False) if _has_parquet() \
        else predictions.to_csv(output_dir / "predictions.csv", index=False)

    if make_plots:
        _write_plots(predictions, summary_table, best, output_dir)

    return summary


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _write_plots(predictions, summary_table, best, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plots are optional
        print("matplotlib not installed; skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    axes[0].barh(summary_table["model"], summary_table["wape"], color="steelblue")
    axes[0].invert_yaxis()
    axes[0].set(xlabel="WAPE (lower is better)", title="Accuracy across all folds")
    axes[0].grid(alpha=0.25, axis="x")

    for name, chunk in predictions.groupby("model", observed=True):
        by_day = metrics_by_horizon_day(chunk)
        axes[1].plot(by_day["days_ahead"], by_day["wape"], label=name, linewidth=1.5)
    axes[1].set(
        xlabel="days ahead of forecast origin",
        ylabel="WAPE",
        title="Error decay over the horizon",
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)

    # One SKU's actuals against the best model, in the most recent fold.
    best_preds = predictions[predictions["model"] == best]
    last_fold = best_preds[best_preds["fold"] == best_preds["fold"].max()]
    busiest = last_fold.groupby("sku_id")["units"].sum().idxmax()
    trace = last_fold[last_fold["sku_id"] == busiest].sort_values("date")
    axes[2].plot(trace["date"], trace["units"], label="actual", linewidth=1.6)
    axes[2].plot(trace["date"], trace["forecast"], label=best, linewidth=1.6, linestyle="--")
    axes[2].set(title=f"{busiest}, most recent fold", ylabel="units")
    axes[2].tick_params(axis="x", rotation=30, labelsize=7)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "backtest.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horizon", type=int, default=28, help="forecast horizon in days")
    p.add_argument("--n-folds", type=int, default=6, help="rolling forecast origins")
    p.add_argument("--n-skus", type=int, default=60)
    p.add_argument("--days", type=int, default=1_095)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = DemandConfig(n_skus=args.n_skus, days=args.days, seed=args.seed)
    run(
        cfg=cfg,
        horizon=args.horizon,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
