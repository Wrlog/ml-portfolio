"""End-to-end training run: generate, split, fit the ladder, price the decision.

Run with defaults:

    python -m src.train

The split is strictly temporal -- train on the first 60% of the calendar, tune
the threshold on the next 20%, report on the last 20%. A random split would be
wrong twice over here: it scatters the transactions of a single fraud burst
across train and test (so the model memorises the attack rather than
generalising to the next one), and it lets the model learn from the future,
which it will not have at scoring time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import GeneratorConfig, generate_transactions
from .evaluate import (
    CostModel,
    choose_threshold,
    cost_curve,
    operating_point,
    precision_at_budget,
    ranking_metrics,
    recall_at_precision,
)
from .features import build_features
from .model import build_models, positive_class_weight


@dataclass
class Split:
    """A temporal train / validation / test partition."""

    X_train: pd.DataFrame
    y_train: pd.Series
    amt_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    amt_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    amt_test: pd.Series
    boundaries: tuple[pd.Timestamp, pd.Timestamp]


def temporal_split(
    df: pd.DataFrame, X: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2
) -> Split:
    if not 0 < train_frac < 1 or not 0 < valid_frac < 1 - train_frac:
        raise ValueError("train_frac and valid_frac must leave a non-empty test set")

    ts = pd.to_datetime(df["timestamp"])
    cut_train = ts.quantile(train_frac)
    cut_valid = ts.quantile(train_frac + valid_frac)

    is_train = ts <= cut_train
    is_valid = (ts > cut_train) & (ts <= cut_valid)
    is_test = ts > cut_valid

    y = df["is_fraud"].astype(int)
    amt = df["amount"].astype(float)
    return Split(
        X_train=X[is_train],
        y_train=y[is_train],
        amt_train=amt[is_train],
        X_valid=X[is_valid],
        y_valid=y[is_valid],
        amt_valid=amt[is_valid],
        X_test=X[is_test],
        y_test=y[is_test],
        amt_test=amt[is_test],
        boundaries=(cut_train, cut_valid),
    )


def _score(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def run(
    cfg: GeneratorConfig,
    costs: CostModel,
    output_dir: Path,
    max_alert_rate: float = 0.01,
    review_budget_per_day: int = 15,
    make_plots: bool = True,
) -> dict:
    print(f"generating {cfg.n_legit:,} legitimate transactions ...")
    df = generate_transactions(cfg)
    print(f"  {len(df):,} rows, {df['is_fraud'].sum():,} fraudulent "
          f"({df['is_fraud'].mean():.3%})")

    # Features are built on the full log *before* splitting. That is safe
    # precisely because every feature is causal (see features.py): a
    # transaction only ever looks backwards, so building them across the
    # boundary reproduces what the scorer sees in production, where an
    # account's history does not reset at midnight on the split date.
    print("building causal features ...")
    X = build_features(df)
    split = temporal_split(df, X)
    print(
        f"  train {len(split.X_train):,} | valid {len(split.X_valid):,} "
        f"| test {len(split.X_test):,}"
    )
    print(f"  boundaries: {split.boundaries[0]:%Y-%m-%d} / {split.boundaries[1]:%Y-%m-%d}")

    n_test_days = max(
        1.0,
        (pd.to_datetime(df["timestamp"]).max() - split.boundaries[1]).total_seconds()
        / 86_400,
    )

    models = build_models(
        random_state=cfg.seed, pos_weight=positive_class_weight(split.y_train)
    )
    results: dict[str, dict] = {}
    scores_test: dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"\nfitting {name} ...")
        model.fit(split.X_train, split.y_train)

        valid_scores = _score(model, split.X_valid)
        test_scores = _score(model, split.X_test)
        scores_test[name] = test_scores

        # The threshold is chosen on validation and then frozen. Picking it on
        # the test set is the quiet version of training on the test set.
        threshold = choose_threshold(
            split.y_valid, valid_scores, split.amt_valid, costs, max_alert_rate
        )
        point = operating_point(
            split.y_test, test_scores, split.amt_test, threshold, costs
        )
        budget = precision_at_budget(
            split.y_test, test_scores, int(review_budget_per_day * n_test_days)
        )

        results[name] = {
            "ranking": ranking_metrics(split.y_test, test_scores),
            "operating_point": point.to_dict(),
            "precision_at_review_budget": budget,
            "recall_at_precision_10pct": recall_at_precision(
                split.y_test, test_scores, 0.10
            ),
            "threshold_source": "validation split, cost-minimising within capacity",
        }
        r = results[name]["ranking"]
        print(
            f"  PR-AUC {r['pr_auc']:.3f} ({r['pr_auc_lift_over_random']:.0f}x base rate)"
            f" | ROC-AUC {r['roc_auc']:.3f}"
        )
        print(
            f"  @threshold {point.threshold:.4f}: precision {point.precision:.1%},"
            f" recall {point.recall:.1%}, {point.alert_rate:.2%} of traffic alerted"
        )
        print(
            f"  net savings ${point.net_savings:,.0f} "
            f"({point.savings_rate:.1%} of the do-nothing loss)"
        )

    best = max(results, key=lambda k: results[k]["ranking"]["pr_auc"])
    print(f"\nbest model by PR-AUC: {best}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "n_legit": cfg.n_legit,
            "n_accounts": cfg.n_accounts,
            "fraud_rate": cfg.fraud_rate,
            "separability": cfg.separability,
            "seed": cfg.seed,
        },
        "cost_model": costs.__dict__,
        "capacity": {
            "max_alert_rate": max_alert_rate,
            "review_budget_per_day": review_budget_per_day,
            "test_days": round(n_test_days, 1),
        },
        "test_set": {
            "rows": int(len(split.y_test)),
            "frauds": int(split.y_test.sum()),
            "do_nothing_cost": costs.baseline_cost(split.y_test, split.amt_test),
        },
        "models": results,
        "best_model": best,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {metrics_path}")

    if make_plots:
        _write_plots(split, scores_test, costs, best, output_dir)

    return summary


def _write_plots(split, scores_test, costs, best, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except ImportError:  # pragma: no cover - plots are optional
        print("matplotlib not installed; skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, scores in scores_test.items():
        precision, recall, _ = precision_recall_curve(split.y_test, scores)
        axes[0].plot(recall, precision, label=name, linewidth=1.6)
    axes[0].axhline(
        split.y_test.mean(), color="grey", linestyle="--", linewidth=1, label="base rate"
    )
    axes[0].set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    curve = cost_curve(split.y_test, scores_test[best], split.amt_test, costs)
    curve = curve[curve["alert_rate"] <= 0.05]
    axes[1].plot(curve["alert_rate"] * 100, curve["total_cost"], linewidth=1.6)
    baseline = costs.baseline_cost(split.y_test, split.amt_test)
    axes[1].axhline(
        baseline, color="crimson", linestyle="--", linewidth=1, label="approve everything"
    )
    low = curve.loc[curve["total_cost"].idxmin()]
    axes[1].scatter([low["alert_rate"] * 100], [low["total_cost"]], color="black", zorder=5)
    axes[1].annotate(
        f"min cost @ {low['alert_rate']:.2%} alerts",
        (low["alert_rate"] * 100, low["total_cost"]),
        textcoords="offset points",
        xytext=(10, 14),
        fontsize=8,
    )
    axes[1].set(
        xlabel="share of traffic alerted (%)",
        ylabel="total cost ($)",
        title=f"Cost curve -- {best} (test)",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "evaluation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-legit", type=int, default=240_000, help="legitimate transactions")
    p.add_argument("--n-accounts", type=int, default=12_000)
    p.add_argument("--fraud-rate", type=float, default=0.0017)
    p.add_argument(
        "--separability",
        type=float,
        default=0.62,
        help="how distinguishable attacks are from normal spend, in [0, 1]",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--max-alert-rate",
        type=float,
        default=0.01,
        help="review capacity as a share of transactions",
    )
    p.add_argument(
        "--review-budget-per-day",
        type=int,
        default=15,
        help="cases a human team can work per day, used for precision@budget",
    )
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = GeneratorConfig(
        n_accounts=args.n_accounts,
        n_legit=args.n_legit,
        fraud_rate=args.fraud_rate,
        separability=args.separability,
        seed=args.seed,
    )
    run(
        cfg=cfg,
        costs=CostModel(),
        output_dir=args.output_dir,
        max_alert_rate=args.max_alert_rate,
        review_budget_per_day=args.review_budget_per_day,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
