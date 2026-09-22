"""End-to-end recommender run: generate, split, fit, rank, report.

    python -m src.train

Every model is scored on the same leave-last-out split, ranking over the full
catalogue with each user's training items masked out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CatalogConfig, generate_interactions, interaction_summary
from .evaluate import evaluate_recommender, metrics_by_user_activity
from .model import build_models
from .split import assert_no_test_item_in_train, drop_repeat_consumption, leave_last_out


def run(
    cfg: CatalogConfig,
    output_dir: Path,
    k: int = 10,
    n_holdout: int = 2,
    collapse_repeats: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> dict:
    print(f"generating {cfg.n_users:,} users x {cfg.n_items:,} items ...")
    interactions, items = generate_interactions(cfg)
    stats = interaction_summary(interactions, items)
    print(
        f"  {stats['interactions']:,} plays, density {stats['density']:.2%}, "
        f"top 1% of items hold {stats['share_of_plays_in_top_1pct_items']:.1%} of plays"
    )

    if collapse_repeats:
        interactions = drop_repeat_consumption(interactions)
        print(f"  collapsed repeats -> {len(interactions):,} distinct (user, item) pairs")

    dataset = leave_last_out(interactions, cfg.n_items, n_holdout=n_holdout)
    assert_no_test_item_in_train(dataset)
    ground_truth = dataset.ground_truth()
    test_users = dataset.test_users
    print(
        f"  train {len(dataset.train_df):,} | test {len(dataset.test_df):,} "
        f"over {len(test_users):,} users"
    )

    binary = dataset.train_matrix.copy()
    binary.data = np.ones_like(binary.data)
    train_popularity = np.asarray(binary.sum(axis=0)).ravel()
    train_counts = np.asarray(binary.sum(axis=1)).ravel()

    results: dict[str, dict] = {}
    for name, model in build_models(seed=cfg.seed).items():
        started = time.time()
        model.fit(dataset.train_matrix)
        fit_seconds = time.time() - started

        recommendations = model.recommend(test_users, k=k, train_matrix=dataset.train_matrix)
        metrics = evaluate_recommender(
            recommendations, test_users, ground_truth, cfg.n_items, train_popularity, k=k
        )
        by_activity = metrics_by_user_activity(
            recommendations, test_users, ground_truth, train_counts, k=k
        )

        results[name] = {
            **metrics,
            "fit_seconds": round(fit_seconds, 2),
            "by_user_activity": by_activity.to_dict(orient="records"),
        }
        if verbose:
            print(
                f"  {name:12s} recall@{k} {metrics['recall']:.4f} | "
                f"NDCG@{k} {metrics['ndcg']:.4f} | coverage {metrics['catalogue_coverage']:.3f} | "
                f"novelty {metrics['novelty']:.2f} | {fit_seconds:.1f}s"
            )

    _print_table(results, k)

    baseline = results["popularity"]["recall"]
    best = max(results, key=lambda name: results[name]["recall"])
    lift = (results[best]["recall"] - baseline) / baseline if baseline else float("nan")
    print(f"\nbest by recall@{k}: {best} ({lift:+.1%} against the popularity baseline)")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "n_users": cfg.n_users,
            "n_items": cfg.n_items,
            "exposure_strength": cfg.exposure_strength,
            "taste_sharpness": cfg.taste_sharpness,
            "popularity_alpha": cfg.popularity_alpha,
            "seed": cfg.seed,
        },
        "dataset": stats,
        "k": k,
        "n_holdout": n_holdout,
        "collapse_repeats": collapse_repeats,
        "test_users": int(len(test_users)),
        "models": results,
        "best_by_recall": best,
        "lift_over_popularity": lift,
    }
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")

    if make_plots:
        _write_plots(results, k, output_dir)

    return summary


def _print_table(results: dict, k: int) -> None:
    frame = pd.DataFrame(
        [
            {
                "model": name,
                f"recall@{k}": r["recall"],
                f"ndcg@{k}": r["ndcg"],
                f"map@{k}": r["map"],
                "hit_rate": r["hit_rate"],
                "coverage": r["catalogue_coverage"],
                "novelty": r["novelty"],
                "gini": r["gini"],
                "pop_pct": r["mean_popularity_percentile"],
            }
            for name, r in results.items()
        ]
    ).sort_values(f"recall@{k}", ascending=False)

    print("\naccuracy and distribution together (a model can win the left half "
          "and be useless):\n")
    formatted = frame.copy()
    for column in formatted.columns[1:]:
        formatted[column] = formatted[column].map(lambda v: f"{v:.4f}")
    print(formatted.to_string(index=False))


def _write_plots(results: dict, k: int, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plots are optional
        print("matplotlib not installed; skipping plots")
        return

    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    axes[0].bar(names, [results[n]["recall"] for n in names], color="steelblue")
    axes[0].axhline(
        results["popularity"]["recall"], color="crimson", linestyle="--", linewidth=1,
        label="popularity baseline",
    )
    axes[0].set(ylabel=f"recall@{k}", title="Accuracy")
    axes[0].tick_params(axis="x", rotation=15, labelsize=8)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")

    # The plot that matters: accuracy against how much of the catalogue is used.
    for name in names:
        axes[1].scatter(
            results[name]["catalogue_coverage"], results[name]["recall"], s=70
        )
        axes[1].annotate(
            name,
            (results[name]["catalogue_coverage"], results[name]["recall"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8,
        )
    axes[1].set(
        xlabel="catalogue coverage",
        ylabel=f"recall@{k}",
        title="Accuracy vs how much catalogue is used",
    )
    axes[1].grid(alpha=0.25)

    for name in names:
        by_activity = pd.DataFrame(results[name]["by_user_activity"])
        axes[2].plot(
            by_activity["activity_bucket"], by_activity["recall"], marker="o",
            label=name, linewidth=1.5,
        )
    axes[2].set(
        xlabel="user activity quartile (0 = coldest)",
        ylabel=f"recall@{k}",
        title="Where each model earns its score",
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "evaluation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Defaults are read off the config object rather than repeated here. Typing
    # them twice let the CLI drift from the library once already: the dataclass
    # said exposure_strength=0.5 and argparse said 0.7, so `python -m src.train`
    # and `run(CatalogConfig(), ...)` quietly evaluated different datasets and
    # disagreed by 2x on the headline metric.
    defaults = CatalogConfig()

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-users", type=int, default=defaults.n_users)
    p.add_argument("--n-items", type=int, default=defaults.n_items)
    p.add_argument("--k", type=int, default=10, help="recommendation list length")
    p.add_argument("--n-holdout", type=int, default=2, help="items held out per user")
    p.add_argument(
        "--exposure-strength",
        type=float,
        default=defaults.exposure_strength,
        help="how much past exposure, rather than taste, drove the observed plays",
    )
    p.add_argument(
        "--taste-sharpness",
        type=float,
        default=defaults.taste_sharpness,
        help="how sharply taste discriminates between items; low values make "
             "the popularity baseline unbeatable",
    )
    p.add_argument(
        "--keep-repeats",
        action="store_true",
        help="keep repeat plays instead of collapsing them to one row per item",
    )
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = CatalogConfig(
        n_users=args.n_users,
        n_items=args.n_items,
        exposure_strength=args.exposure_strength,
        taste_sharpness=args.taste_sharpness,
        seed=args.seed,
    )
    run(
        cfg=cfg,
        output_dir=args.output_dir,
        k=args.k,
        n_holdout=args.n_holdout,
        collapse_repeats=not args.keep_repeats,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
