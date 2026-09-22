"""Train duplicate detection under both splitting protocols and compare.

    python -m src.train

The run does the same work twice -- once with a naive random split of the pair
table, once with a cluster-disjoint split -- and reports both. The difference
between the two numbers is the result this project exists to show: the first
protocol reports a model that is better than the model actually is.

Within each protocol the pipeline is honest end to end: the TF-IDF vectorisers
are fit on training questions only, and the decision threshold is chosen on a
validation slice carved out of training, never on the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PairConfig, generate_pairs
from .evaluate import (
    best_f1_threshold,
    calibration_table,
    classification_metrics,
    metrics_by_pair_kind,
    worst_errors,
)
from .features import PairFeaturizer, QuestionTargetEncoder
from .model import build_models
from .split import (
    SplitResult,
    assert_no_question_overlap,
    cluster_disjoint_split,
    connected_components,
    random_pair_split,
    split_report,
)


def evaluate_protocol(
    df: pd.DataFrame, split: SplitResult, seed: int, verbose: bool = True
) -> dict:
    """Fit every model under one splitting protocol and score it on that split."""
    if verbose:
        print(f"\n{split.summary(df)}")

    train_df = df.loc[split.train].reset_index(drop=True)
    test_df = df.loc[split.test].reset_index(drop=True)

    # A validation slice for threshold selection only. Carved from training, so
    # under the disjoint protocol it inherits the disjointness.
    rng = np.random.default_rng(seed)
    is_valid = rng.random(len(train_df)) < 0.2
    fit_df = train_df.loc[~is_valid].reset_index(drop=True)
    valid_df = train_df.loc[is_valid].reset_index(drop=True)

    # Fit on the fitting slice only. Fitting on the whole corpus would leak
    # test-set vocabulary and document frequencies into the features.
    featurizer = PairFeaturizer().fit(fit_df)
    pairwise = {
        "fit": featurizer.transform(fit_df),
        "valid": featurizer.transform(valid_df),
        "test": featurizer.transform(test_df),
    }

    # Identity-keyed statistics, also fitted on the fitting slice only. Whether
    # this block helps is entirely a property of the split, not of the code.
    encoder = QuestionTargetEncoder().fit(fit_df)
    unseen = encoder.unseen_share(test_df)
    if verbose:
        print(f"  questions in test never seen during fitting: {unseen:.1%}")

    with_stats = {
        key: pd.concat([pairwise[key], encoder.transform(frame)], axis=1)
        for key, frame in (("fit", fit_df), ("valid", valid_df), ("test", test_df))
    }

    results: dict[str, dict] = {}
    for name, spec in build_models(random_state=seed).items():
        blocks = with_stats if spec.uses_question_stats else pairwise
        model = spec.estimator
        model.fit(blocks["fit"], fit_df["is_duplicate"])

        valid_scores = model.predict_proba(blocks["valid"])[:, 1]
        threshold = best_f1_threshold(valid_df["is_duplicate"], valid_scores)

        test_scores = model.predict_proba(blocks["test"])[:, 1]
        metrics = classification_metrics(
            test_df["is_duplicate"], test_scores, threshold=threshold
        )
        by_kind = metrics_by_pair_kind(test_df, test_scores, threshold)

        results[name] = {
            "overall": metrics,
            "uses_question_stats": spec.uses_question_stats,
            "by_negative_type": by_kind.to_dict(orient="records"),
            "calibration": calibration_table(
                test_df["is_duplicate"], test_scores
            ).to_dict(orient="records"),
        }
        if verbose:
            hard = by_kind[by_kind["negatives"] == "hard_negative"]
            hard_auc = float(hard["roc_auc"].iloc[0]) if len(hard) else float("nan")
            print(
                f"  {name:24s} ROC-AUC {metrics['roc_auc']:.3f} | "
                f"PR-AUC {metrics['pr_auc']:.3f} | acc {metrics['accuracy']:.3f} | "
                f"hard-negative ROC-AUC {hard_auc:.3f}"
            )

    return {
        "split": split_report(df, split),
        "unseen_test_question_share": unseen,
        "models": results,
        "_test_df": test_df,
        "_featurizer": featurizer,
    }


def run(cfg: PairConfig, output_dir: Path, test_size: float = 0.25,
        make_plots: bool = True, verbose: bool = True) -> dict:
    print(f"generating {cfg.n_pairs:,} question pairs over {cfg.n_intents} intents ...")
    df = generate_pairs(cfg)
    print(
        f"  {df['is_duplicate'].mean():.1%} duplicates, "
        f"{(df['pair_kind'] == 'hard_negative').mean():.1%} hard negatives"
    )

    components = connected_components(df)
    print(
        f"  cluster graph: {components.nunique()} connected component(s); "
        f"largest covers {components.value_counts().iloc[0] / len(df):.1%} of pairs"
    )
    if components.nunique() == 1:
        print("  -> no partition preserves every pair, so the disjoint split drops "
              "boundary-crossing ones")

    protocols = {
        "random_pair_split": random_pair_split(df, test_size, seed=cfg.seed),
        "cluster_disjoint_split": cluster_disjoint_split(df, test_size, seed=cfg.seed),
    }

    outputs = {}
    for key, split in protocols.items():
        try:
            assert_no_question_overlap(df, split)
            leak = None
        except AssertionError as exc:
            leak = str(exc).split(".")[0]
        outputs[key] = evaluate_protocol(df, split, seed=cfg.seed, verbose=verbose)
        outputs[key]["question_overlap"] = leak

    _print_comparison(outputs)

    clean = outputs["cluster_disjoint_split"]
    print("\nmost confident mistakes (LightGBM, clean protocol):")
    featurizer = clean["_featurizer"]
    test_df = clean["_test_df"]
    model = build_models(cfg.seed)["lightgbm"].estimator
    train_df = df.loc[protocols["cluster_disjoint_split"].train]
    model.fit(featurizer.transform(train_df), train_df["is_duplicate"])
    scores = model.predict_proba(featurizer.transform(test_df))[:, 1]
    for _, row in worst_errors(test_df, scores, n=4).iterrows():
        print(f"  label={row['is_duplicate']} score={row['score']:.3f} [{row['pair_kind']}]")
        print(f"    Q1: {row['question1']}")
        print(f"    Q2: {row['question2']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "n_pairs": cfg.n_pairs,
            "n_intents": cfg.n_intents,
            "duplicate_rate": cfg.duplicate_rate,
            "hard_negative_share": cfg.hard_negative_share,
            "seed": cfg.seed,
        },
        "protocols": {
            key: {k: v for k, v in value.items() if not k.startswith("_")}
            for key, value in outputs.items()
        },
    }
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")

    if make_plots:
        _write_plots(outputs, output_dir)

    return summary


def _print_comparison(outputs: dict) -> None:
    print("\n" + "=" * 78)
    print("the same models, scored under each protocol (ROC-AUC on that protocol's test set)")
    print("=" * 78)
    models = list(outputs["random_pair_split"]["models"])
    print(f"{'model':<26}{'leaky split':>12}{'clean split':>12}{'inflation':>12}")
    for name in models:
        leaky = outputs["random_pair_split"]["models"][name]["overall"]["roc_auc"]
        clean = outputs["cluster_disjoint_split"]["models"][name]["overall"]["roc_auc"]
        flag = " <- identity-keyed features" if outputs["random_pair_split"][
            "models"
        ][name]["uses_question_stats"] else ""
        print(f"{name:<26}{leaky:>12.3f}{clean:>12.3f}{leaky - clean:>+12.3f}{flag}")


def _write_plots(outputs: dict, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plots are optional
        print("matplotlib not installed; skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    models = list(outputs["random_pair_split"]["models"])
    x = np.arange(len(models))
    leaky = [outputs["random_pair_split"]["models"][m]["overall"]["roc_auc"] for m in models]
    clean = [
        outputs["cluster_disjoint_split"]["models"][m]["overall"]["roc_auc"] for m in models
    ]
    axes[0].bar(x - 0.2, leaky, width=0.4, label="random pair split (leaky)")
    axes[0].bar(x + 0.2, clean, width=0.4, label="cluster-disjoint split")
    axes[0].set_xticks(x, models, rotation=15, fontsize=8)
    axes[0].set(ylabel="ROC-AUC", title="What the protocol does to the score")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")

    calib = pd.DataFrame(
        outputs["cluster_disjoint_split"]["models"]["lightgbm"]["calibration"]
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect")
    axes[1].plot(
        calib["mean_predicted"], calib["observed_rate"], marker="o", linewidth=1.6,
        label="lightgbm",
    )
    axes[1].set(
        xlabel="mean predicted probability",
        ylabel="observed duplicate rate",
        title="Calibration (clean protocol)",
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
    p.add_argument("--n-pairs", type=int, default=40_000)
    p.add_argument("--n-intents", type=int, default=900)
    p.add_argument("--duplicate-rate", type=float, default=0.37)
    p.add_argument(
        "--hard-negative-share",
        type=float,
        default=0.6,
        help="share of negatives that share two of three intent slots",
    )
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = PairConfig(
        n_pairs=args.n_pairs,
        n_intents=args.n_intents,
        duplicate_rate=args.duplicate_rate,
        hard_negative_share=args.hard_negative_share,
        seed=args.seed,
    )
    run(cfg, args.output_dir, test_size=args.test_size, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
