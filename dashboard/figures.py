"""Figures for the results dashboard, one set per project.

Each function reads that project's metrics.json and returns PNG bytes. The
figure is chosen to show the point the project's README says is worth
reading, not to show every number that exists.

No dual-axis charts: where two measures are on different scales they get two
panels, so nothing is made to look correlated by axis choice.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .theme import Theme, bar_labels, finish, legend, style_axes

NICE = {
    "rule_baseline": "Rule baseline",
    "logistic_regression": "Logistic regression",
    "lightgbm": "LightGBM",
    "lightgbm_weighted": "LightGBM + weight",
    "lightgbm_tweedie": "LightGBM (Tweedie)",
    "lightgbm_question_stats": "LightGBM + question stats",
    "tfidf_cosine": "TF-IDF cosine",
    "seasonal_naive": "Seasonal naive",
    "item_knn": "Item k-NN",
    "als": "Implicit ALS",
    "popularity": "Popularity",
    "random": "Random",
}


def label(key: str) -> str:
    return NICE.get(key, key.replace("_", " ").title())


# --- fraud detection -------------------------------------------------------

def fraud(metrics: dict, theme: Theme) -> bytes:
    """Ranking quality and money saved, as two panels rather than two axes."""
    models = list(metrics["models"])
    pr_auc = [metrics["models"][m]["ranking"]["pr_auc"] for m in models]
    savings = [metrics["models"][m]["operating_point"]["savings_rate"] * 100
               for m in models]
    names = [label(m) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax, values, title, ylab, fmt in (
        (axes[0], pr_auc, "Ranking quality", "PR-AUC", "{:.3f}"),
        (axes[1], savings, "Money kept", "Net savings (% of do-nothing loss)", "{:.0f}%"),
    ):
        style_axes(ax, theme)
        best = int(np.argmax(values))
        colors = [theme.series[0] if i == best else theme.muted
                  for i in range(len(values))]
        bars = ax.bar(range(len(values)), values, color=colors, width=0.64)
        bar_labels(ax, bars, values, theme, fmt)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
        ax.set_title(title, fontsize=11, loc="left", pad=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    return finish(fig)


# --- demand forecasting ----------------------------------------------------

def demand(metrics: dict, theme: Theme) -> bytes:
    """Error by model, and how error grows with how far ahead you forecast."""
    rows = sorted(metrics["models"], key=lambda r: r["wape"])
    names = [label(r["model"]) for r in rows]
    wape = [r["wape"] for r in rows]

    horizon = metrics.get("best_model_by_horizon_week", [])
    days = [h["days_ahead"] for h in horizon]
    h_wape = [h["wape"] for h in horizon]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    style_axes(ax, theme)
    colors = [theme.series[0] if i == 0 else theme.muted for i in range(len(wape))]
    bars = ax.barh(range(len(wape))[::-1], wape, color=colors, height=0.62)
    for bar, value in zip(bars, wape, strict=True):
        ax.text(value + max(wape) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", color=theme.ink_2, fontsize=9)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=8.5)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=theme.grid, linewidth=0.8)
    ax.set_xlim(0, max(wape) * 1.16)
    ax.set_xlabel("WAPE (lower is better)", fontsize=9)
    ax.set_title("Error by model, six rolling origins", fontsize=11, loc="left", pad=8)

    ax = axes[1]
    style_axes(ax, theme)
    if days:
        ax.plot(days, h_wape, color=theme.series[0], linewidth=2)
        ax.scatter(days, h_wape, s=18, color=theme.series[0],
                   edgecolors=theme.surface, linewidths=0.8, zorder=3)
    ax.set_xlabel("Days ahead", fontsize=9)
    ax.set_ylabel("WAPE", fontsize=9)
    ax.set_title("Error by days ahead", fontsize=11, loc="left", pad=8)
    fig.tight_layout()
    return finish(fig)


# --- semantic similarity ---------------------------------------------------

def semantic(metrics: dict, theme: Theme) -> bytes:
    """The protocol moves the score more than the model does."""
    protocols = metrics["protocols"]
    proto_names = list(protocols)
    models = list(protocols[proto_names[0]]["models"])

    def score(proto: str, model: str) -> float:
        overall = protocols[proto]["models"][model]["overall"]
        for key in ("pr_auc", "average_precision", "roc_auc", "f1"):
            if key in overall:
                return overall[key]
        return float("nan")

    metric_name = next(
        k for k in ("pr_auc", "average_precision", "roc_auc", "f1")
        if k in protocols[proto_names[0]]["models"][models[0]]["overall"]
    )

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    style_axes(ax, theme)
    for i, proto in enumerate(proto_names):
        values = [score(proto, m) for m in models]
        offset = (i - (len(proto_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width * 0.92,
                      color=theme.series[i], label=proto.replace("_", " "))
        bar_labels(ax, bars, values, theme, "{:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in models], rotation=15, ha="right",
                       fontsize=8.5)
    ax.set_ylabel(metric_name.replace("_", " ").upper(), fontsize=9)
    ax.set_title("Same models, two split protocols", fontsize=11, loc="left", pad=8)
    ax.set_ylim(0, 1.12)
    legend(ax, theme, loc="upper right", ncol=2)
    fig.tight_layout()
    return finish(fig)


# --- recommender -----------------------------------------------------------

def recommender(metrics: dict, theme: Theme) -> bytes:
    """Accuracy against catalogue coverage: ranking on recall alone hides this."""
    models = metrics["models"]
    names = list(models)
    recall = [models[m]["recall"] for m in names]
    coverage = [models[m]["catalogue_coverage"] for m in names]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style_axes(ax, theme, xgrid=True)
    for i, m in enumerate(names):
        ax.scatter(recall[i], coverage[i], s=95,
                   color=theme.series[i % len(theme.series)],
                   edgecolors=theme.surface, linewidths=1.4, zorder=3)
        ax.annotate(label(m), (recall[i], coverage[i]),
                    textcoords="offset points", xytext=(9, 4),
                    fontsize=9, color=theme.ink)
    k = metrics.get("k", 10)
    ax.set_xlabel(f"Recall@{k} (did it find what the user played)", fontsize=9)
    ax.set_ylabel("Catalogue coverage (share of the catalogue it ever shows)",
                  fontsize=9)
    ax.set_title("Recall vs catalogue coverage", fontsize=11, loc="left", pad=8)
    pad_x = (max(recall) - min(recall) or 0.01) * 0.22
    ax.set_xlim(min(recall) - pad_x, max(recall) + pad_x * 2.2)
    ax.set_ylim(-0.03, min(1.05, max(coverage) * 1.25 + 0.05))
    fig.tight_layout()
    return finish(fig)
