"""Synthetic implicit-feedback listening log.

Implicit feedback is not ratings. There are no negatives: a user who never
played a track may dislike it, or may simply never have been shown it. Almost
every mistake in a recommender project traces back to forgetting that, so the
generator makes the distinction explicit -- it emits *plays*, never a "did not
like" signal, and the absence of a row means nothing on its own.

Two structural properties are reproduced because they drive the results:

**A power-law catalogue.** A small head of items takes most of the plays. This
is why the popularity baseline in `model.py` is hard to beat and why catalogue
coverage has to be reported next to accuracy: a model that recommends the top
200 items to everyone scores respectably and is worthless as a product.

**Exposure bias.** What a user plays depends on what they were shown, and they
were shown popular things. So the observed log is not a sample of preference,
it is a sample of preference *filtered through* past recommendations, and a
model trained on it inherits that filter. The `exposure_strength` knob controls
how much, and turning it down is the closest this simulation gets to an
unbiased log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GENRES = [
    "indie_rock", "hip_hop", "jazz", "electronic", "classical", "metal",
    "folk", "rnb", "ambient", "punk", "latin", "country",
]


@dataclass
class CatalogConfig:
    n_users: int = 4_000
    n_items: int = 2_000
    n_factors: int = len(GENRES)  # true latent dimension, unknown to the model
    mean_plays_per_user: float = 55.0
    min_plays_per_user: int = 8
    popularity_alpha: float = 1.1  # lower means a heavier head
    exposure_strength: float = 0.5  # 0 = taste only, 1 = popularity dominates
    taste_concentration: float = 0.35  # Dirichlet alpha; lower means narrower taste
    # How sharply taste discriminates between items. This exists because the
    # two effects are not on comparable scales: a Zipf popularity curve spans
    # several nats, while a Dirichlet affinity spans barely two, so an even
    # 50/50 blend is in practice dominated by popularity and *no* personalised
    # model can beat the popularity baseline. Raising this widens taste's
    # dynamic range until personalisation is worth doing -- which is the
    # regime a recommender is built for. Lower it toward 1.0 to see the
    # opposite regime, where the baseline wins and the honest answer is not to
    # build a model.
    taste_sharpness: float = 5.0
    n_days: int = 180
    seed: int = 43

    def __post_init__(self) -> None:
        if not 0.0 <= self.exposure_strength <= 1.0:
            raise ValueError("exposure_strength must be in [0, 1]")
        if self.taste_sharpness <= 0:
            raise ValueError("taste_sharpness must be positive")
        if self.min_plays_per_user < 3:
            raise ValueError(
                "users need at least 3 plays to support a leave-last-out split "
                "with anything left to train on"
            )
        if self.n_items < 50 or self.n_users < 50:
            raise ValueError("catalogue and audience must be large enough to rank")


def _weighted_sample_without_replacement(
    rng: np.random.Generator, probabilities: np.ndarray, k: int
) -> np.ndarray:
    """Draw `k` distinct indices with probability proportional to `probabilities`.

    Uses the Gumbel top-k trick: perturb each log-weight with Gumbel noise and
    take the k largest. This is exact -- it yields precisely the same
    distribution as sequential weighted draws without replacement -- and costs
    one pass over the catalogue.

    `rng.choice(..., replace=False, p=...)` computes the same thing and is the
    obvious call to reach for, but it degenerates on a skewed distribution:
    drawing a few hundred distinct items from a power-law catalogue sends it
    into a rejection loop that can run for minutes. That is worth a comment
    because the failure is a hang, not an error.
    """
    keys = np.log(probabilities + 1e-300) + rng.gumbel(size=probabilities.size)
    if k >= probabilities.size:
        return np.argsort(-keys)
    return np.argpartition(-keys, k)[:k]


def _item_table(rng: np.random.Generator, cfg: CatalogConfig) -> pd.DataFrame:
    """Items get a genre mix and an intrinsic popularity."""
    genre_weights = rng.dirichlet(np.full(cfg.n_factors, 0.3), size=cfg.n_items)
    # Zipf-ish popularity: rank r gets weight ~ 1 / r^alpha.
    ranks = np.arange(1, cfg.n_items + 1)
    popularity = 1.0 / np.power(ranks, cfg.popularity_alpha)
    popularity = rng.permutation(popularity)  # decouple popularity from item id
    popularity = popularity / popularity.sum()

    primary = genre_weights.argmax(axis=1)
    return pd.DataFrame(
        {
            "item_id": np.arange(cfg.n_items),
            "primary_genre": [GENRES[i] for i in primary],
            "popularity": popularity,
        }
    ), genre_weights


def generate_interactions(cfg: CatalogConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (interactions, items).

    `interactions` has one row per play: user_id, item_id, timestamp, play_count.
    """
    cfg = cfg or CatalogConfig()
    rng = np.random.default_rng(cfg.seed)

    items, item_genres = _item_table(rng, cfg)
    popularity = items["popularity"].to_numpy()

    # Each user has a narrow taste profile over genres.
    user_taste = rng.dirichlet(
        np.full(cfg.n_factors, cfg.taste_concentration), size=cfg.n_users
    )

    # Activity is itself heavy-tailed: a few users generate most of the log.
    n_plays = np.clip(
        rng.poisson(rng.gamma(2.0, cfg.mean_plays_per_user / 2.0, size=cfg.n_users)),
        cfg.min_plays_per_user,
        cfg.n_items // 2,
    )

    log_popularity = np.log(popularity + 1e-12)
    rows_u, rows_i, rows_c = [], [], []

    for user in range(cfg.n_users):
        affinity = item_genres @ user_taste[user]
        # Blend taste with exposure in log space, so `exposure_strength` mixes
        # two multiplicative effects rather than two probabilities.
        logits = (
            (1.0 - cfg.exposure_strength) * cfg.taste_sharpness * np.log(affinity + 1e-9)
            + cfg.exposure_strength * log_popularity
        )
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        k = int(min(n_plays[user], (probabilities > 0).sum()))
        chosen = _weighted_sample_without_replacement(rng, probabilities, k)
        # Repeat plays are themselves a signal of strength, unlike a rating.
        counts = 1 + rng.geometric(0.55, size=k) - 1

        rows_u.append(np.full(k, user))
        rows_i.append(chosen)
        rows_c.append(counts)

    user_ids = np.concatenate(rows_u)
    item_ids = np.concatenate(rows_i)
    counts = np.concatenate(rows_c)

    start = pd.Timestamp("2024-01-01")
    offsets = rng.integers(0, cfg.n_days * 86_400, size=len(user_ids))
    timestamps = start + pd.to_timedelta(offsets, unit="s")

    interactions = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "timestamp": timestamps,
            "play_count": np.maximum(counts, 1).astype(int),
        }
    ).sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    return interactions, items


def interaction_summary(interactions: pd.DataFrame, items: pd.DataFrame) -> dict:
    """Head-heaviness and sparsity, the two numbers that frame every result."""
    plays_per_item = interactions.groupby("item_id").size().sort_values(ascending=False)
    total = len(interactions)
    head = int(np.ceil(0.01 * len(items)))
    return {
        "interactions": total,
        "users": int(interactions["user_id"].nunique()),
        "items_with_plays": int(plays_per_item.size),
        "catalogue_size": int(len(items)),
        "density": total / (interactions["user_id"].nunique() * len(items)),
        "median_plays_per_user": float(interactions.groupby("user_id").size().median()),
        "share_of_plays_in_top_1pct_items": float(plays_per_item.head(head).sum() / total),
        "items_never_played": int(len(items) - plays_per_item.size),
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    frame, catalogue = generate_interactions()
    print(frame.head(8).to_string(index=False))
    for key, value in interaction_summary(frame, catalogue).items():
        print(f"{key:>32}: {value:,.4f}" if isinstance(value, float) else f"{key:>32}: {value:,}")
