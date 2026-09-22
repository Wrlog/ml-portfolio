"""Model behaviour, the ALS implementation, and an end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.data import CatalogConfig, generate_interactions, interaction_summary
from src.evaluate import evaluate_recommender
from src.model import (
    ALSRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    build_models,
)
from src.split import drop_repeat_consumption, leave_last_out
from src.train import run


@pytest.fixture(scope="module")
def dataset():
    interactions, _ = generate_interactions(
        CatalogConfig(n_users=400, n_items=500, seed=109)
    )
    return leave_last_out(drop_repeat_consumption(interactions), 500, n_holdout=2)


def test_recommendations_are_distinct_and_in_range(dataset) -> None:
    for model in build_models(seed=1).values():
        model.fit(dataset.train_matrix)
        recommendations = model.recommend(
            dataset.test_users[:40], k=10, train_matrix=dataset.train_matrix
        )
        assert recommendations.shape == (40, 10)
        assert ((recommendations >= 0) & (recommendations < 500)).all()
        for row in recommendations:
            assert len(set(row.tolist())) == 10, f"{model.name} repeated an item"


def test_recommendations_are_ordered_by_score(dataset) -> None:
    """Rank 1 must really be the highest-scoring item, not merely in the top k."""
    model = ALSRecommender(n_factors=8, n_iterations=3).fit(dataset.train_matrix)
    users = dataset.test_users[:20]
    recommendations = model.recommend(users, k=10, train_matrix=None)

    scores = model.user_factors_[users] @ model.item_factors_.T
    for row_scores, row_items in zip(scores, recommendations):
        picked = row_scores[row_items]
        assert np.all(np.diff(picked) <= 1e-9), "recommendations are not sorted"


def test_popularity_gives_everyone_the_same_list(dataset) -> None:
    model = PopularityRecommender().fit(dataset.train_matrix)
    unmasked = model.recommend(dataset.test_users[:10], k=10, train_matrix=None)
    assert (unmasked == unmasked[0]).all()


def test_popularity_counts_users_not_plays() -> None:
    """One obsessive listener must not be able to push an item up the chart."""
    matrix = sparse.csr_matrix(
        np.array(
            [
                [100.0, 1.0, 0.0],  # one user, 100 plays of item 0
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    model = PopularityRecommender().fit(matrix)
    assert model.item_scores_[1] > model.item_scores_[0]


def test_als_loss_decreases_monotonically(dataset) -> None:
    """The check that the alternating solve is actually solving something."""
    model = ALSRecommender(n_factors=16, n_iterations=8).fit(dataset.train_matrix)
    history = model.loss_history_

    assert len(history) == 8
    assert history[-1] < history[0]
    # Allow a hair of numerical noise, but no real increases.
    assert all(later <= earlier * 1.001 for earlier, later in zip(history, history[1:]))


def test_als_factor_shapes(dataset) -> None:
    model = ALSRecommender(n_factors=12, n_iterations=3).fit(dataset.train_matrix)
    assert model.user_factors_.shape == (dataset.n_users, 12)
    assert model.item_factors_.shape == (dataset.n_items, 12)
    assert np.isfinite(model.user_factors_).all()
    assert np.isfinite(model.item_factors_).all()


def test_als_recovers_planted_block_structure() -> None:
    """Two disjoint user groups, two disjoint item groups.

    If ALS cannot separate these, the implementation is wrong -- this is the
    easiest possible latent structure.
    """
    n_users, n_items = 60, 40
    dense = np.zeros((n_users, n_items))
    dense[:30, :20] = 1.0  # group A likes the first half of the catalogue
    dense[30:, 20:] = 1.0  # group B likes the second half
    matrix = sparse.csr_matrix(dense)

    model = ALSRecommender(
        n_factors=8, n_iterations=20, regularization=1.0, alpha=10.0
    ).fit(matrix)

    scores = model.user_factors_ @ model.item_factors_.T
    assert scores[:30, :20].mean() > scores[:30, 20:].mean()
    assert scores[30:, 20:].mean() > scores[30:, :20].mean()


def test_als_is_reproducible(dataset) -> None:
    a = ALSRecommender(n_factors=8, n_iterations=4, seed=5).fit(dataset.train_matrix)
    b = ALSRecommender(n_factors=8, n_iterations=4, seed=5).fit(dataset.train_matrix)
    assert np.allclose(a.user_factors_, b.user_factors_)


def test_item_knn_truncates_to_k_neighbours(dataset) -> None:
    model = ItemKNNRecommender(k_neighbours=5).fit(dataset.train_matrix)
    nonzero_per_row = (model.similarity_ != 0).sum(axis=1)
    assert nonzero_per_row.max() <= 5
    assert np.allclose(np.diag(model.similarity_), 0.0), "an item is its own neighbour"


def test_k_larger_than_catalogue_is_rejected(dataset) -> None:
    model = PopularityRecommender().fit(dataset.train_matrix)
    with pytest.raises(ValueError, match="smaller than the catalogue"):
        model.recommend(dataset.test_users[:5], k=500, train_matrix=None)


def test_every_model_beats_random(dataset) -> None:
    ground_truth = dataset.ground_truth()
    users = dataset.test_users
    binary = dataset.train_matrix.copy()
    binary.data = np.ones_like(binary.data)
    popularity = np.asarray(binary.sum(axis=0)).ravel()

    def recall(model):
        model.fit(dataset.train_matrix)
        recommendations = model.recommend(users, k=10, train_matrix=dataset.train_matrix)
        return evaluate_recommender(
            recommendations, users, ground_truth, 500, popularity, k=10
        )["recall"]

    floor = recall(RandomRecommender(seed=0))
    for model in (PopularityRecommender(), ItemKNNRecommender(), ALSRecommender()):
        assert recall(model) > floor * 3, f"{model.name} barely beats random"


def test_generator_produces_a_power_law_catalogue() -> None:
    interactions, items = generate_interactions(
        CatalogConfig(n_users=500, n_items=600, seed=113)
    )
    stats = interaction_summary(interactions, items)
    assert stats["share_of_plays_in_top_1pct_items"] > 0.03
    assert 0.0 < stats["density"] < 0.2


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="exposure_strength"):
        CatalogConfig(exposure_strength=1.5)
    with pytest.raises(ValueError, match="taste_sharpness"):
        CatalogConfig(taste_sharpness=0.0)
    with pytest.raises(ValueError, match="at least 3 plays"):
        CatalogConfig(min_plays_per_user=1)
    with pytest.raises(ValueError, match="large enough"):
        CatalogConfig(n_items=10)


def test_cli_defaults_match_the_config_object() -> None:
    """These drifted apart once and silently changed the headline result.

    The CLI said exposure_strength=0.7 while the dataclass said 0.5, so
    `python -m src.train` and `run(CatalogConfig(), ...)` evaluated different
    datasets.
    """
    from src.train import parse_args

    args = parse_args([])
    defaults = CatalogConfig()

    assert args.exposure_strength == defaults.exposure_strength
    assert args.taste_sharpness == defaults.taste_sharpness
    assert args.n_users == defaults.n_users
    assert args.n_items == defaults.n_items
    assert args.seed == defaults.seed


def test_run_produces_a_metrics_file(tmp_path) -> None:
    summary = run(
        cfg=CatalogConfig(n_users=300, n_items=400, seed=127),
        output_dir=tmp_path,
        k=10,
        make_plots=False,
        verbose=False,
    )

    assert (tmp_path / "metrics.json").exists()
    assert summary["best_by_recall"] in summary["models"]
    for result in summary["models"].values():
        assert 0.0 <= result["recall"] <= 1.0
        assert 0.0 <= result["catalogue_coverage"] <= 1.0
