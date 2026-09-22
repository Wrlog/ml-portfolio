"""Generator structure, metric behaviour, and an end-to-end run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import PairConfig, generate_pairs
from src.evaluate import (
    best_f1_threshold,
    calibration_table,
    classification_metrics,
    metrics_by_pair_kind,
    worst_errors,
)
from src.model import CosineBaseline, build_models
from src.train import run


@pytest.fixture(scope="module")
def pairs() -> pd.DataFrame:
    return generate_pairs(PairConfig(n_pairs=6_000, n_intents=200, seed=79))


def test_question_bank_is_reused_across_pairs(pairs: pd.DataFrame) -> None:
    """Each question must appear in many pairs, as in the real dataset.

    If questions were unique per row nothing could cross a split, and the
    entire protocol comparison would be vacuous.
    """
    questions = pd.concat([pairs["question1"], pairs["question2"]])
    appearances = len(questions) / questions.nunique()
    assert appearances > 5


def test_duplicate_pairs_never_repeat_the_same_text(pairs: pd.DataFrame) -> None:
    """Otherwise string equality would solve the positive class."""
    assert (pairs["question1"] == pairs["question2"]).sum() == 0


def test_duplicate_rate_matches_the_config() -> None:
    cfg = PairConfig(n_pairs=4_000, n_intents=150, duplicate_rate=0.45, seed=83)
    assert generate_pairs(cfg)["is_duplicate"].mean() == pytest.approx(0.45, abs=0.01)


def test_duplicates_share_a_cluster_and_negatives_do_not(pairs: pd.DataFrame) -> None:
    duplicates = pairs[pairs["is_duplicate"] == 1]
    negatives = pairs[pairs["is_duplicate"] == 0]
    assert (duplicates["cluster1"] == duplicates["cluster2"]).all()
    assert (negatives["cluster1"] != negatives["cluster2"]).all()


def test_hard_negatives_are_present_in_the_requested_proportion() -> None:
    cfg = PairConfig(n_pairs=4_000, n_intents=150, hard_negative_share=0.6, seed=89)
    df = generate_pairs(cfg)
    negatives = df[df["is_duplicate"] == 0]
    share = (negatives["pair_kind"] == "hard_negative").mean()
    assert share == pytest.approx(0.6, abs=0.05)


def test_duplicate_popularity_is_skewed(pairs: pd.DataFrame) -> None:
    """Popular intents attract more duplicates -- the source of the leak.

    With flat popularity the identity-keyed features carry no signal and the
    protocol comparison shows nothing.
    """
    counts = pairs[pairs["is_duplicate"] == 1]["cluster1"].value_counts()
    assert counts.iloc[0] > counts.median() * 4


def test_generation_is_reproducible() -> None:
    cfg = PairConfig(n_pairs=1_500, n_intents=120, seed=97)
    pd.testing.assert_frame_equal(generate_pairs(cfg), generate_pairs(cfg))


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate_rate"):
        PairConfig(duplicate_rate=0.0)
    with pytest.raises(ValueError, match="hard_negative_share"):
        PairConfig(hard_negative_share=1.5)
    with pytest.raises(ValueError, match="intents"):
        PairConfig(n_intents=10)
    with pytest.raises(ValueError, match="surface forms"):
        PairConfig(forms_per_intent=1)


def test_metrics_report_the_majority_class_floor() -> None:
    y = np.array([1] * 37 + [0] * 63)
    scores = np.linspace(0, 1, 100)
    metrics = classification_metrics(y, scores, threshold=0.5)
    assert metrics["majority_class_accuracy"] == pytest.approx(0.63)
    assert metrics["positive_rate"] == pytest.approx(0.37)


def test_single_class_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="both classes"):
        classification_metrics(np.zeros(10), np.linspace(0, 1, 10))


def test_best_threshold_beats_the_default() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.2).astype(int)
    scores = np.clip(rng.beta(2, 8, 2_000) + 0.25 * y, 0, 1)

    from sklearn.metrics import f1_score

    tuned = best_f1_threshold(y, scores)
    assert f1_score(y, (scores >= tuned).astype(int)) >= f1_score(
        y, (scores >= 0.5).astype(int)
    )


def test_hard_negatives_score_worse_than_random_ones(pairs: pd.DataFrame) -> None:
    """If they did not, the generator would not be producing a real task."""
    from src.features import PairFeaturizer

    featurizer = PairFeaturizer().fit(pairs)
    scores = CosineBaseline().fit(featurizer.transform(pairs)).predict_proba(
        featurizer.transform(pairs)
    )[:, 1]

    table = metrics_by_pair_kind(pairs, scores, threshold=0.5).set_index("negatives")
    assert table.loc["hard_negative", "roc_auc"] < table.loc["random_negative", "roc_auc"]


def test_calibration_table_covers_the_data(pairs: pd.DataFrame) -> None:
    rng = np.random.default_rng(1)
    scores = rng.random(len(pairs))
    table = calibration_table(pairs["is_duplicate"], scores, n_bins=10)
    assert table["n"].sum() == len(pairs)
    assert set(table.columns) >= {"mean_predicted", "observed_rate", "gap"}


def test_worst_errors_returns_the_most_confident_mistakes(pairs: pd.DataFrame) -> None:
    scores = 1.0 - pairs["is_duplicate"].to_numpy(dtype=float)  # maximally wrong
    errors = worst_errors(pairs, scores, n=3)
    assert len(errors) == 3
    assert {"question1", "question2", "score"} <= set(errors.columns)


def test_every_model_fits_and_ranks_better_than_chance(pairs: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score

    from src.features import PairFeaturizer, QuestionTargetEncoder

    featurizer = PairFeaturizer().fit(pairs)
    encoder = QuestionTargetEncoder().fit(pairs)
    base = featurizer.transform(pairs)
    with_stats = pd.concat([base, encoder.transform(pairs)], axis=1)

    for name, spec in build_models().items():
        X = with_stats if spec.uses_question_stats else base
        spec.estimator.fit(X, pairs["is_duplicate"])
        scores = spec.estimator.predict_proba(X)[:, 1]
        assert roc_auc_score(pairs["is_duplicate"], scores) > 0.6, name


def test_run_reports_both_protocols(tmp_path) -> None:
    summary = run(
        cfg=PairConfig(n_pairs=4_000, n_intents=150, seed=101),
        output_dir=tmp_path,
        make_plots=False,
        verbose=False,
    )

    assert (tmp_path / "metrics.json").exists()
    assert set(summary["protocols"]) == {"random_pair_split", "cluster_disjoint_split"}

    leaky = summary["protocols"]["random_pair_split"]
    clean = summary["protocols"]["cluster_disjoint_split"]

    # The leaky protocol is the one that fails the overlap assertion, and the
    # one whose test questions were nearly all seen during fitting.
    assert leaky["question_overlap"] is not None
    assert clean["question_overlap"] is None
    assert leaky["unseen_test_question_share"] < 0.2
    assert clean["unseen_test_question_share"] == pytest.approx(1.0)
