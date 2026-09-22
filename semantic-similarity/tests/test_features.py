"""Feature symmetry, fit/transform discipline, and the target encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import PairConfig, generate_pairs
from src.features import (
    FEATURE_NAMES,
    PairFeaturizer,
    QuestionTargetEncoder,
    content_tokens,
    tokenize,
)


@pytest.fixture(scope="module")
def pairs() -> pd.DataFrame:
    return generate_pairs(PairConfig(n_pairs=3_000, n_intents=150, seed=73))


@pytest.fixture(scope="module")
def featurizer(pairs: pd.DataFrame) -> PairFeaturizer:
    return PairFeaturizer().fit(pairs)


def test_features_are_symmetric(pairs: pd.DataFrame, featurizer: PairFeaturizer) -> None:
    """f(q1, q2) must equal f(q2, q1).

    Question order in these datasets is arbitrary. An asymmetric feature --
    "length of question 1" -- teaches the model an artefact of how rows were
    written down, and it will not survive the rows being written down
    differently.
    """
    swapped = pairs.rename(columns={"question1": "question2", "question2": "question1"})

    original = featurizer.transform(pairs)
    reversed_ = featurizer.transform(swapped)

    pd.testing.assert_frame_equal(original, reversed_, check_exact=False, rtol=1e-9)


def test_identical_questions_are_maximally_similar(featurizer: PairFeaturizer) -> None:
    same = pd.DataFrame(
        {
            "question1": ["How do I install Docker on Ubuntu?"],
            "question2": ["How do I install Docker on Ubuntu?"],
        }
    )
    row = featurizer.transform(same).iloc[0]

    assert row["token_jaccard"] == pytest.approx(1.0)
    assert row["content_jaccard"] == pytest.approx(1.0)
    assert row["tfidf_word_cosine"] > 0.99
    assert row["abs_char_length_diff"] == 0.0
    assert row["unique_to_shorter"] == 0.0


def test_unrelated_questions_score_low(featurizer: PairFeaturizer) -> None:
    different = pd.DataFrame(
        {
            "question1": ["How do I install Docker on Ubuntu?"],
            "question2": ["What is the best way to learn guitar as a beginner?"],
        }
    )
    row = featurizer.transform(different).iloc[0]
    assert row["content_jaccard"] < 0.15
    assert row["tfidf_word_cosine"] < 0.3


def test_hard_negatives_have_high_overlap(pairs: pd.DataFrame, featurizer) -> None:
    """The property that makes them hard -- if it is absent, the task is fake."""
    features = featurizer.transform(pairs)
    hard = features.loc[(pairs["pair_kind"] == "hard_negative").to_numpy()]
    random_neg = features.loc[(pairs["pair_kind"] == "random_negative").to_numpy()]

    assert hard["content_jaccard"].mean() > random_neg["content_jaccard"].mean() * 1.5


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        PairFeaturizer().transform(pd.DataFrame({"question1": ["a"], "question2": ["b"]}))
    with pytest.raises(RuntimeError, match="fit must be called"):
        QuestionTargetEncoder().transform(
            pd.DataFrame({"question1": ["a"], "question2": ["b"]})
        )


def test_schema_is_stable(pairs: pd.DataFrame, featurizer: PairFeaturizer) -> None:
    features = featurizer.transform(pairs)
    assert list(features.columns) == FEATURE_NAMES
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy()).all()


def test_stopwords_are_removed_from_content_tokens() -> None:
    tokens = tokenize("What is the best way to install Docker?")
    content = content_tokens(tokens)
    assert "install" in content and "docker" in content
    assert "the" not in content and "what" not in content


def test_target_encoder_falls_back_to_the_prior_for_unseen_questions(
    pairs: pd.DataFrame,
) -> None:
    """The mechanism behind the whole result.

    Under a disjoint split every test question is unseen, so the encoding
    collapses to a constant and can carry no information.
    """
    encoder = QuestionTargetEncoder().fit(pairs)

    unseen = pd.DataFrame(
        {
            "question1": ["A question that appears in no training pair at all?"],
            "question2": ["Another entirely novel question, also unseen?"],
        }
    )
    encoded = encoder.transform(unseen).iloc[0]

    assert encoder.unseen_share(unseen) == 1.0
    assert encoded["question_target_enc_min"] == pytest.approx(encoder.prior_)
    assert encoded["question_target_enc_max"] == pytest.approx(encoder.prior_)
    assert encoded["question_seen_count_max"] == 0.0


def test_target_encoder_sees_almost_everything_after_a_random_split(
    pairs: pd.DataFrame,
) -> None:
    from src.split import random_pair_split

    split = random_pair_split(pairs, test_size=0.25, seed=0)
    encoder = QuestionTargetEncoder().fit(pairs.loc[split.train])
    assert encoder.unseen_share(pairs.loc[split.test]) < 0.05


def test_target_encoding_is_symmetric(pairs: pd.DataFrame) -> None:
    encoder = QuestionTargetEncoder().fit(pairs)
    swapped = pairs.rename(columns={"question1": "question2", "question2": "question1"})
    pd.testing.assert_frame_equal(encoder.transform(pairs), encoder.transform(swapped))
