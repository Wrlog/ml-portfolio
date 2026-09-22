"""Pair features for duplicate detection, without a GPU.

The transformer answer to this task is a cross-encoder, and it is better. It is
also a 100M-parameter model with a latency budget that rules it out as a
first-pass filter over a question bank of millions. This module builds the
cheap layer that runs first: a handful of lexical similarity features that a
linear model or a GBDT can score in microseconds.

Two things are done carefully because they are where this task leaks:

* **The vectorisers are fit on training questions only.** Fitting TF-IDF on the
  full corpus lets test-set vocabulary and document frequencies into the model.
  The effect is small on a large corpus and embarrassing to explain, so the
  featuriser is a proper fit/transform object rather than a function.
* **Features are symmetric.** `f(q1, q2)` must equal `f(q2, q1)`. Question
  order in these datasets is arbitrary, so any asymmetric feature -- "length of
  question 1" -- teaches the model an artefact of row construction. Where a
  direction matters, both the min and the max go in, never the raw pair.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Deliberately small: these carry no topical information but dominate raw
# overlap counts, since every question in the corpus is phrased as a question.
STOPWORDS = frozenset(
    """a an the is are was were do does did i my me you your it its of to in on
    for with at by from how what why when where which who whom can could should
    would will shall there here that this these those and or but if then than
    as be been being have has had not no any some best easiest way ways get
    someone anyone""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

FEATURE_NAMES = [
    "tfidf_word_cosine",
    "tfidf_char_cosine",
    "token_jaccard",
    "content_jaccard",
    "idf_weighted_overlap",
    "shared_token_count",
    "min_token_count",
    "max_token_count",
    "token_count_ratio",
    "abs_char_length_diff",
    "min_char_length",
    "common_prefix_words",
    "same_leading_word",
    "same_final_content_word",
    "digit_overlap",
    "unique_to_shorter",
]


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def content_tokens(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _row_cosine(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    """Cosine per row-pair. TF-IDF output is L2-normalised, so this is a dot."""
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


class PairFeaturizer:
    """Fit vectorisers on training questions, then build symmetric pair features."""

    def __init__(self, max_word_features: int = 50_000, max_char_features: int = 60_000):
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=max_word_features,
        )
        # Character n-grams absorb the typos and spacing corruptions that word
        # tokens miss entirely.
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=max_char_features,
        )
        self.idf_: dict[str, float] = {}
        self.default_idf_ = 1.0
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "PairFeaturizer":
        corpus = pd.concat([df["question1"], df["question2"]], ignore_index=True).astype(str)
        self.word_vectorizer.fit(corpus)
        self.char_vectorizer.fit(corpus)

        vocabulary = self.word_vectorizer.vocabulary_
        idf = self.word_vectorizer.idf_
        self.idf_ = {
            term: float(idf[index]) for term, index in vocabulary.items() if " " not in term
        }
        # Unseen terms are rare by construction, so they get the highest idf
        # observed rather than a neutral value.
        self.default_idf_ = float(idf.max()) if len(idf) else 1.0
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("PairFeaturizer.fit must be called before transform")

        q1 = df["question1"].astype(str)
        q2 = df["question2"].astype(str)

        word_cosine = _row_cosine(
            self.word_vectorizer.transform(q1), self.word_vectorizer.transform(q2)
        )
        char_cosine = _row_cosine(
            self.char_vectorizer.transform(q1), self.char_vectorizer.transform(q2)
        )

        rows = []
        for text1, text2 in zip(q1, q2):
            rows.append(self._lexical_features(text1, text2))
        lexical = pd.DataFrame(rows, index=df.index)

        lexical["tfidf_word_cosine"] = word_cosine
        lexical["tfidf_char_cosine"] = char_cosine
        return lexical[FEATURE_NAMES]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _lexical_features(self, text1: str, text2: str) -> dict:
        tokens1, tokens2 = tokenize(text1), tokenize(text2)
        set1, set2 = set(tokens1), set(tokens2)
        content1, content2 = content_tokens(tokens1), content_tokens(tokens2)
        shared = set1 & set2

        # Overlap weighted by how informative the shared words are. Two
        # questions sharing "kubernetes" are far more alike than two sharing
        # "how" -- plain overlap counts cannot see the difference.
        shared_weight = sum(self.idf_.get(t, self.default_idf_) for t in shared)
        total_weight = sum(self.idf_.get(t, self.default_idf_) for t in set1 | set2)

        # The decisive feature for hard negatives: the words present in only
        # one of the two questions, relative to the shorter question. For
        # "...on Android" vs "...on iOS" this is small in absolute terms and
        # carries the entire meaning.
        shorter = content1 if len(content1) <= len(content2) else content2
        longer = content2 if len(content1) <= len(content2) else content1
        unique_to_shorter = len(shorter - longer) / max(len(shorter), 1)

        digits1 = {t for t in tokens1 if t.isdigit()}
        digits2 = {t for t in tokens2 if t.isdigit()}

        prefix = 0
        for a, b in zip(tokens1, tokens2):
            if a != b:
                break
            prefix += 1

        final1 = sorted(content1)[-1] if content1 else ""
        final2 = sorted(content2)[-1] if content2 else ""

        return {
            "token_jaccard": _jaccard(set1, set2),
            "content_jaccard": _jaccard(content1, content2),
            "idf_weighted_overlap": shared_weight / total_weight if total_weight else 0.0,
            "shared_token_count": float(len(shared)),
            "min_token_count": float(min(len(tokens1), len(tokens2))),
            "max_token_count": float(max(len(tokens1), len(tokens2))),
            "token_count_ratio": (
                min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2), 1)
            ),
            "abs_char_length_diff": float(abs(len(text1) - len(text2))),
            "min_char_length": float(min(len(text1), len(text2))),
            "common_prefix_words": float(prefix),
            "same_leading_word": float(
                bool(tokens1) and bool(tokens2) and tokens1[0] == tokens2[0]
            ),
            "same_final_content_word": float(bool(final1) and final1 == final2),
            "digit_overlap": _jaccard(digits1, digits2),
            "unique_to_shorter": unique_to_shorter,
        }


class QuestionTargetEncoder:
    """Per-question target statistics -- the feature that makes the split matter.

    Target encoding an entity id is an ordinary, widely used technique: for each
    question, the average label over the *training* pairs it appears in. It is
    computed here correctly, from training rows only, with a smoothed fallback
    to the global prior for questions never seen.

    That is what makes it the right demonstration. The feature is not a bug.
    Paired with a random split of the pair table it produces a large, entirely
    fake improvement, because the same question sits on both sides of the split
    and arrives at test time carrying its own answer. Paired with a
    cluster-disjoint split every test question is unseen, the encoding falls
    back to the prior, and the improvement vanishes.

    The models built on symmetric lexical features alone barely move between
    the two protocols -- there is nothing in them to memorise. Add one
    identity-keyed feature and the protocol decides the reported score.
    """

    COLUMNS = [
        "question_target_enc_min",
        "question_target_enc_max",
        "question_seen_count_min",
        "question_seen_count_max",
    ]

    def __init__(self, smoothing: float = 5.0):
        self.smoothing = smoothing
        self.stats_: dict[str, tuple[float, int]] = {}
        self.prior_ = 0.5
        self.fitted_ = False

    def fit(self, df: pd.DataFrame, label: str = "is_duplicate") -> "QuestionTargetEncoder":
        self.prior_ = float(df[label].mean())

        long = pd.concat(
            [
                pd.DataFrame({"q": df["question1"].astype(str), "y": df[label]}),
                pd.DataFrame({"q": df["question2"].astype(str), "y": df[label]}),
            ],
            ignore_index=True,
        )
        grouped = long.groupby("q", observed=True)["y"].agg(["sum", "count"])
        # Smoothed toward the prior so a question seen once does not get a
        # target encoding of exactly 0 or 1.
        smoothed = (grouped["sum"] + self.smoothing * self.prior_) / (
            grouped["count"] + self.smoothing
        )
        self.stats_ = {
            q: (float(smoothed.loc[q]), int(grouped.loc[q, "count"])) for q in grouped.index
        }
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("QuestionTargetEncoder.fit must be called before transform")

        def lookup(text: str) -> tuple[float, int]:
            return self.stats_.get(str(text), (self.prior_, 0))

        left = [lookup(t) for t in df["question1"]]
        right = [lookup(t) for t in df["question2"]]

        enc1 = np.array([v for v, _ in left])
        enc2 = np.array([v for v, _ in right])
        seen1 = np.array([c for _, c in left], dtype=float)
        seen2 = np.array([c for _, c in right], dtype=float)

        # Symmetric, like every other feature in this module.
        return pd.DataFrame(
            {
                "question_target_enc_min": np.minimum(enc1, enc2),
                "question_target_enc_max": np.maximum(enc1, enc2),
                "question_seen_count_min": np.minimum(seen1, seen2),
                "question_seen_count_max": np.maximum(seen1, seen2),
            },
            index=df.index,
        )

    def unseen_share(self, df: pd.DataFrame) -> float:
        """Share of questions in `df` the encoder has never seen.

        Near 0 under a random pair split and near 1 under a disjoint one --
        the single number that explains the whole result.
        """
        texts = pd.concat([df["question1"], df["question2"]]).astype(str)
        return float((~texts.isin(self.stats_.keys())).mean())
