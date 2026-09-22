"""Models over the pair features.

The ladder is short because the point of this project is the evaluation
protocol, not the estimator:

1. `CosineBaseline` -- rank by TF-IDF cosine alone. This is what "just use
   similarity" means in practice, and on easy negatives it is nearly as good as
   anything else. Its failure on hard negatives is the entire argument for the
   rest.
2. Logistic regression on the pair features, calibrated by construction.
3. LightGBM, which can express the interaction the linear model cannot:
   *high overlap AND a content word present in only one question* is the
   signature of a hard negative, and neither half of that means much alone.
4. LightGBM plus per-question target statistics. This one is in the table to
   be caught. It is the only model whose score depends on the splitting
   protocol, and comparing its two numbers is the result of the project.

Each entry declares which feature block it consumes, so `train.py` can build
the identity-keyed features once and hand them only to the model that asked
for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CosineBaseline(BaseEstimator, ClassifierMixin):
    """Predict duplication from one similarity column, with no fitting.

    The column is already a number in [0, 1], so it is used directly as a
    score. It is not a calibrated probability and the log-loss will say so --
    which is the honest way to report a similarity threshold dressed up as a
    classifier.
    """

    def __init__(self, column: str = "tfidf_word_cosine"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        if self.column not in X.columns:
            raise KeyError(f"{self.column!r} is not in the feature frame")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(X[self.column].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class ModelSpec:
    """An estimator plus the feature block it is fitted on."""

    estimator: object
    uses_question_stats: bool = False


def _lightgbm(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_models(random_state: int = 29) -> dict[str, ModelSpec]:
    return {
        "tfidf_cosine": ModelSpec(CosineBaseline()),
        "logistic_regression": ModelSpec(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=2_000)),
                ]
            )
        ),
        "lightgbm": ModelSpec(_lightgbm(random_state)),
        "lightgbm_question_stats": ModelSpec(
            _lightgbm(random_state), uses_question_stats=True
        ),
    }
