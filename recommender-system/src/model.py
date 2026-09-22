"""Recommenders, from the one that is embarrassingly hard to beat.

`PopularityRecommender` recommends the same list to everybody. It is not
personalised, it took four lines to write, and on a head-heavy catalogue it
scores well enough that a great many published improvements disappear when
measured against it properly. Every number in this project is reported next to
it for that reason.

`ALSRecommender` implements Hu, Koren & Volinsky's implicit-feedback ALS
directly, because the interesting part is exactly the part a library hides. The
key idea is that implicit data has no negatives, so instead of fitting the
observed entries and ignoring the rest, the model fits *every* cell of the
matrix with a confidence that grows with the observed count:

    preference  p_ui = 1 if the user played the item, else 0
    confidence  c_ui = 1 + alpha * r_ui

Unobserved cells get p = 0 at confidence 1 -- a weak assertion that the user
would not have played it, not a statement that they disliked it. Fitting all
n_users x n_items cells naively would be impossible; the trick that makes it
linear in the observed data is precomputing `YtY` once per iteration and adding
only the low-rank correction from each user's own items.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


class BaseRecommender(ABC):
    """Common interface: fit a user-item matrix, return top-k item ids."""

    name: str = "base"

    @abstractmethod
    def fit(self, matrix: sparse.csr_matrix) -> "BaseRecommender":
        ...

    @abstractmethod
    def _scores(self, user_ids: np.ndarray) -> np.ndarray:
        """Dense score matrix, shape (len(user_ids), n_items)."""

    def recommend(
        self,
        user_ids: np.ndarray,
        k: int,
        train_matrix: sparse.csr_matrix | None = None,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Top-k item ids per user, excluding items already in `train_matrix`.

        The exclusion is not optional in practice. Left in, a recommender that
        simply re-ranks a user's own history scores near-perfectly offline and
        recommends nothing anyone can act on.
        """
        user_ids = np.asarray(user_ids)
        out = np.empty((len(user_ids), k), dtype=np.int64)

        for start in range(0, len(user_ids), batch_size):
            batch = user_ids[start : start + batch_size]
            scores = self._scores(batch).astype(np.float64, copy=True)

            if train_matrix is not None:
                seen = train_matrix[batch]
                scores[seen.nonzero()] = -np.inf

            if k >= scores.shape[1]:
                raise ValueError("k must be smaller than the catalogue size")
            top = np.argpartition(-scores, kth=k, axis=1)[:, :k]
            ordered = np.take_along_axis(
                top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1
            )
            out[start : start + len(batch)] = ordered
        return out


class RandomRecommender(BaseRecommender):
    """The floor. Any metric that does not clear this by a wide margin is broken."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, matrix: sparse.csr_matrix) -> "RandomRecommender":
        self.n_items_ = matrix.shape[1]
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def _scores(self, user_ids: np.ndarray) -> np.ndarray:
        return self.rng_.random((len(user_ids), self.n_items_))


class PopularityRecommender(BaseRecommender):
    """Same ranking for everyone, by number of distinct users who played it.

    Distinct users rather than total plays, so one obsessive listener cannot
    push a track up the chart.
    """

    name = "popularity"

    def fit(self, matrix: sparse.csr_matrix) -> "PopularityRecommender":
        binary = matrix.copy()
        binary.data = np.ones_like(binary.data)
        self.item_scores_ = np.asarray(binary.sum(axis=0)).ravel()
        self.n_items_ = matrix.shape[1]
        return self

    def _scores(self, user_ids: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores_, (len(user_ids), 1))


class ItemKNNRecommender(BaseRecommender):
    """Item-item cosine similarity, truncated to the k nearest neighbours.

    Truncation is what makes this work rather than an optimisation: keeping the
    full similarity matrix lets thousands of weak, mostly popularity-driven
    similarities accumulate, and the recommendations collapse back toward the
    popularity baseline.
    """

    name = "item_knn"

    def __init__(self, k_neighbours: int = 100, shrink: float = 10.0):
        self.k_neighbours = k_neighbours
        self.shrink = shrink

    def fit(self, matrix: sparse.csr_matrix) -> "ItemKNNRecommender":
        self.train_ = matrix.tocsr()
        item_user = matrix.T.tocsr().astype(np.float32)

        norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1))).ravel()
        similarity = (item_user @ item_user.T).toarray()
        denominator = np.outer(norms, norms) + self.shrink + 1e-9
        similarity /= denominator
        np.fill_diagonal(similarity, 0.0)

        if self.k_neighbours < similarity.shape[1]:
            # Select exactly k per row. Thresholding on the k-th largest value
            # instead would keep every tie at the cut-off, which on sparse
            # count data -- where many similarities are identical -- quietly
            # leaves more neighbours than asked for.
            keep = np.argpartition(-similarity, self.k_neighbours, axis=1)[
                :, : self.k_neighbours
            ]
            mask = np.zeros_like(similarity, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            similarity[~mask] = 0.0

        self.similarity_ = similarity
        self.n_items_ = matrix.shape[1]
        return self

    def _scores(self, user_ids: np.ndarray) -> np.ndarray:
        profile = self.train_[user_ids].toarray()
        return profile @ self.similarity_


class ALSRecommender(BaseRecommender):
    """Implicit-feedback alternating least squares (Hu, Koren & Volinsky 2008)."""

    name = "als"

    def __init__(
        self,
        n_factors: int = 32,
        n_iterations: int = 15,
        # Tuned by sweep (see README). Both values matter more than the factor
        # count: with ~50 observations per user, weak regularisation overfits
        # each user's own history, and a large alpha amplifies that by making
        # those few observations enormously confident.
        regularization: float = 60.0,
        alpha: float = 8.0,
        seed: int = 43,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.seed = seed

    def fit(self, matrix: sparse.csr_matrix) -> "ALSRecommender":
        rng = np.random.default_rng(self.seed)
        n_users, n_items = matrix.shape

        # Confidence, not preference. c = 1 + alpha * r means a single play is
        # weak evidence and twenty plays is strong evidence; an unobserved cell
        # keeps confidence 1, asserting almost nothing.
        confidence = matrix.copy().astype(np.float64)
        confidence.data = self.alpha * confidence.data

        user_items = confidence.tocsr()
        item_users = confidence.T.tocsr()

        self.user_factors_ = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors_ = rng.normal(0, 0.01, (n_items, self.n_factors))
        self.loss_history_: list[float] = []

        for iteration in range(self.n_iterations):
            self.user_factors_ = self._solve(user_items, self.item_factors_)
            self.item_factors_ = self._solve(item_users, self.user_factors_)
            self.loss_history_.append(self._loss(user_items))

        self.n_items_ = n_items
        return self

    def _solve(self, matrix: sparse.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        """One half-iteration: least squares for every row of `matrix`.

        `gramian` is the contribution of *all* items at confidence 1, computed
        once. Each row then adds only the correction from its own observed
        entries, which is what keeps the cost proportional to the number of
        observations rather than to n_users x n_items.

        The system is solved by Cholesky rather than `np.linalg.solve`. `A` is
        symmetric positive definite by construction -- a Gram matrix, plus a
        positive-semidefinite correction, plus a positive diagonal -- so
        Cholesky is both the correct factorisation and about twice the speed of
        a general LU on paper. In practice the gap was far larger: on the
        machine this was developed on, `np.linalg.solve` took 148 ms per 48x48
        system against 0.046 ms for `cho_solve`, a pathology in that LAPACK
        build which turned a 5-second fit into one that never finished. Since
        this loop runs tens of thousands of times, the factorisation choice is
        not a micro-optimisation here.
        """
        n_rows = matrix.shape[0]
        n_factors = fixed.shape[1]
        gramian = fixed.T @ fixed + self.regularization * np.eye(n_factors)

        out = np.zeros((n_rows, n_factors))
        indptr, indices, data = matrix.indptr, matrix.indices, matrix.data

        for row in range(n_rows):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                continue  # no observations: leave the factors at zero
            columns = indices[start:end]
            extra_confidence = data[start:end]  # this is alpha * r, so c - 1

            factors = fixed[columns]
            # A = YtY + Y_u^T (C_u - I) Y_u + lambda*I
            A = gramian + (factors.T * extra_confidence) @ factors
            # b = Y_u^T C_u p_u, and p_u is 1 exactly on the observed entries
            b = ((1.0 + extra_confidence)[:, None] * factors).sum(axis=0)
            out[row] = cho_solve(cho_factor(A, lower=True, check_finite=False), b,
                                 check_finite=False)
        return out

    def _loss(self, user_items: sparse.csr_matrix) -> float:
        """The full ALS objective, including the unobserved cells.

        The objective sums over *every* user-item cell:

            L = sum_ui c_ui (p_ui - x_u . y_i)^2 + lambda (||X||^2 + ||Y||^2)

        Summing that directly would mean touching n_users x n_items entries.
        The identity that avoids it: split each observed cell's contribution
        into "what it would have contributed at c = 1, p = 0" plus a
        correction, leaving a term over all cells that is a pure Gram product,

            sum_ui (x_u . y_i)^2 = trace( (X^T X)(Y^T Y) )

        which costs O(f^2 (n_users + n_items)). An earlier version of this
        method summed only the observed cells; that quantity is *not* the
        objective being minimised and is not guaranteed to fall every
        iteration, so it was useless as a convergence check -- and the
        monotonicity test caught it.
        """
        rows, columns = user_items.nonzero()
        observed = (self.user_factors_[rows] * self.item_factors_[columns]).sum(axis=1)
        confidence = 1.0 + user_items.data  # data holds alpha * r

        # Observed cells: c(1 - s)^2, minus the s^2 the all-cells term already
        # counted for them at confidence 1.
        observed_term = float(
            (confidence * np.square(1.0 - observed) - np.square(observed)).sum()
        )
        all_cells_term = float(
            np.trace(
                (self.user_factors_.T @ self.user_factors_)
                @ (self.item_factors_.T @ self.item_factors_)
            )
        )
        penalty = self.regularization * (
            float(np.square(self.user_factors_).sum())
            + float(np.square(self.item_factors_).sum())
        )
        return observed_term + all_cells_term + penalty

    def _scores(self, user_ids: np.ndarray) -> np.ndarray:
        return self.user_factors_[user_ids] @ self.item_factors_.T


def build_models(seed: int = 43) -> dict[str, BaseRecommender]:
    return {
        "random": RandomRecommender(seed=seed),
        "popularity": PopularityRecommender(),
        "item_knn": ItemKNNRecommender(k_neighbours=100, shrink=10.0),
        "als": ALSRecommender(seed=seed),
    }
