"""The tests that matter here are the leakage tests.

A feature bug that leaks the future does not raise an exception and does not
show up as a bad metric -- it shows up as a *suspiciously good* metric, and
then as a model that fails in production. So the property is asserted
directly: recomputing the features on a truncated log must reproduce exactly
the rows that were already there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import GeneratorConfig, generate_transactions
from src.features import ALL_FEATURES, NO_PRIOR_MINUTES, build_features


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    return generate_transactions(GeneratorConfig(n_accounts=300, n_legit=4_000, seed=11))


def test_no_future_leakage(transactions: pd.DataFrame) -> None:
    """Features for the first k rows must not depend on rows after k.

    This is the real-time guarantee: at scoring time only the past exists, so
    a feature computed then must equal the one computed later on the full log.
    """
    full = build_features(transactions)

    for k in (500, 1_500, 3_000):
        prefix = build_features(transactions.iloc[:k].copy())
        pd.testing.assert_frame_equal(
            prefix,
            full.iloc[:k],
            check_dtype=True,
            obj=f"first {k} rows recomputed on a truncated log",
        )


def test_rolling_windows_exclude_the_current_row(transactions: pd.DataFrame) -> None:
    """A transaction must never count itself in its own velocity window.

    If it did, `txn_count_1h` would be >= 1 everywhere and would encode
    nothing but "this row exists".
    """
    feats = build_features(transactions)
    assert (feats["txn_count_1h"] >= 0).all()
    # Most accounts transact far less than hourly, so the modal value is zero.
    assert (feats["txn_count_1h"] == 0).mean() > 0.5


def test_velocity_matches_a_brute_force_count() -> None:
    """Check the vectorised window against an obvious O(n^2) implementation."""
    df = generate_transactions(GeneratorConfig(n_accounts=40, n_legit=600, seed=3))
    feats = build_features(df)

    ts = pd.to_datetime(df["timestamp"]).to_numpy()
    acct = df["account_id"].to_numpy()
    window = np.timedelta64(1, "h")

    for i in np.random.default_rng(0).choice(len(df), size=60, replace=False):
        same_account = acct == acct[i]
        earlier = (ts < ts[i]) & (ts >= ts[i] - window)
        expected = int((same_account & earlier).sum())
        assert feats["txn_count_1h"].iloc[i] == expected


def test_first_transaction_has_no_history(transactions: pd.DataFrame) -> None:
    feats = build_features(transactions)
    first_rows = ~transactions.duplicated(subset="account_id", keep="first").values

    assert (feats.loc[first_rows, "prior_txn_count"] == 0).all()
    assert (feats.loc[first_rows, "has_account_history"] == 0).all()
    assert (feats.loc[first_rows, "minutes_since_prev_txn"] == NO_PRIOR_MINUTES).all()
    # With no history the ratio has nothing to compare against, so it falls
    # back to "exactly typical" rather than to a NaN the model must guess at.
    assert (feats.loc[first_rows, "amount_to_account_mean"] == 1.0).all()


def test_no_missing_values_and_stable_schema(transactions: pd.DataFrame) -> None:
    feats = build_features(transactions)
    assert list(feats.columns) == ALL_FEATURES
    assert not feats.isna().any().any()
    assert len(feats) == len(transactions)


def test_missing_input_column_raises() -> None:
    df = generate_transactions(GeneratorConfig(n_accounts=20, n_legit=100))
    with pytest.raises(KeyError, match="missing columns"):
        build_features(df.drop(columns=["amount"]))
