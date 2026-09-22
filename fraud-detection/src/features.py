"""Causal feature engineering for a transaction stream.

Every feature here is computed from a transaction's own fields plus *strictly
earlier* transactions on the same account. That constraint is the whole point:
the obvious way to build "average spend for this account" is a groupby mean
over the full table, and it leaks the future into the past, which inflates
offline metrics and produces a model that collapses in production. The unit
tests in `tests/test_features.py` assert the no-leakage property directly.

Nothing in this module touches the label, so it is safe to run before any
train/test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "log_amount",
    "amount_to_account_mean",
    "amount_zscore_account",
    "distance_from_home_km",
    "log_account_age_days",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "prior_txn_count",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "minutes_since_prev_txn",
]

BINARY_FEATURES = [
    "card_present",
    "is_foreign",
    "is_night",
    "is_weekend",
    "has_account_history",
    "is_new_category_for_account",
]

CATEGORICAL_FEATURES = ["merchant_category"]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# A large but finite stand-in for "no previous transaction on this account".
# Using a sentinel rather than NaN keeps the tree splits interpretable: the
# model can isolate first-ever transactions as their own branch.
NO_PRIOR_MINUTES = 60 * 24 * 365.0


def _account_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account aggregates over strictly prior transactions.

    Implemented with cumulative sums rather than `expanding().mean()` so the
    cost stays linear -- at issuer volumes the apply-per-group version is the
    difference between seconds and hours.
    """
    srt = df.sort_values(["account_id", "timestamp"], kind="mergesort")
    grp = srt.groupby("account_id", sort=False)

    prior_count = grp.cumcount().astype("float64")
    cum_sum = grp["amount"].cumsum() - srt["amount"]
    cum_sq = (srt["amount"] ** 2).groupby(srt["account_id"]).cumsum() - srt["amount"] ** 2

    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(prior_count > 0, cum_sum / prior_count, np.nan)
        prior_var = np.where(
            prior_count > 1,
            (cum_sq - prior_count * np.square(prior_mean)) / (prior_count - 1),
            np.nan,
        )
    prior_std = np.sqrt(np.clip(prior_var, 0.0, None))

    prev_ts = grp["timestamp"].shift(1)
    minutes_since_prev = (srt["timestamp"] - prev_ts).dt.total_seconds() / 60.0

    # First time this account has transacted in this merchant category.
    new_category = (
        srt.groupby(["account_id", "merchant_category"], sort=False).cumcount() == 0
    )

    out = pd.DataFrame(
        {
            "prior_txn_count": prior_count.to_numpy(),
            "prior_mean_amount": prior_mean,
            "prior_std_amount": prior_std,
            "minutes_since_prev_txn": minutes_since_prev.to_numpy(),
            "is_new_category_for_account": new_category.to_numpy(),
        },
        index=srt.index,
    )
    return out.reindex(df.index)


def _rolling_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and sums over prior transactions in fixed time windows.

    `closed="left"` is what makes the window exclude the current row. Without
    it each transaction counts itself and the feature silently encodes
    "a transaction happened here", which is true of every row in the table.
    """
    srt = df.sort_values(["account_id", "timestamp"], kind="mergesort")
    indexed = srt.set_index("timestamp")
    rolled = indexed.groupby("account_id", sort=False)["amount"]

    count_1h = rolled.rolling("1h", closed="left").count().to_numpy()
    count_24h = rolled.rolling("24h", closed="left").count().to_numpy()
    sum_24h = rolled.rolling("24h", closed="left").sum().to_numpy()

    out = pd.DataFrame(
        {
            "txn_count_1h": count_1h,
            "txn_count_24h": count_24h,
            "amount_sum_24h": sum_24h,
        },
        index=srt.index,
    ).fillna(0.0)
    return out.reindex(df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready feature frame for a raw transaction log."""
    required = {"account_id", "timestamp", "amount", "merchant_category"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"transaction frame is missing columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    history = _account_history_features(df)
    windows = _rolling_window_features(df)

    ts = df["timestamp"].dt
    hour = ts.hour.astype("float64")

    feats = pd.DataFrame(index=df.index)
    feats["log_amount"] = np.log1p(df["amount"])
    feats["distance_from_home_km"] = df["distance_from_home_km"].astype("float64")
    feats["log_account_age_days"] = np.log1p(df["account_age_days"].astype("float64"))
    feats["hour"] = hour
    # Hour is cyclical: 23:00 and 00:00 are adjacent, which a raw integer hides
    # from any linear model.
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feats["day_of_week"] = ts.dayofweek.astype("float64")
    feats["is_night"] = hour.isin([23, 0, 1, 2, 3, 4]).to_numpy()
    feats["is_weekend"] = (ts.dayofweek >= 5).to_numpy()

    prior_mean = history["prior_mean_amount"]
    prior_std = history["prior_std_amount"]
    # The single most useful signal: this charge relative to what the account
    # normally spends. A $400 charge is unremarkable on one card and a red
    # flag on another.
    feats["amount_to_account_mean"] = np.where(
        prior_mean.to_numpy() > 0, df["amount"] / prior_mean, 1.0
    )
    feats["amount_zscore_account"] = np.where(
        prior_std.to_numpy() > 1e-6,
        (df["amount"] - prior_mean) / prior_std,
        0.0,
    )
    feats["prior_txn_count"] = history["prior_txn_count"]
    feats["minutes_since_prev_txn"] = history["minutes_since_prev_txn"].fillna(
        NO_PRIOR_MINUTES
    )
    feats["has_account_history"] = (history["prior_txn_count"] > 0).to_numpy()
    feats["is_new_category_for_account"] = history[
        "is_new_category_for_account"
    ].to_numpy()

    feats["txn_count_1h"] = windows["txn_count_1h"]
    feats["txn_count_24h"] = windows["txn_count_24h"]
    feats["amount_sum_24h"] = windows["amount_sum_24h"]

    feats["card_present"] = df["card_present"].astype(bool)
    feats["is_foreign"] = df["is_foreign"].astype(bool)
    feats["merchant_category"] = df["merchant_category"].astype("category")

    for col in BINARY_FEATURES:
        feats[col] = feats[col].astype(np.int8)

    return feats[ALL_FEATURES]


def make_preprocessor() -> ColumnTransformer:
    """Preprocessing for the linear baseline.

    Gradient-boosted trees need none of this -- they handle raw scales and
    take the categorical column natively -- so only the logistic regression
    is wrapped in it.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
