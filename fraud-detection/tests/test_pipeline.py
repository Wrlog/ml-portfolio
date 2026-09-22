"""Generator sanity checks and an end-to-end training smoke test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import GeneratorConfig, generate_transactions
from src.evaluate import CostModel
from src.features import build_features
from src.model import RuleBaseline, build_models, positive_class_weight
from src.train import run, temporal_split


def test_generator_hits_the_requested_fraud_rate() -> None:
    cfg = GeneratorConfig(n_accounts=800, n_legit=20_000, fraud_rate=0.005, seed=5)
    df = generate_transactions(cfg)
    assert df["is_fraud"].mean() == pytest.approx(cfg.fraud_rate, rel=0.15)


def test_generator_is_time_sorted_and_reproducible() -> None:
    cfg = GeneratorConfig(n_accounts=200, n_legit=2_000, seed=42)
    first = generate_transactions(cfg)
    second = generate_transactions(cfg)

    assert first["timestamp"].is_monotonic_increasing
    pd.testing.assert_frame_equal(first, second)


def test_separability_controls_difficulty() -> None:
    """At separability 0 fraud is drawn from the legitimate distribution.

    The generator is only useful if this knob does what it claims, so the
    claim is tested rather than asserted in a docstring.
    """
    easy = generate_transactions(
        GeneratorConfig(n_accounts=600, n_legit=12_000, fraud_rate=0.02,
                        separability=1.0, seed=9)
    )
    hard = generate_transactions(
        GeneratorConfig(n_accounts=600, n_legit=12_000, fraud_rate=0.02,
                        separability=0.0, seed=9)
    )

    def foreign_gap(df: pd.DataFrame) -> float:
        by_class = df.groupby("is_fraud")["is_foreign"].mean()
        return float(by_class.loc[1] - by_class.loc[0])

    assert foreign_gap(easy) > foreign_gap(hard)
    assert abs(foreign_gap(hard)) < 0.05


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="separability"):
        GeneratorConfig(separability=1.5)
    with pytest.raises(ValueError, match="fraud_rate"):
        GeneratorConfig(fraud_rate=0.0)


def test_temporal_split_does_not_overlap_in_time() -> None:
    df = generate_transactions(GeneratorConfig(n_accounts=400, n_legit=6_000, seed=13))
    X = build_features(df)
    split = temporal_split(df, X)

    ts = pd.to_datetime(df["timestamp"])
    train_max = ts[split.X_train.index].max()
    valid_min = ts[split.X_valid.index].min()
    valid_max = ts[split.X_valid.index].max()
    test_min = ts[split.X_test.index].min()

    assert train_max <= valid_min
    assert valid_max <= test_min
    assert len(split.X_train) + len(split.X_valid) + len(split.X_test) == len(df)


def test_positive_class_weight_balances_the_classes() -> None:
    y = np.array([0] * 990 + [1] * 10)
    assert positive_class_weight(y) == pytest.approx(99.0)
    with pytest.raises(ValueError, match="no positive examples"):
        positive_class_weight(np.zeros(10))


def test_rule_baseline_scores_are_probabilities() -> None:
    df = generate_transactions(GeneratorConfig(n_accounts=200, n_legit=2_000, seed=17))
    X = build_features(df)
    proba = RuleBaseline().fit(X, df["is_fraud"]).predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_every_model_fits_and_ranks_better_than_chance() -> None:
    from sklearn.metrics import roc_auc_score

    df = generate_transactions(
        GeneratorConfig(n_accounts=1_200, n_legit=25_000, fraud_rate=0.01, seed=21)
    )
    X = build_features(df)
    split = temporal_split(df, X)

    for name, model in build_models(pos_weight=positive_class_weight(split.y_train)).items():
        model.fit(split.X_train, split.y_train)
        scores = model.predict_proba(split.X_test)[:, 1]
        assert roc_auc_score(split.y_test, scores) > 0.6, f"{name} ranks no better than chance"


def test_run_produces_a_metrics_file(tmp_path) -> None:
    summary = run(
        cfg=GeneratorConfig(n_accounts=600, n_legit=12_000, fraud_rate=0.01, seed=23),
        costs=CostModel(),
        output_dir=tmp_path,
        make_plots=False,
    )

    assert (tmp_path / "metrics.json").exists()
    assert summary["best_model"] in summary["models"]
    for result in summary["models"].values():
        assert 0.0 <= result["ranking"]["pr_auc"] <= 1.0
        assert result["operating_point"]["alerts"] >= 0
