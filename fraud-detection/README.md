# Card fraud detection under extreme class imbalance

Catching fraudulent card transactions when 1 in 600 is fraud, and setting the
alert threshold by pricing the two kinds of mistake instead of defaulting to
0.5.

The data is synthetic. `src/data.py` simulates an issuer's transaction log, so
the numbers below are reproducible but say nothing about a real payments
portfolio. I didn't use the public Kaggle dataset because it's 28 anonymised
PCA components, which leaves no feature engineering to show or explain to a
stakeholder.

## The problem

An issuer approves or declines each card transaction in real time. Declining
a good customer costs support time and some churn risk. Approving a fraud
costs the full amount plus a chargeback fee. The costs differ, and they vary
per transaction because amounts span two orders of magnitude: missing a
$2,400 charge is much worse than missing an $8 one. So the threshold matters
as much as the classifier.

At a 0.17% base rate the usual metrics don't help. Approving everything gets
99.83% accuracy. ROC-AUC is dominated by the 99.8% of the curve nobody
operates on; the rule baseline below scores 0.966 and is still the worst
model by a wide margin. The 0.5 cut-off is arbitrary, and the best model here
operates at 0.012. The report leads with PR-AUC, quoted as a lift over the
base rate (its floor), and then a cost curve that turns scores into dollars.

## Results

240,409 transactions over 180 simulated days. Trained on the first 60% of the
calendar, threshold tuned on the next 20%, all numbers below from the
untouched final 20% (48,082 transactions, 97 frauds). Reproduce with
`python -m src.train`.

| Model | PR-AUC | Lift over base rate | ROC-AUC | Precision @ threshold | Recall @ threshold | Net savings |
|---|---|---|---|---|---|---|
| Rule baseline (no learning) | 0.301 | 149× | 0.966 | 43.0% | 38.1% | $6,553 (46%) |
| Logistic regression | 0.506 | 251× | 0.991 | 44.2% | 51.5% | $7,541 (53%) |
| **LightGBM** | **0.754** | **374×** | **0.996** | **70.7%** | **67.0%** | **$8,537 (60%)** |
| LightGBM + `scale_pos_weight` | 0.399 | 198× | 0.883 | 54.5% | 62.9% | $8,392 (59%) |

Net savings are relative to approving everything, which would have cost
$14,221 on this test set. The threshold was fixed on validation data; choosing
it on the test set would be training on the test set.

With a review team that can work 15 cases a day (543 over the test window),
16.0% of the cases LightGBM sends to an analyst are real fraud, and it catches
89.7% of all fraud.

The rule baseline is the fair comparison, and it holds up. A seven-rule
if/else engine (foreign, card-not-present, large relative to history, night,
velocity, distance, risky merchant category) recovers 46% of losses on its
own. No fraud team runs a random model, so a lift over random means little.
LightGBM is worth deploying because it beats the rules by 14 points of
recovered loss.

`scale_pos_weight` made things worse. It's the standard reflex for imbalance,
and here it cost 35 points of PR-AUC (0.754 → 0.399) and 11 of ROC-AUC. The
balancing weight at this base rate is about 590×, so a few hundred positives
dominate every split and the trees memorise those frauds. The imbalance is
really a question of where to cut the score, and `evaluate.py` handles that
by pricing the errors.

## Features

Every feature uses the transaction's own fields plus strictly earlier
transactions on the same account. The obvious "average spend for this
account" is a `groupby` mean over the whole table, which leaks the future
into the past: offline metrics improve and production performance collapses.

The features doing most of the work:

| Feature | Why it works |
|---|---|
| `amount_to_account_mean` | A $400 charge is unremarkable on one card and a red flag on another. Absolute amount alone cannot express this. |
| `txn_count_1h`, `txn_count_24h` | Fraud arrives in bursts. A compromised card is tested, then drained over minutes. |
| `minutes_since_prev_txn` | Separates the burst pattern from ordinary repeat spend. |
| `is_new_category_for_account` | First-ever gift-card purchase on a 3-year-old grocery-and-fuel card. |
| `hour_sin` / `hour_cos` | Hour is cyclical; as a raw integer, 23:00 and 00:00 look 23 apart. |

`tests/test_features.py` checks for leakage directly: features recomputed on a
truncated log must reproduce the existing rows exactly. It also checks the
vectorised time-window counts against a brute-force O(n²) version.

## Cost model

```python
CostModel(
    loss_given_fraud=1.0,        # share of the amount the issuer eats
    chargeback_fee=25.00,        # fixed admin cost per confirmed loss
    review_cost=3.50,            # analyst time per alert raised
    false_positive_friction=12.00,  # blocked good customer: support + churn risk
)
```

These are order-of-magnitude assumptions, not measurements. They're in one
dataclass so a fraud lead can change them and get a new operating point
without touching the model. Raising `false_positive_friction` from $12 to $40
moves the threshold up and the alert volume down, and the code says by how
much.

The search is also capped by review capacity (`--max-alert-rate`, default 1%
of traffic). Uncapped, the cost optimum alerts on 3% of transactions, far
more than a team working 15 cases a day can handle.

## Running it

```bash
pip install -r ../requirements.txt

python -m src.train                 # full run, ~25s on a laptop CPU
python -m src.train --n-legit 40000 # quick version
python -m pytest tests -q           # 24 tests, ~26s
```

Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--separability` | 0.62 | How distinguishable attacks are from normal spend, in [0, 1]. At 0.0 fraud is drawn from the legitimate distribution and no model can beat chance, which is a useful check that the pipeline is not manufacturing signal. |
| `--max-alert-rate` | 0.01 | Review capacity as a share of transactions. |
| `--fraud-rate` | 0.0017 | Base rate. Raise it to see the metrics stop being interesting. |

Outputs go to `artifacts/`: `metrics.json` with every number above, and
`evaluation.png` with the precision-recall and cost curves.

## Limitations

- No calibration. The LightGBM scores rank well but aren't probabilities. The
  cost analysis only needs the ordering, but showing an analyst "probability
  this is fraud" would need Platt scaling or isotonic regression first.
- No account grouping in the split. The temporal split mostly keeps an attack
  burst on one side, but an account can appear in both train and test. That's
  right here (production models score returning customers) and wrong for a
  model meant to generalise to unseen accounts.
- No drift handling. Fraud patterns move, so a real deployment needs scheduled
  retraining and a population-stability check on the features.
- The feature importances reflect the generator, not payments. The pipeline,
  leakage tests and cost framing are what transfer.
