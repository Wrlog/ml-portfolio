# Card fraud detection under extreme class imbalance

Catching fraudulent card transactions when 1 in 600 is fraudulent, and deciding
where to set the alert threshold by pricing the two kinds of mistake instead of
defaulting to 0.5.

> **The data is synthetic.** `src/data.py` simulates an issuer's transaction
> log. Every number below is reproducible by running the code, but none of it
> is evidence about a real payments portfolio. The generator exists because the
> public Kaggle dataset is 28 anonymised PCA components, which makes feature
> engineering impossible to demonstrate and impossible to explain to a
> stakeholder.

## The problem

An issuer approves or declines a card transaction in real time. Declining a
good customer costs support time and some churn risk; approving a fraudulent
charge costs the full amount plus a chargeback fee. Those costs are not equal,
and they are not even constant per transaction — the amount varies by two
orders of magnitude, so a missed $2,400 charge and a missed $8 charge are not
the same error.

That makes the threshold, not the classifier, the deliverable.

## Why the usual metrics are useless here

At a 0.17% base rate:

- **Accuracy** is 99.83% for a model that approves everything.
- **ROC-AUC** is dominated by the 99.8% of the curve nobody ever operates on.
  The rule baseline below scores 0.966 ROC-AUC and is still the worst model in
  the table by a wide margin.
- **The 0.5 cut-off** is arbitrary. The best model here operates at a threshold
  of 0.012.

So the report leads with **PR-AUC** (quoted as a lift over the base rate, which
is its floor), and then with a **cost curve** that converts scores into dollars.

## Results

240,409 transactions over 180 simulated days; trained on the first 60% of the
calendar, threshold tuned on the next 20%, all numbers below from the untouched
final 20% (48,082 transactions, 97 frauds). Reproduce with `python -m src.train`.

| Model | PR-AUC | Lift over base rate | ROC-AUC | Precision @ threshold | Recall @ threshold | Net savings |
|---|---|---|---|---|---|---|
| Rule baseline (no learning) | 0.301 | 149× | 0.966 | 43.0% | 38.1% | $6,553 (46%) |
| Logistic regression | 0.506 | 251× | 0.991 | 44.2% | 51.5% | $7,541 (53%) |
| **LightGBM** | **0.754** | **374×** | **0.996** | **70.7%** | **67.0%** | **$8,537 (60%)** |
| LightGBM + `scale_pos_weight` | 0.399 | 198× | 0.883 | 54.5% | 62.9% | $8,392 (59%) |

"Net savings" is against approving every transaction, which would have cost
$14,221 on this test set. The threshold was chosen on validation data and then
frozen — picking it on the test set is the quiet version of training on it.

Under a review team that can work 15 cases a day (543 over the test window),
LightGBM puts a real fraud in front of an analyst **16.0%** of the time and
catches **89.7%** of all fraud.

### Two results worth reading twice

**The rule baseline is the honest comparison, and it is not bad.** A
seven-rule if/else engine — foreign, card-not-present, large relative to
history, night, velocity, distance, risky merchant category — recovers 46% of
the losses on its own. A portfolio project that quotes a lift over *random* is
measuring against something no fraud team has ever run. The gradient boosting
model is worth deploying because it beats the incumbent by 14 points of
recovered loss, not because it beats a coin flip.

**`scale_pos_weight` made the model worse.** Class reweighting is the reflex
answer to imbalance, and here it cost 35 points of PR-AUC (0.754 → 0.399) and
11 points of ROC-AUC. At this base rate the balancing weight is ~590×, so a few
hundred positives dominate every split and the trees fit those specific frauds
rather than the pattern. Imbalance is a problem with *where you cut the score*,
not with the loss function, and it is fixed in `evaluate.py` by pricing the
errors.

## Features

Every feature is computed from a transaction's own fields plus **strictly
earlier** transactions on the same account. The obvious implementation of
"average spend for this account" is a `groupby` mean over the whole table, and
it leaks the future into the past: offline metrics improve, production
performance collapses.

The features that carry the model:

| Feature | Why it works |
|---|---|
| `amount_to_account_mean` | A $400 charge is unremarkable on one card and a red flag on another. Absolute amount alone cannot express this. |
| `txn_count_1h`, `txn_count_24h` | Fraud arrives in bursts. A compromised card is tested, then drained over minutes. |
| `minutes_since_prev_txn` | Separates the burst pattern from ordinary repeat spend. |
| `is_new_category_for_account` | First-ever gift-card purchase on a 3-year-old grocery-and-fuel card. |
| `hour_sin` / `hour_cos` | Hour is cyclical; as a raw integer, 23:00 and 00:00 look 23 apart. |

`tests/test_features.py` asserts the no-leakage property directly: features
recomputed on a truncated log must exactly reproduce the rows that were already
there. It also checks the vectorised time-window counts against a brute-force
O(n²) implementation.

## The cost model

```python
CostModel(
    loss_given_fraud=1.0,        # share of the amount the issuer eats
    chargeback_fee=25.00,        # fixed admin cost per confirmed loss
    review_cost=3.50,            # analyst time per alert raised
    false_positive_friction=12.00,  # blocked good customer: support + churn risk
)
```

These are order-of-magnitude assumptions, not measurements. They live in one
dataclass precisely so a fraud lead can change them and re-derive the operating
point without touching the model — which is the point of the exercise. Changing
`false_positive_friction` from $12 to $40 moves the optimal threshold up and
the alert volume down, and the code will tell you by how much.

The threshold search is also capacity-constrained (`--max-alert-rate`, default
1% of traffic). The unconstrained cost optimum will happily alert on 3% of
transactions and flood a team that can work 15 cases a day.

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
| `--separability` | 0.62 | How distinguishable attacks are from normal spend, in [0, 1]. At 0.0 fraud is drawn from the legitimate distribution and no model can beat chance — a useful check that the pipeline is not manufacturing signal. |
| `--max-alert-rate` | 0.01 | Review capacity as a share of transactions. |
| `--fraud-rate` | 0.0017 | Base rate. Raise it to see the metrics stop being interesting. |

Outputs land in `artifacts/`: `metrics.json` with every number above, and
`evaluation.png` with the precision-recall curves and the cost curve.

## What this does not do

- **No calibration layer.** The LightGBM scores rank well but are not
  calibrated probabilities; the cost analysis only needs the ordering, but a
  "probability this is fraud" shown to an analyst would need Platt scaling or
  isotonic regression first.
- **No account-level grouping in the split.** The split is temporal, which
  keeps a single attack burst from straddling train and test in most cases, but
  an account can appear on both sides of the boundary. That is correct here
  (production models score returning customers) but it would be wrong for a
  model meant to generalise to unseen accounts.
- **No drift handling.** Fraud patterns move; a real deployment needs scheduled
  retraining and a population-stability check on the feature distributions.
- **Synthetic data means the feature importances are an artefact of the
  generator**, not a finding about payments. The pipeline, the leakage tests
  and the cost framing are what transfer.
