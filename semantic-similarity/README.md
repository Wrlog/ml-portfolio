# Duplicate question detection, and the split that decides the score

Pairwise semantic similarity on Quora-style question pairs, built to
demonstrate one thing: **on this task the evaluation protocol moves the
reported number more than the model does.**

> **The data is synthetic.** `src/data.py` simulates the structure of the Quora
> Question Pairs dataset — a question bank reused across many pairs, 37%
> duplicates, and hard negatives that share most of their words. Every number
> below is reproducible by running the code; none of it is evidence about
> Quora's real corpus.

## The problem

Given two questions, decide whether they ask the same thing. In production this
is the second stage of a deduplication system: a retrieval step finds candidates
that are already *lexically similar*, and this model decides which of them are
genuinely duplicates.

That framing sets the difficulty. The model never sees an easy pair, because
easy pairs are filtered out before it. So the dataset here is 60% hard
negatives:

```
"How do I reset my password on Android?"
"How do I reset my password on iOS?"
```

90% word overlap, different answers, not duplicates.

## The result

Same data, same models, same code. The only difference is how the pairs were
split into train and test.

| Model | Random pair split | Cluster-disjoint split | Inflation |
|---|---|---|---|
| TF-IDF cosine | 0.702 | 0.726 | −0.024 |
| Logistic regression | 0.786 | 0.794 | −0.008 |
| **LightGBM** | **0.886** | **0.788** | **+0.098** |
| LightGBM + question target encoding | 0.859 | 0.786 | +0.073 |

(ROC-AUC. 40,000 pairs over 900 intents; reproduce with `python -m src.train`.)

**The leaky protocol does not just inflate the headline number — it changes
which model you would pick.** Under the random split, LightGBM beats logistic
regression by 10 points and is the obvious choice. Under the honest split the
two are within noise of each other (0.788 vs 0.794), and the simpler, faster,
more interpretable model is at least as good.

Anyone reporting the first table ships the wrong model *and* promises a number
they will miss by ten points.

### Why the gap is largest for the most flexible model

This was not what I expected going in. The obvious culprit is the explicit
identity-keyed feature — target encoding on question text — and it does inflate
(+0.073). But **plain LightGBM inflates more (+0.098) with no identity feature
at all.**

The mechanism: duplicate pairs concentrate on popular topics (`src/data.py`
samples them with a power-law weight, as the real dataset was assembled —
heavily asked questions accumulate restatements). A model with enough capacity
can partially *recognise the topic* from lexical statistics and exploit
"questions like this one tend to be duplicates". When clusters straddle the
split, that shortcut scores; when they do not, it is worthless.

Logistic regression is too constrained to find the shortcut, which is why it
shows no gap. **Model capacity determines exposure to the leak** — so the
naive protocol systematically favours the models most able to cheat.

## Why the split is hard to do right

A question does not appear once. Each question text here sits in ~15 pairs, as
in the real dataset, which is a *graph over a finite question set* rather than
a list of independent examples.

The textbook fix — partition the graph into connected components and assign
whole components — **does not work**, and finding out why is the instructive
part. Negative pairs join arbitrary clusters, so the graph collapses:

```
cluster graph: 1 connected component(s); largest covers 100.0% of pairs
```

6,000 pairs over 200 clusters are already one blob. No partition preserves
every pair.

So `cluster_disjoint_split` assigns *clusters* to sides and **discards** the
pairs that straddle the boundary. That is a real cost, reported rather than
hidden:

- **37.9% of pairs are dropped** at a 25% test size;
- the drops are almost entirely negatives (a duplicate pair joins a cluster to
  itself and can never straddle), so both sides come out far more
  duplicate-heavy — the test side lands near **70%** against an original 37%.

That second effect is not cosmetic. PR-AUC and accuracy both move with the base
rate, so leaving it uncorrected would confound the protocol effect with a
base-rate effect and invalidate the entire comparison. `_rebalance` trims
surplus positives on each side back to the original 37%, at the cost of more
data, so the two protocols are scored on like for like.

Verified mechanically rather than asserted:

```python
assert_no_question_overlap(df, split)   # raises on the random split
```

## Features

No GPU, no transformers. A cross-encoder would be more accurate and is the
right answer when latency allows; this is the cheap filter that runs first over
millions of candidates.

Sixteen symmetric pair features: TF-IDF cosine over word and character n-grams,
Jaccard over raw and content tokens, IDF-weighted overlap, and length
statistics. Two design rules:

- **Fit on training questions only.** The vectorisers are a fit/transform
  object, not a function, so test-set vocabulary and document frequencies
  cannot reach the model.
- **Every feature is symmetric**: `f(q1, q2) == f(q2, q1)`. Question order in
  these datasets is arbitrary, so `length_of_question_1` teaches the model an
  artefact of how rows were written down. Where direction matters, both the min
  and the max go in. `tests/test_features.py` asserts this by swapping the
  columns and comparing frames.

The feature carrying the hard negatives is `unique_to_shorter` — the share of
content words present in only one question. For *Android* vs *iOS* it is small
in absolute terms and holds the entire meaning.

## Running it

```bash
pip install -r ../requirements.txt

python -m src.train                        # both protocols, ~50s on a laptop CPU
python -m src.train --n-pairs 12000        # quick version, ~30s
python -m src.data                         # print a sample of generated pairs
python -m pytest tests -q                  # 35 tests, ~34s
```

| Flag | Default | Effect |
|---|---|---|
| `--hard-negative-share` | 0.6 | Share of negatives sharing two of three intent slots. Drop it to 0 and every model jumps ~15 points — which is what a benchmark built only on random negatives is measuring. |
| `--n-intents` | 900 | Distinct meanings. Fewer intents means denser clusters and a bigger leak. |
| `--test-size` | 0.25 | Cluster budget for the test side, before boundary pairs are dropped. |

## What this does not do

- **No cross-encoder, no embeddings.** A fine-tuned transformer would beat
  0.79 ROC-AUC substantially. The protocol finding applies to it too, and more
  so — higher capacity, more exposure to the shortcut.
- **The hard negatives are lexical, not semantic.** They differ by a slot
  substitution, so a model that understood negation, tense or quantifiers would
  gain nothing here that it would gain on real data.
- **Some generated questions are semantically odd** ("get in shape ETFs"),
  because the three intent slots are sampled independently. The matching task
  stays well posed — the label depends only on whether two questions came from
  the same intent — but it means this corpus cannot evaluate anything that
  relies on real-world plausibility.
- **Calibration is not fixed, only measured.** `calibration_table` shows the
  LightGBM scores drift from the diagonal. Used as a ranked filter that does
  not matter; shown to a moderator as "87% likely duplicate" it does, and the
  fix is isotonic regression on a validation slice.
