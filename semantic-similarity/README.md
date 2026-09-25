# Duplicate question detection

Pairwise semantic similarity on Quora-style question pairs. On this task, how
you split train and test changes the score more than which model you use.

The data is synthetic. `src/data.py` mimics the structure of the Quora Question
Pairs dataset: a question bank reused across many pairs, 37% duplicates, and
hard negatives that share most of their words. The numbers below come from
running the code and say nothing about Quora's real corpus.

## The problem

Given two questions, decide whether they ask the same thing. In a
deduplication system this runs after a retrieval step that has already found
*lexically similar* candidates, so easy pairs never reach it. 60% of the
negatives here are hard ones like:

```
"How do I reset my password on Android?"
"How do I reset my password on iOS?"
```

90% word overlap, different answers, not duplicates.

## Results

Same data, models and code. Only the train/test split changes.

| Model | Random pair split | Cluster-disjoint split | Inflation |
|---|---|---|---|
| TF-IDF cosine | 0.702 | 0.726 | −0.024 |
| Logistic regression | 0.786 | 0.794 | −0.008 |
| **LightGBM** | **0.886** | **0.788** | **+0.098** |
| LightGBM + question target encoding | 0.859 | 0.786 | +0.073 |

(ROC-AUC. 40,000 pairs over 900 intents; reproduce with `python -m src.train`.)

The random split also changes which model you'd pick. There, LightGBM beats
logistic regression by 10 points. With the cluster-disjoint split they're
within noise (0.788 vs 0.794), so you'd ship the wrong model and miss the
promised score by about ten points.

I expected the target encoding on question text to be the main leak. It does
inflate (+0.073), but plain LightGBM inflates more (+0.098) with no identity
feature at all. Duplicate pairs concentrate on popular topics (`src/data.py`
samples them with a power-law weight, like the real dataset, where frequently
asked questions pick up more restatements). A flexible model can partly
recognise the topic from lexical statistics and learn that questions like this
one tend to be duplicates. That only works when clusters straddle the split.
Logistic regression is too constrained to find the shortcut, so it shows no
gap. The more flexible the model, the bigger the inflation.

## The split

Each question text appears in about 15 pairs, as in the real dataset, so the
data is a graph over a finite set of questions rather than a list of
independent examples.

The usual fix is to assign whole connected components to train or test. That
fails here because negative pairs link arbitrary clusters, and the graph
collapses into one component:

```
cluster graph: 1 connected component(s); largest covers 100.0% of pairs
```

Even 6,000 pairs over 200 clusters form a single blob. So
`cluster_disjoint_split` assigns clusters to each side and drops the pairs that
straddle the boundary:

- 37.9% of pairs are dropped at a 25% test size.
- Nearly all of them are negatives, since a duplicate pair links a cluster to
  itself and can't straddle. Both sides become much more duplicate-heavy; the
  test side lands near 70% against the original 37%.

PR-AUC and accuracy depend on the base rate, so `_rebalance` trims surplus
positives on each side back to 37% (losing more data) to keep the comparison
fair. A check for leakage:

```python
assert_no_question_overlap(df, split)   # raises on the random split
```

## Features

No GPU or transformers. A cross-encoder would be more accurate when latency
allows; this is the cheap filter that runs first over millions of candidates.

Sixteen symmetric pair features: TF-IDF cosine over word and character
n-grams, Jaccard over raw and content tokens, IDF-weighted overlap, and length
statistics.

- The vectorisers are a fit/transform object fitted on training questions
  only, so test-set vocabulary and document frequencies can't reach the model.
- Every feature satisfies `f(q1, q2) == f(q2, q1)`. Question order is
  arbitrary, so something like `length_of_question_1` would teach the model an
  artefact of how the rows were written. Where direction matters, both the min
  and max go in. `tests/test_features.py` checks this by swapping the columns
  and comparing frames.

The hard negatives mostly come down to `unique_to_shorter`, the share of
content words found in only one question. For *Android* vs *iOS* it's small,
but it's the whole difference in meaning.

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
| `--hard-negative-share` | 0.6 | Share of negatives sharing two of three intent slots. Drop it to 0 and every model jumps ~15 points, which is what a benchmark built only on random negatives is measuring. |
| `--n-intents` | 900 | Distinct meanings. Fewer intents means denser clusters and a bigger leak. |
| `--test-size` | 0.25 | Cluster budget for the test side, before boundary pairs are dropped. |

## Limitations

- No cross-encoder or embeddings. A fine-tuned transformer would do well above
  0.79 ROC-AUC, and with more capacity it would be even more exposed to the
  leak.
- The hard negatives are lexical (one slot substituted), so understanding
  negation, tense or quantifiers wouldn't help here the way it would on real
  data.
- Some generated questions are odd ("get in shape ETFs") because the three
  intent slots are sampled independently. The label only depends on shared
  intent, so the task still works, but nothing that needs real-world
  plausibility can be tested.
- Calibration is measured, not fixed. `calibration_table` shows the LightGBM
  scores drifting off the diagonal. That doesn't matter for a ranked filter,
  but it does if a moderator sees "87% likely duplicate". Isotonic regression
  on a validation slice would fix it.
