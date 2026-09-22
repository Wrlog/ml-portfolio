# Implicit-feedback recommender

Top-10 recommendations from a listening log, with implicit-ALS written from
scratch and evaluated on ranking metrics — reported next to the popularity
baseline, because that is the comparison that decides whether the model is
worth running.

> **The data is synthetic.** `src/data.py` simulates a listening log with a
> power-law catalogue, narrow per-user taste, and exposure bias. Every number
> below is reproducible by running the code; none of it is evidence about a
> real music service.

## The problem

Show each user ten things they have not played and would want to. The data is
**implicit**: plays, not ratings. That single fact drives everything else.

There are **no negatives**. A user who never played a track may dislike it, or
may never have been shown it. So:

- RMSE is meaningless — there is no rating to regress against.
- You cannot train a classifier on "played vs not played" without asserting
  that everything unplayed is disliked, which is false for ~99% of the
  catalogue.
- The output is an *ordering*, so the metrics are Recall@K, NDCG@K and MAP@K.

Hu, Koren & Volinsky's answer, implemented in `src/model.py`, is to fit every
cell of the matrix with a **confidence** that grows with the play count:

```
preference  p_ui = 1 if played, else 0
confidence  c_ui = 1 + alpha * r_ui
```

An unplayed cell gets `p = 0` at confidence 1 — a weak assertion, not a claim
of dislike. A track played twenty times gets `p = 1` at confidence 161.

## Results

4,000 users, 2,000 items, 220k plays (2.75% density). Each user's two most
recent plays held out; models rank the **full catalogue** with each user's own
history masked out. Reproduce with `python -m src.train`.

| Model | Recall@10 | NDCG@10 | MAP@10 | Coverage | Novelty | Gini | Fit time |
|---|---|---|---|---|---|---|---|
| **Item-kNN** | **0.1251** | **0.0887** | **0.0578** | 0.350 | 9.42 | 0.911 | 0.4s |
| ALS (from scratch) | 0.1136 | 0.0754 | 0.0470 | **0.569** | **9.64** | **0.833** | 12.8s |
| Popularity | 0.0527 | 0.0339 | 0.0202 | 0.017 | 8.11 | 0.994 | 0.0s |
| Random | 0.0044 | 0.0024 | 0.0012 | 1.000 | 11.24 | 0.124 | 0.0s |

Item-kNN is the most accurate, at **+137% recall over the popularity
baseline**. ALS is 9% behind on recall and touches **63% more of the
catalogue**.

### Why the popularity baseline is in the table

It recommends the same ten items to everybody, it took four lines to write, and
it more than doubles random. A great many published recommender improvements
disappear when measured against a properly tuned popularity baseline rather
than against random, so quoting a lift over random is how a recsys project
overstates itself.

It also sets the standard for "is this worth building". Item-kNN is 2.4× better
here — that is a real answer to a real question, and the code prints the lift
explicitly.

### Accuracy alone would pick the wrong model

Every accuracy metric here is maximised by recommending the head of the
catalogue to everyone. Popularity reaches Recall@10 of 0.053 while touching
**1.7% of the catalogue** and a Gini of 0.994 — near-total inequality of
exposure. As a product that is a chart, not a recommender: nothing gets
discovered, the long tail never earns anything, and the catalogue might as well
be 30 items.

So three distribution metrics sit next to the accuracy ones:

| Metric | What it catches |
|---|---|
| **Coverage** | Share of the catalogue that anyone ever sees |
| **Novelty** | Mean self-information of recommendations; low means recycling the chart |
| **Gini** | Inequality of exposure across items |

That reframes the choice at the top of the table. Item-kNN wins accuracy; ALS
covers 57% of the catalogue against kNN's 35% and spreads exposure
substantially more evenly. For a service whose value depends on the back
catalogue being findable, 9% of recall is a reasonable price for that — and
that is a product decision the metrics table is there to inform, not settle.

### The ALS implementation

Written out rather than imported, because the part worth understanding is the
part a library hides. Fitting all `n_users × n_items` cells is impossible
directly; the trick is that the normal-equation matrix splits into a term
shared by every user and a low-rank correction from that user's own items:

```
A_u = YtY + Y_u^T (C_u - I) Y_u + lambda*I
```

`YtY` is computed once per half-iteration, so cost scales with the number of
*observations*, not with the size of the matrix.

Two implementation notes that turned out to matter more than the factor count:

**Regularisation and alpha dominate.** With ~50 observations per user and 48
factors, the defaults most tutorials use (λ≈0.01, α=40) badly overfit each
user's own history — recall roughly halved. A sweep put the useful region at
λ=60, α=8, 32 factors. Large α is not "more signal", it is "more confident
about very little data".

**The system is solved by Cholesky, not `np.linalg.solve`.** `A` is symmetric
positive definite by construction, so Cholesky is the correct factorisation.
On the machine this was developed on it was also the difference between a fit
that finishes and one that does not: `np.linalg.solve` took **148 ms** per
48×48 system against **0.046 ms** for `cho_solve` — a pathology in that LAPACK
build, but this loop runs ~90,000 times.

The loss reported in `loss_history_` is the *full* objective including the
unobserved cells, computed via `trace((XᵀX)(YᵀY))` rather than by touching
every cell. An earlier version summed only observed cells, which is not the
quantity being minimised and did not fall monotonically — the convergence test
in `tests/test_models.py` caught it.

## The two evaluation traps

**1. Recommending items the user already has.** `recommend()` takes the
training matrix so it can mask them. Without the mask, a model that merely
re-ranks a user's own history scores near-perfectly offline and recommends
nothing anyone can act on. `tests/test_split.py` asserts the mask works *and*
demonstrates the inflation when it is switched off.

**2. Holding out a random item instead of the most recent one.** A random
hold-out lets the model train on what the user did *after* the item it is being
asked to predict. Recommenders run forwards in time, so the evaluation does
too.

Ranking is over the full 2,000-item catalogue rather than against sampled
negatives. Sampling a hundred negatives per user is much cheaper and is known
to be inconsistent — it can reverse the order of two models (Krichene & Rendle,
KDD 2020) — and at this catalogue size there is no reason to take the risk.

## What the simulation assumes

Two knobs control whether personalisation is worth doing at all, and both are
worth playing with:

| Flag | Default | Effect |
|---|---|---|
| `--exposure-strength` | 0.5 | How much of the observed log was driven by *what users were shown* rather than what they like. At 1.0 the log is pure popularity, and no model can beat the baseline. |
| `--taste-sharpness` | 5.0 | How sharply taste discriminates between items. |

`taste_sharpness` exists because the two effects are not on comparable scales:
a Zipf popularity curve spans several nats while a Dirichlet affinity spans
barely two, so an even 50/50 blend is in practice dominated by popularity. At
`--taste-sharpness 1`, popularity wins outright and **the honest conclusion is
not to build a model** — which is a real outcome worth being able to produce on
demand.

This is a simulation tuned to exhibit the regime a recommender is built for. It
is not evidence that ALS beats popularity on any particular real dataset.

## Running it

```bash
pip install -r ../requirements.txt

python -m src.train                          # full run, ~20s on a laptop CPU
python -m src.train --n-users 800 --n-items 600   # quick version
python -m src.data                           # catalogue and sparsity summary
python -m pytest tests -q                    # 37 tests, ~9s
```

Outputs land in `artifacts/`: `metrics.json`, and `evaluation.png` with
accuracy, the accuracy-vs-coverage trade-off, and accuracy by user activity
quartile.

## What this does not do

- **No cold start.** A user with no history gets zero factors from ALS and
  nothing sensible from kNN. A real system falls back to popularity or content
  features, and the interesting engineering is the handover.
- **No content features.** Everything here is collaborative. Genre, audio
  embeddings or text would help the tail items that collaborative signal never
  reaches.
- **No sequence.** The split is temporal but the models are not: they see a bag
  of items, not an order. Session-based models are the obvious next step and
  would change the architecture, not just the hyperparameters.
- **Exposure bias is simulated, not corrected.** The generator has the bias and
  the models inherit it. Inverse-propensity weighting or a debiased evaluation
  set would be the actual fix, and neither is here.
- **Offline metrics are not the product.** Nothing here measures whether a
  recommendation was *useful* — only whether it matched a play that already
  happened, which by construction cannot reward showing someone something they
  would never have found.
