# Implicit-feedback recommender

Top-10 recommendations from a listening log. I wrote implicit ALS from scratch
and compared it against item-kNN and a popularity baseline on ranking metrics.

The data is synthetic. `src/data.py` simulates a listening log with a
power-law catalogue, narrow per-user taste and exposure bias. The numbers below
come from running the code and say nothing about a real music service.

## The problem

Show each user ten tracks they haven't played and would want to. The data is
implicit (plays, not ratings), so there are no negatives: a user who never
played a track might dislike it or might never have seen it. That means:

- RMSE doesn't apply, since there's no rating to regress against.
- A "played vs not played" classifier would assume everything unplayed is
  disliked, which is false for ~99% of the catalogue.
- The output is an ordering, so the metrics are Recall@K, NDCG@K and MAP@K.

`src/model.py` follows Hu, Koren & Volinsky and fits every cell of the matrix
with a confidence that grows with play count:

```
preference  p_ui = 1 if played, else 0
confidence  c_ui = 1 + alpha * r_ui
```

An unplayed cell gets `p = 0` at confidence 1, a weak signal rather than a
claim of dislike. A track played twenty times gets `p = 1` at confidence 161.

## Results

4,000 users, 2,000 items, 220k plays (2.75% density). Each user's two most
recent plays are held out, and models rank the full catalogue with the user's
own history masked. Reproduce with `python -m src.train`.

| Model | Recall@10 | NDCG@10 | MAP@10 | Coverage | Novelty | Gini | Fit time |
|---|---|---|---|---|---|---|---|
| **Item-kNN** | **0.1251** | **0.0887** | **0.0578** | 0.350 | 9.42 | 0.911 | 0.4s |
| ALS (from scratch) | 0.1136 | 0.0754 | 0.0470 | **0.569** | **9.64** | **0.833** | 12.8s |
| Popularity | 0.0527 | 0.0339 | 0.0202 | 0.017 | 8.11 | 0.994 | 0.0s |
| Random | 0.0044 | 0.0024 | 0.0012 | 1.000 | 11.24 | 0.124 | 0.0s |

Item-kNN is the most accurate, with +137% recall over popularity. ALS is 9%
behind on recall but covers 63% more of the catalogue.

Popularity recommends the same ten items to everyone, took four lines to
write, and still more than doubles random. Plenty of published recommender
gains vanish against a well-tuned popularity baseline, so the code prints the
lift over popularity (2.4× for Item-kNN), not over random.

Accuracy alone favours recommending the head of the catalogue to everyone.
Popularity gets Recall@10 of 0.053 while showing only 1.7% of the catalogue,
with a Gini of 0.994, as if the catalogue were 30 items. So the table also has
three distribution metrics:

| Metric | What it catches |
|---|---|
| **Coverage** | Share of the catalogue that anyone ever sees |
| **Novelty** | Mean self-information of recommendations; low means recycling the chart |
| **Gini** | Inequality of exposure across items |

On those, ALS covers 57% of the catalogue against kNN's 35% and spreads
exposure much more evenly. If the back catalogue matters to the service,
giving up 9% of recall for that seems reasonable, but it's a product call.

## ALS implementation

I wrote this myself rather than using a library. Fitting all
`n_users × n_items` cells directly isn't feasible, but each user's
normal-equation matrix splits into a term shared by all users plus a low-rank
correction from that user's own items:

```
A_u = YtY + Y_u^T (C_u - I) Y_u + lambda*I
```

`YtY` is computed once per half-iteration, so cost scales with the number of
observations rather than the size of the matrix.

Regularisation and alpha mattered more than the factor count. With ~50
observations per user and 48 factors, the usual tutorial defaults (λ≈0.01,
α=40) badly overfit each user's history and roughly halved recall. A sweep put
the useful region at λ=60, α=8, 32 factors. A large α just makes the model
more confident about very little data.

`A` is symmetric positive definite, so I solve it with Cholesky rather than
`np.linalg.solve`. On my development machine this decided whether the fit
finished at all: `np.linalg.solve` took 148 ms per 48×48 system against
0.046 ms for `cho_solve`. That's a quirk of that LAPACK build, but the loop
runs ~90,000 times.

`loss_history_` records the full objective including unobserved cells,
computed via `trace((XᵀX)(YᵀY))` without touching every cell. An earlier
version summed only observed cells, which isn't what's being minimised and
didn't fall monotonically; the convergence test in `tests/test_models.py`
caught it.

## Evaluation

`recommend()` takes the training matrix so it can mask items the user already
has. Without the mask, re-ranking a user's own history scores near-perfectly
offline. `tests/test_split.py` checks the mask and shows the inflation when
it's off.

The hold-out is each user's most recent plays, not random ones. A random
hold-out lets the model train on what the user did after the item it's asked
to predict.

Ranking is over the full 2,000-item catalogue rather than sampled negatives.
Sampling a hundred negatives per user is cheaper but inconsistent and can
reverse the order of two models (Krichene & Rendle, KDD 2020). At this
catalogue size there's no need.

## Simulation settings

Two flags control whether personalisation is worth doing:

| Flag | Default | Effect |
|---|---|---|
| `--exposure-strength` | 0.5 | How much of the observed log was driven by *what users were shown* rather than what they like. At 1.0 the log is pure popularity, and no model can beat the baseline. |
| `--taste-sharpness` | 5.0 | How sharply taste discriminates between items. |

`taste_sharpness` exists because the two effects are on different scales. A
Zipf popularity curve spans several nats and a Dirichlet affinity barely two,
so a 50/50 blend is dominated by popularity. At `--taste-sharpness 1`
popularity wins outright and the right answer is not to build a model.

The defaults are tuned to the regime a recommender is built for, so none of
this shows that ALS beats popularity on any particular real dataset.

## Running it

```bash
pip install -r ../requirements.txt

python -m src.train                          # full run, ~20s on a laptop CPU
python -m src.train --n-users 800 --n-items 600   # quick version
python -m src.data                           # catalogue and sparsity summary
python -m pytest tests -q                    # 37 tests, ~9s
```

Outputs go to `artifacts/`: `metrics.json`, and `evaluation.png` with
accuracy, the accuracy-vs-coverage trade-off, and accuracy by user activity
quartile.

## Limitations

- No cold start. A user with no history gets zero factors from ALS and nothing
  sensible from kNN. A real system would fall back to popularity or content
  features.
- No content features. Genre, audio embeddings or text would help tail items
  that collaborative signal never reaches.
- No sequence. The split is temporal but the models see an unordered bag of
  items. Session-based models would be the next step and need a different
  architecture.
- Exposure bias is simulated but not corrected. Inverse-propensity weighting
  or a debiased evaluation set would address it; neither is implemented.
- Offline metrics only check whether a recommendation matched a play that
  already happened, so they can't reward real discovery.
