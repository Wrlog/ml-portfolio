"""Synthetic duplicate-question pairs.

Modelled on the structure of the Quora Question Pairs task: two questions, one
binary label for "these ask the same thing". The generator reproduces three
properties of that dataset that drive every downstream decision.

**1. A question bank, not independently generated rows.** Surface forms are
rendered once per intent into a fixed bank, and pairs are then sampled *from
that bank*. So a single question text participates in a dozen or more pairs,
exactly as in Quora, where the dataset is a graph over a finite question set
rather than a list of independent examples.

This is the property that makes the splitting protocol matter, and it is worth
stating plainly because getting it wrong invalidates the experiment: an earlier
version of this generator rendered every question independently, so almost no
text repeated, no question could appear on both sides of a split, and the leaky
and clean protocols produced identical scores. The leak needs something to leak.

**2. Hard negatives.** The interesting non-duplicates are not random pairs;
they are questions that share most of their words and differ in the one that
matters:

    "How do I reset my password on Android?"
    "How do I reset my password on iOS?"

Lexical overlap is ~90% and the answer is different. A generator that only
emits random negatives produces a task any bag-of-words model solves, and a
reported accuracy that means nothing.

**3. Class imbalance in the honest direction.** Roughly 37% of pairs are
duplicates, matching the published Quora rate, so accuracy has a non-trivial
majority-class floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACTIONS = {
    "reset": ["reset", "change", "update", "recover"],
    "install": ["install", "set up", "configure", "get running"],
    "learn": ["learn", "get good at", "study", "pick up"],
    "invest": ["invest in", "buy", "put money into", "start investing in"],
    "lose_weight": ["lose weight", "get in shape", "burn fat", "slim down"],
    "migrate": ["migrate", "move", "port", "transfer"],
    "debug": ["debug", "troubleshoot", "fix", "diagnose"],
    "deploy": ["deploy", "ship", "release", "roll out"],
    "optimise": ["speed up", "optimise", "make faster", "improve performance of"],
    "back_up": ["back up", "archive", "snapshot", "make a copy of"],
}

OBJECTS = {
    "password": ["my password", "my account password", "my login"],
    "python": ["Python", "Python 3", "the Python interpreter"],
    "database": ["a database", "my database", "the database"],
    "docker": ["Docker", "a Docker container", "containers"],
    "website": ["a website", "my site", "a web app"],
    "guitar": ["guitar", "the guitar", "acoustic guitar"],
    "index_funds": ["index funds", "ETFs", "an index fund"],
    "photos": ["my photos", "my photo library", "my pictures"],
    "kubernetes": ["Kubernetes", "a Kubernetes cluster", "k8s"],
    "react_app": ["a React app", "my React project", "a frontend app"],
}

QUALIFIERS = {
    "android": ["on Android", "on an Android phone", "using Android"],
    "ios": ["on iOS", "on an iPhone", "using iOS"],
    "windows": ["on Windows", "on Windows 11", "under Windows"],
    "linux": ["on Linux", "on Ubuntu", "under Linux"],
    "aws": ["on AWS", "in AWS", "using Amazon Web Services"],
    "beginner": ["as a beginner", "as a complete beginner", "with no experience"],
    "cheaply": ["cheaply", "on a budget", "without spending much"],
    "quickly": ["quickly", "fast", "in a week"],
    "2024": ["in 2024", "this year", "right now"],
    "at_scale": ["at scale", "for a large team", "in production"],
}

# Question frames. Each takes the action, object and qualifier phrases.
FRAMES = [
    "How do I {action} {object} {qualifier}?",
    "How can I {action} {object} {qualifier}?",
    "What is the best way to {action} {object} {qualifier}?",
    "What's the easiest way to {action} {object} {qualifier}?",
    "Any advice on how to {action} {object} {qualifier}?",
    "Is there a good way to {action} {object} {qualifier}?",
    "How should someone {action} {object} {qualifier}?",
    "Can anyone explain how to {action} {object} {qualifier}?",
    "I want to {action} {object} {qualifier} -- where do I start?",
    "What do I need to know to {action} {object} {qualifier}?",
]


@dataclass
class PairConfig:
    n_pairs: int = 40_000
    n_intents: int = 900
    forms_per_intent: int = 6  # size of each intent's slice of the question bank
    duplicate_rate: float = 0.37  # the published Quora rate
    hard_negative_share: float = 0.6  # share of negatives that share two slots
    typo_rate: float = 0.06  # share of questions carrying a small corruption
    seed: int = 29

    def __post_init__(self) -> None:
        if not 0 < self.duplicate_rate < 1:
            raise ValueError("duplicate_rate must be a proportion")
        if not 0 <= self.hard_negative_share <= 1:
            raise ValueError("hard_negative_share must be a proportion")
        if self.n_intents < 50:
            raise ValueError("need enough intents to build hard negatives from")
        if self.forms_per_intent < 2:
            raise ValueError("need at least two surface forms per intent to pair them")


def _corrupt(rng: np.random.Generator, text: str) -> str:
    """Small realistic noise: a dropped character, a doubled space, lost case."""
    choice = rng.integers(0, 3)
    if choice == 0 and len(text) > 12:
        cut = int(rng.integers(5, len(text) - 2))
        return text[:cut] + text[cut + 1 :]
    if choice == 1:
        return text.replace(" ", "  ", 1)
    return text.lower()


def _intent_table(rng: np.random.Generator, cfg: PairConfig) -> pd.DataFrame:
    actions = list(ACTIONS)
    objects = list(OBJECTS)
    qualifiers = list(QUALIFIERS)

    # Sample without replacement over the product space so two intents are
    # never secretly identical.
    combos = set()
    while len(combos) < cfg.n_intents:
        combos.add(
            (
                actions[rng.integers(len(actions))],
                objects[rng.integers(len(objects))],
                qualifiers[rng.integers(len(qualifiers))],
            )
        )
    return pd.DataFrame(
        sorted(combos), columns=["action", "object", "qualifier"]
    ).assign(intent_id=lambda d: np.arange(len(d)))


def _render(rng: np.random.Generator, intent: pd.Series, cfg: PairConfig) -> str:
    frame = FRAMES[rng.integers(len(FRAMES))]
    text = frame.format(
        action=ACTIONS[intent["action"]][rng.integers(len(ACTIONS[intent["action"]]))],
        object=OBJECTS[intent["object"]][rng.integers(len(OBJECTS[intent["object"]]))],
        qualifier=QUALIFIERS[intent["qualifier"]][
            rng.integers(len(QUALIFIERS[intent["qualifier"]]))
        ],
    )
    if rng.random() < cfg.typo_rate:
        text = _corrupt(rng, text)
    return text


def _neighbour_index(intents: pd.DataFrame) -> dict[int, list[int]]:
    """For each intent, the intents sharing exactly two of its three slots.

    These are the hard negatives: near-identical wording, different answer.
    """
    neighbours: dict[int, list[int]] = {}
    for slots in (("action", "object"), ("action", "qualifier"), ("object", "qualifier")):
        for _, group in intents.groupby(list(slots), observed=True):
            ids = group["intent_id"].tolist()
            if len(ids) < 2:
                continue
            for i in ids:
                neighbours.setdefault(i, []).extend(j for j in ids if j != i)
    return {k: sorted(set(v)) for k, v in neighbours.items()}


def build_question_bank(cfg: PairConfig, rng: np.random.Generator,
                        intents: pd.DataFrame) -> dict[int, list[str]]:
    """Render a fixed set of surface forms per intent, deduplicated.

    Pairs are then drawn from this bank, so every question text recurs across
    many pairs. Deduplication matters: two identical renders of one intent
    would form a "duplicate" pair that is trivially solvable by string equality.
    """
    bank: dict[int, list[str]] = {}
    for intent_id, row in zip(intents["intent_id"], intents.to_dict("records")):
        forms: list[str] = []
        attempts = 0
        while len(forms) < cfg.forms_per_intent and attempts < cfg.forms_per_intent * 12:
            candidate = _render(rng, pd.Series(row), cfg)
            if candidate not in forms:
                forms.append(candidate)
            attempts += 1
        if len(forms) < 2:  # pathological, but never silently produce a bad bank
            raise RuntimeError(f"intent {intent_id} yielded fewer than two surface forms")
        bank[int(intent_id)] = forms
    return bank


def generate_pairs(cfg: PairConfig | None = None) -> pd.DataFrame:
    """Return question pairs with a duplicate label and their intent clusters.

    The `cluster` columns are what makes an honest split possible. Real datasets
    ship question ids instead, and you reconstruct the clusters yourself with a
    union-find over the positive pairs.
    """
    cfg = cfg or PairConfig()
    rng = np.random.default_rng(cfg.seed)
    intents = _intent_table(rng, cfg)
    neighbours = _neighbour_index(intents)
    bank = build_question_bank(cfg, rng, intents)

    n_dup = int(round(cfg.n_pairs * cfg.duplicate_rate))
    n_neg = cfg.n_pairs - n_dup
    n_hard = int(round(n_neg * cfg.hard_negative_share))

    # Duplicate pairs concentrate on popular topics; negatives are drawn
    # uniformly. This is how the real dataset was assembled -- heavily asked
    # questions accumulate many restatements -- and it is the *reason* the
    # identity-keyed features in features.py work at all. A question from a
    # popular intent sits in many more duplicate pairs than a rare one, so its
    # historical label average predicts the label of the next pair it appears
    # in, while saying nothing about whether these two questions mean the same
    # thing. With popularity flat, the target encoding is pure noise and the
    # leak has nothing to exploit.
    popularity = rng.pareto(1.4, size=cfg.n_intents) + 1.0
    popularity = popularity / popularity.sum()

    def draw(intent_id: int) -> str:
        forms = bank[intent_id]
        return forms[rng.integers(len(forms))]

    def draw_two(intent_id: int) -> tuple[str, str]:
        forms = bank[intent_id]
        i, j = rng.choice(len(forms), size=2, replace=False)
        return forms[i], forms[j]

    records = []

    popular_intents = rng.choice(cfg.n_intents, size=n_dup, p=popularity)
    for intent_id in popular_intents:
        q1, q2 = draw_two(int(intent_id))
        records.append((q1, q2, 1, int(intent_id), int(intent_id), "duplicate"))

    hard_pool = [i for i in range(cfg.n_intents) if neighbours.get(i)]
    for k in range(n_neg):
        if k < n_hard and hard_pool:
            a = int(hard_pool[rng.integers(len(hard_pool))])
            options = neighbours[a]
            b = int(options[rng.integers(len(options))])
            kind = "hard_negative"
        else:
            a = int(rng.integers(cfg.n_intents))
            b = int(rng.integers(cfg.n_intents))
            while b == a:
                b = int(rng.integers(cfg.n_intents))
            kind = "random_negative"
        records.append((draw(a), draw(b), 0, a, b, kind))

    df = pd.DataFrame(
        records,
        columns=["question1", "question2", "is_duplicate", "cluster1", "cluster2", "pair_kind"],
    )
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df["pair_id"] = np.arange(len(df))
    df["pair_kind"] = df["pair_kind"].astype("category")
    return df[
        ["pair_id", "question1", "question2", "is_duplicate", "cluster1", "cluster2", "pair_kind"]
    ]


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    frame = generate_pairs(PairConfig(n_pairs=12))
    for _, row in frame.iterrows():
        print(f"[{row['pair_kind']:15s}] dup={row['is_duplicate']}")
        print(f"  Q1: {row['question1']}")
        print(f"  Q2: {row['question2']}")
