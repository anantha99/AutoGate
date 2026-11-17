"""Utterance-level examples for the decision heads.

Every benchmark row is one utterance in one context. The five questions are
context-free, so the heads train on **unique utterances** (the row id
without its context index, as ``autogate_bench.dataset.utterance_key``):
each utterance counts once, not once per context. Predictions are made per
utterance and copied to its rows; the context enters only in the rulebook.

Questions and their vocabularies (index = logit position):

* ``capability``, ``distraction``, ``actuation``, ``consequence``: the
  policy tags, in ``autogate_router.labels.TAG_VOCABS`` order;
* ``sensitive``: ``("no", "yes")``, true when the row has sensitive spans.

``joint`` is the (capability, distraction, actuation) combination: a
closed-world question over the combinations seen in the training split
(:func:`joint_vocab`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from autogate_router.labels import TAG_VOCABS

SENSITIVE_VOCAB: tuple[str, ...] = ("no", "yes")
QUESTIONS: dict[str, tuple[str, ...]] = {**TAG_VOCABS, "sensitive": SENSITIVE_VOCAB}
ROUTING_TAGS: tuple[str, ...] = ("capability", "distraction", "actuation")


def utterance_key(row_id: str) -> str:
    return row_id.rsplit("-", 1)[0]


def row_answer(row: Mapping[str, Any], question: str) -> str:
    if question == "sensitive":
        return "yes" if row["has_sensitive_spans"] else "no"
    return str(row[question])


@dataclass
class Utterances:
    """One entry per unique utterance key, in first-seen row order."""

    keys: list[str]
    texts: list[str]
    split: np.ndarray
    intent: np.ndarray
    variant: np.ndarray
    seed_id: np.ndarray
    answers: dict[str, np.ndarray]
    """question -> answer index per utterance."""
    row_utt: np.ndarray
    """per row of the source list, the index of its utterance."""

    def __len__(self) -> int:
        return len(self.keys)

    def mask(self, *splits: str) -> np.ndarray:
        return np.isin(self.split, splits)

    def combo(self) -> list[tuple[str, str, str]]:
        return [
            tuple(QUESTIONS[q][self.answers[q][i]] for q in ROUTING_TAGS) for i in range(len(self))
        ]


def build_utterances(rows: Sequence[Mapping[str, Any]]) -> Utterances:
    index: dict[str, int] = {}
    first: list[Mapping[str, Any]] = []
    row_utt = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        k = utterance_key(r["id"])
        j = index.get(k)
        if j is None:
            j = index[k] = len(first)
            first.append(r)
        row_utt[i] = j
    answers = {
        q: np.array([vocab.index(row_answer(r, q)) for r in first], dtype=np.int64)
        for q, vocab in QUESTIONS.items()
    }
    return Utterances(
        keys=list(index),
        texts=[r["utterance"] for r in first],
        split=np.array([r["split"] for r in first]),
        intent=np.array([r["intent"] for r in first]),
        variant=np.array([r.get("variant", "plain") for r in first]),
        seed_id=np.array([r.get("seed_id", r["intent"]) for r in first]),
        answers=answers,
        row_utt=row_utt,
    )


def joint_vocab(utts: Utterances, mask: np.ndarray) -> tuple[tuple[str, str, str], ...]:
    """The (capability, distraction, actuation) combinations present under ``mask``, sorted."""
    combos = utts.combo()
    return tuple(sorted({combos[i] for i in np.flatnonzero(mask)}))


def joint_answers(utts: Utterances, vocab: Sequence[tuple[str, str, str]]) -> np.ndarray:
    """Index into ``vocab`` per utterance, -1 for a combination outside it."""
    pos = {c: i for i, c in enumerate(vocab)}
    return np.array([pos.get(c, -1) for c in utts.combo()], dtype=np.int64)


def intent_folds(utts: Utterances, mask: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Leave-intents-out folds over the utterances under ``mask``.

    Intents are sorted by their (actuation, distraction, capability) tags and
    dealt round-robin into ``k`` folds after a seeded shuffle within each tag
    group, so every fold holds out intents of every common tag value (the
    same idea as the OOD pick in ``autogate_bench.splits``). Returns, per
    fold, a boolean mask of held-out utterances.
    """
    rng = np.random.default_rng(seed)
    intents = sorted(set(utts.intent[mask]))
    tag_of = {}
    for i in np.flatnonzero(mask):
        tag_of.setdefault(
            utts.intent[i],
            tuple(int(utts.answers[q][i]) for q in ("actuation", "distraction", "capability")),
        )
    groups: dict[tuple, list[str]] = {}
    for name in intents:
        groups.setdefault(tag_of[name], []).append(name)
    order: list[str] = []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        order += g
    fold_of = {name: j % k for j, name in enumerate(order)}
    fold = np.array([fold_of.get(n, -1) for n in utts.intent])
    return [mask & (fold == f) for f in range(k)]


def seed_folds(utts: Utterances, mask: np.ndarray, k: int = 3, seed: int = 0) -> list[np.ndarray]:
    """Held-out-seed folds over the utterances under ``mask``: every utterance of a seed
    (its paraphrases and ASR copies) is held out together, so the intents are seen in
    training but the phrasing is new, like the val and test splits."""
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(set(utts.seed_id[mask])))
    rng.shuffle(seeds)
    fold_of = {s: j % k for j, s in enumerate(seeds)}
    fold = np.array([fold_of.get(s, -1) for s in utts.seed_id])
    return [mask & (fold == f) for f in range(k)]
