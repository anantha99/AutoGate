"""Intent registry and safety lane (Phase 3, runs R6b and R6l).

Registry
    Registered example phrases with their intent's tags. An utterance whose
    nearest registered phrase (cosine on the cached vectors) has similarity
    >= ``tau`` takes that intent's capability, distraction, actuation and
    consequence; the heads answer everything else (and sensitivity always).

    * **fair**: the phrases are the training split's seeds only.
    * **registered**: also the five seeds of each OOD intent, added with no
      retraining. Scored on OOD rows that are not those seeds. A new
      protocol, not comparable with R4's OOD numbers (rules analogue R0d).

    ``tau`` is chosen without heads and without val, test or OOD
    (:func:`choose_tau`): over leave-intents-out folds of the training
    intents, the registry holds the seeds of the other folds' intents; a
    match is *harmful* when the matched intent's routing tags (capability,
    distraction, actuation) differ from the utterance's. ``tau`` is the
    smallest grid value whose harmful share of matches is <= 1%.

Safety lane
    A keyword match that sets actuation to safety_critical (so the rulebook
    routes LOCAL at once). Keywords are words and word pairs that occur in
    the training seeds of safety-critical intents and in no other training
    seed; any keyword that fires on a train or val row whose route is
    REFUSE is then dropped. The lane can only make actuation more cautious.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from autogate_decide.data import QUESTIONS, ROUTING_TAGS, Utterances, intent_folds

TAU_GRID: tuple[float, ...] = tuple(round(x, 3) for x in np.arange(0.70, 0.991, 0.01))
HARMFUL_TARGET = 0.01
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _seed_mask(rows: Sequence[Mapping[str, Any]], utts: Utterances) -> np.ndarray:
    seed = np.zeros(len(utts), dtype=bool)
    for i, r in enumerate(rows):
        if r.get("source") == "seed":
            seed[utts.row_utt[i]] = True
    return seed


@dataclass
class Registry:
    vectors: np.ndarray  # [n, d], L2-normalised
    intents: np.ndarray  # [n]
    tags: dict[str, dict[str, str]]  # intent -> tag -> value
    tau: float

    def match(self, x: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Best similarity and matched intent per row of ``x``."""
        best = np.empty(len(x))
        arg = np.empty(len(x), dtype=np.int64)
        for i in range(0, len(x), chunk):
            s = x[i : i + chunk] @ self.vectors.T
            arg[i : i + chunk] = s.argmax(1)
            best[i : i + chunk] = s.max(1)
        return best, self.intents[arg]


def intent_tags(utts: Utterances) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for i, name in enumerate(utts.intent):
        if name not in out:
            out[name] = {
                q: QUESTIONS[q][utts.answers[q][i]]
                for q in ("capability", "distraction", "actuation", "consequence")
            }
    return out


def choose_tau(
    rows: Sequence[Mapping[str, Any]], utts: Utterances, x: np.ndarray, folds: int = 5
) -> tuple[float, list[dict[str, Any]]]:
    train = utts.mask("train")
    seed = _seed_mask(rows, utts)
    tags = intent_tags(utts)
    route_key = {n: tuple(t[q] for q in ROUTING_TAGS) for n, t in tags.items()}
    sims, harmful = [], []
    for held in intent_folds(utts, train, folds, seed=1):
        reg = train & seed & ~held
        queries = train & ~seed
        r = Registry(x[reg], utts.intent[reg], tags, 0.0)
        best, matched = r.match(x[queries])
        truth = utts.intent[queries]
        sims.append(best)
        harmful.append(
            np.array([route_key[m] != route_key[t] for m, t in zip(matched, truth, strict=True)])
        )
    s, h = np.concatenate(sims), np.concatenate(harmful).astype(bool)
    if len(s) == 0:
        return 1.01, []
    scan = []
    for tau in TAU_GRID:
        m = s >= tau
        scan.append(
            {
                "tau": tau,
                "match_rate": float(m.mean()),
                "harmful_share": float(h[m].mean()) if m.any() else 0.0,
            }
        )
    ok = [row["tau"] for row in scan if row["harmful_share"] <= HARMFUL_TARGET]
    return (min(ok) if ok else 1.01), scan


def build_registry(
    rows: Sequence[Mapping[str, Any]],
    utts: Utterances,
    x: np.ndarray,
    tau: float,
    mode: str = "fair",
) -> Registry:
    seed = _seed_mask(rows, utts)
    splits = ("train",) if mode == "fair" else ("train", "ood")
    reg = seed & utts.mask(*splits)
    return Registry(x[reg], utts.intent[reg], intent_tags(utts), tau)


def apply_registry(
    probs: dict[str, np.ndarray],
    registry: Registry,
    x: np.ndarray,
    joint_vocab: Sequence[tuple[str, str, str]] = (),
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Probabilities with matched utterances' tags replaced by one-hot answers."""
    best, matched = registry.match(x)
    hit = best >= registry.tau
    out = {q: p.copy() for q, p in probs.items()}
    for i in np.flatnonzero(hit):
        t = registry.tags[matched[i]]
        for q in ("capability", "distraction", "actuation", "consequence"):
            if q in out:
                out[q][i] = 0.0
                out[q][i, QUESTIONS[q].index(t[q])] = 1.0
        if "joint" in out:
            key = tuple(t[q] for q in ROUTING_TAGS)
            if key in joint_vocab:
                out["joint"][i] = 0.0
                out["joint"][i, list(joint_vocab).index(key)] = 1.0
    return out, hit


# --------------------------------------------------------------------------- #
# Safety lane
# --------------------------------------------------------------------------- #


def _grams(text: str) -> set[str]:
    words = _WORD.findall(text.lower())
    return set(words) | {f"{a} {b}" for a, b in zip(words, words[1:], strict=False)}


@dataclass
class SafetyLane:
    keywords: frozenset[str]
    dropped: tuple[str, ...] = ()

    def fires(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([bool(_grams(t) & self.keywords) for t in texts])


def build_lane(rows: Sequence[Mapping[str, Any]]) -> SafetyLane:
    seeds = [r for r in rows if r["split"] == "train" and r.get("source") == "seed"]
    sc, other = set(), set()
    for r in seeds:
        (sc if r["actuation"] == "safety_critical" else other).update(_grams(r["utterance"]))
    keywords = sc - other
    dropped = set()
    for r in rows:
        if r["split"] in ("train", "val") and r["route"] == "REFUSE":
            dropped |= _grams(r["utterance"]) & keywords
    return SafetyLane(frozenset(keywords - dropped), tuple(sorted(dropped)))


def apply_lane(probs: dict[str, np.ndarray], fires: np.ndarray) -> dict[str, np.ndarray]:
    out = {q: p.copy() for q, p in probs.items()}
    sc = QUESTIONS["actuation"].index("safety_critical")
    out["actuation"][fires] = 0.0
    out["actuation"][fires, sc] = 1.0
    return out
