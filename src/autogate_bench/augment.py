"""Train-only augmentation: synthetic goals and minimal pairs.

Two extra sources of training rows, written by generation agents against
``docs/augmentation-spec.md`` and kept in one directory (default
``data/augment``)::

    data/augment/synthetic_goals.yaml   new in-car goals, invented for a tag combination
    data/augment/minimal_pairs.yaml     look-alike utterances of two existing intents
                                        whose tags differ so that the route changes

A **synthetic goal** is a goal outside the taxonomy with its own four tags
and 8 to 12 utterances. Its rows are labelled by the rulebook from those
tags (``rulebook.route_tags``); the router's intent head ignores them
(``IGNORE_INDEX``) because the goal is not one of its classes, while the tag
heads learn from them. Their aim is more distinct goals per rare tag value.

A **minimal pair** is two utterances with high word overlap that belong to
two existing intents. Each side is an ordinary row of its intent, labelled
exactly like the intent's other rows. Their aim is the confusable,
route-changing cases (the adversarial slice).

Every augmentation row is in ``train``. ``generate --augment DIR`` appends
the rows after every other row, each utterance with its own rng, so no
existing row (and no rng stream behind one) changes.

Validation. Errors (the file is wrong, generation refuses it):

* structure: unknown group, tag, style or variant; a missing field; a goal
  name that is not snake_case, repeats, or is a taxonomy intent; a pair id
  that is not snake_case or repeats;
* goal tags that break the ``Intent`` invariants: actuation other than
  ``none`` only for ``vehicle_control`` goals, which must also be
  ``local_ok``, ``low`` distraction and have a consequence;
* text rules of the paraphrase spec: lowercase, ASCII, no punctuation but
  apostrophes and one trailing ``?``, at most 24 words, slots from the seed
  slot set only, no 4+ digit run and no synthetic-pool first name;
* leak guards: a goal utterance with token Jaccard above 0.5 to any seed or
  paraphrase of an OOD intent, or a goal name/description that contains an
  OOD intent's key words (:data:`OOD_KEYWORDS`); a pair that uses an OOD
  intent; any augmentation utterance with token Jaccard above 0.7 to a
  seed or paraphrase of a val or test seed; an exact duplicate (after
  ``paraphrases.normalize``) of any seed, paraphrase or other augmentation
  utterance. Token Jaccard uses the rules baseline's tokenizer
  (``autogate_eval.baselines.rules.content_tokens``);
* pairs: the two intents must differ and exist; ``differs_in`` must equal
  the set of tags that actually differ; the two texts must have token
  Jaccard of at least 0.4; and some context (with or without sensitive
  spans) must route them differently, otherwise the pair is not a hard
  negative for routing;
* with ``strict``: a goal with fewer than 8 or more than 12 utterances,
  fewer than 3 styles, or no hinglish or kannada_english line.

Warnings (worth a human look): the strict-only rules when not strict; two
utterances of one goal with word-set Jaccard above 0.6; a goal utterance
that looks like (token Jaccard >= 0.5) an existing intent with different
routing tags; a pair side whose nearest existing utterance belongs to a
different intent than its own (both only when at least two content tokens
are shared); a pair text with Jaccard above 0.5 to an OOD utterance.

CLI::

    python -m autogate_bench.augment validate [--dir data/augment] [--strict]
    python -m autogate_bench.augment stats [--dir data/augment]
    python -m autogate_bench.augment review [--dir data/augment] --out docs/augment-review.md
"""

from __future__ import annotations

import argparse
import itertools
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from autogate_bench.contexts import ALL_CONTEXTS
from autogate_bench.intents import INTENTS, INTENTS_BY_NAME
from autogate_bench.paraphrases import (
    DEFAULT_PARAPHRASES_DIR,
    WRITTEN_VARIANTS,
    Paraphrase,
    _format_errors,
    _LineLoader,
    load_paraphrases,
    normalize,
    word_set,
)
from autogate_bench.paraphrases import jaccard as word_jaccard
from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.rulebook import route
from autogate_bench.schema import (
    ActuationClass,
    Capability,
    Consequence,
    Context,
    DistractionLevel,
    Intent,
    IntentGroup,
)
from autogate_bench.seeds import (
    _PLACEHOLDER,
    DEFAULT_SEEDS_PATH,
    SLOTS,
    Seed,
    SeedStyle,
    load_seeds,
)
from autogate_bench.splits import OOD_INTENTS, TEST, TRAIN, VAL, assign_splits, seed_id

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUGMENT_DIR = REPO_ROOT / "data" / "augment"
GOALS_FILE = "synthetic_goals.yaml"
PAIRS_FILE = "minimal_pairs.yaml"

TAG_NAMES: tuple[str, ...] = ("capability", "distraction", "actuation", "consequence")
ROUTING_TAGS: tuple[str, ...] = ("capability", "distraction", "actuation")
TAG_ENUMS: dict[str, type] = {
    "capability": Capability,
    "distraction": DistractionLevel,
    "actuation": ActuationClass,
    "consequence": Consequence,
}

MIN_UTTERANCES = 8
MAX_UTTERANCES = 12
MIN_STYLES = 3
CODE_MIXED: frozenset[str] = frozenset({"hinglish", "kannada_english"})
OOD_MAX_JACCARD = 0.5  # goal utterance vs any OOD seed or paraphrase: must be <= this
VALTEST_MAX_JACCARD = 0.7  # any augmentation utterance vs val/test: must be <= this
PAIR_MIN_JACCARD = 0.4  # the two sides of a pair must look alike
NEAR_DUPLICATE = 0.6  # word-set Jaccard between two utterances of one goal
CONFLICT_JACCARD = 0.5  # goal utterance vs an existing intent with other routing tags
MIN_SHARED = 2  # content tokens shared before a similarity warning fires (short texts are noisy)

_SNAKE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")

# Key words of the OOD intents' goals. A synthetic goal whose name or
# description contains every word of one entry (each word matched as a
# prefix of a word in the text, so "toll" also catches "tolls") is a goal
# the OOD split tests, and is rejected. Derived from the OOD intent names
# plus their common synonyms; tests/test_augment.py checks one entry list per
# OOD intent.
OOD_KEYWORDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "set_ambient_lighting": (("ambient",), ("mood", "light")),
    "fog_lights_on": (("fog",),),
    "open_fuel_door": (
        ("fuel", "door"),
        ("fuel", "flap"),
        ("fuel", "lid"),
        ("fuel", "cap"),
        ("petrol", "flap"),
        ("petrol", "lid"),
        ("tank", "flap"),
        ("tank", "lid"),
        ("filler",),
    ),
    "route_avoiding_tolls": (("toll",),),
    "compose_long_email": (
        ("long", "email"),
        ("compose", "email"),
        ("draft", "email"),
        ("write", "email"),
    ),
    "read_news_headlines": (("news",), ("headline",)),
    "estimate_trip_cost": (("trip", "cost"), ("journey", "cost"), ("drive", "cost")),
    "pair_bluetooth_device": (("bluetooth",), ("pair",)),
}

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Utterance:
    """One line of a goal or one side of a pair. Fields are kept as written
    so the validator can report unknown values; ``line`` is the 1-based file
    line and takes no part in equality."""

    text: str
    style: str
    variant: str
    line: int = field(default=0, compare=False, repr=False)

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(m.group(1) for m in _PLACEHOLDER.finditer(self.text))


@dataclass(frozen=True, slots=True)
class Goal:
    """A synthetic in-car goal: a tag combination and utterances for it."""

    name: str
    description: str
    group: str
    capability: str
    distraction: str
    actuation: str
    consequence: str
    rationale: str
    utterances: tuple[Utterance, ...]
    line: int = field(default=0, compare=False, repr=False)

    def tags(self) -> dict[str, str]:
        return {t: getattr(self, t) for t in TAG_NAMES}

    def to_intent(self) -> Intent:
        """The goal as an ``Intent`` (raises ``ValueError`` on an unknown value or a
        broken invariant), so the rulebook and the context sampler can use it."""
        return Intent(
            self.name,
            IntentGroup(self.group),
            Capability(self.capability),
            DistractionLevel(self.distraction),
            ActuationClass(self.actuation),
            self.description,
            consequence=Consequence(self.consequence),
        )


@dataclass(frozen=True, slots=True)
class PairSide:
    intent: str
    utterance: Utterance


@dataclass(frozen=True, slots=True)
class Pair:
    """Two look-alike utterances of two existing intents whose route differs."""

    id: str
    differs_in: tuple[str, ...]
    a: PairSide
    b: PairSide
    note: str
    line: int = field(default=0, compare=False, repr=False)

    @property
    def sides(self) -> tuple[tuple[str, PairSide], ...]:
        return (("a", self.a), ("b", self.b))


@dataclass(frozen=True)
class Augmentation:
    goals: tuple[Goal, ...] = ()
    pairs: tuple[Pair, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.goals or self.pairs)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_yaml_list(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        try:
            raw = yaml.load(f, Loader=_LineLoader)  # a SafeLoader subclass
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list")
    return raw


def _mapping(item: Any, keys: Iterable[str], where: str) -> tuple[dict, int]:
    if not isinstance(item, dict):
        raise ValueError(f"{where}: expected a mapping")
    item = dict(item)
    line = item.pop("__line__", 0)
    want = set(keys)
    if set(item) != want:
        missing, extra = sorted(want - set(item)), sorted(set(item) - want)
        raise ValueError(
            f"{where} (line {line}): expected keys {sorted(want)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
        )
    return item, line


def _str(value: Any, name: str, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: {name!r} must be a string, got {value!r}")
    return value


def _utterance(item: Any, where: str) -> Utterance:
    d, line = _mapping(item, ("text", "style", "variant"), where)
    return Utterance(*(_str(d[k], k, where) for k in ("text", "style", "variant")), line)


GOAL_KEYS = (
    "name",
    "description",
    "group",
    "capability",
    "distraction",
    "actuation",
    "consequence",
    "rationale",
    "utterances",
)
PAIR_KEYS = ("id", "differs_in", "a", "b", "note")


def load_goals(path: str | Path) -> tuple[Goal, ...]:
    """Parse ``synthetic_goals.yaml``; a missing file is empty. Raises ``ValueError``
    on malformed structure only (content is checked by :func:`validate`)."""
    path = Path(path)
    goals = []
    for i, item in enumerate(_load_yaml_list(path)):
        where = f"{path}: goal {i}"
        d, line = _mapping(item, GOAL_KEYS, where)
        if not isinstance(d["utterances"], list):
            raise ValueError(f"{where} (line {line}): 'utterances' must be a list")
        utts = tuple(_utterance(u, f"{where} utterance {k}") for k, u in enumerate(d["utterances"]))
        goals.append(
            Goal(
                *(_str(d[k], k, where) for k in GOAL_KEYS[:-1]),
                utterances=utts,
                line=line,
            )
        )
    return tuple(goals)


def load_pairs(path: str | Path) -> tuple[Pair, ...]:
    """Parse ``minimal_pairs.yaml``; a missing file is empty."""
    path = Path(path)
    pairs = []
    for i, item in enumerate(_load_yaml_list(path)):
        where = f"{path}: pair {i}"
        d, line = _mapping(item, PAIR_KEYS, where)
        differs = d["differs_in"]
        if not isinstance(differs, list) or not all(isinstance(t, str) for t in differs):
            raise ValueError(f"{where} (line {line}): 'differs_in' must be a list of tag names")
        sides = []
        for key in ("a", "b"):
            sd, sline = _mapping(d[key], ("intent", "text", "style", "variant"), f"{where} {key}")
            sw = f"{where} {key}"
            sides.append(
                PairSide(
                    _str(sd["intent"], "intent", sw),
                    Utterance(*(_str(sd[k], k, sw) for k in ("text", "style", "variant")), sline),
                )
            )
        pairs.append(
            Pair(
                _str(d["id"], "id", where),
                tuple(differs),
                sides[0],
                sides[1],
                _str(d["note"], "note", where),
                line,
            )
        )
    return tuple(pairs)


def load_augmentation(directory: str | Path = DEFAULT_AUGMENT_DIR) -> Augmentation:
    """Both files of an augmentation directory (either may be absent)."""
    directory = Path(directory)
    return Augmentation(load_goals(directory / GOALS_FILE), load_pairs(directory / PAIRS_FILE))


# --------------------------------------------------------------------------- #
# Token similarity (the rules baseline's tokenizer)
# --------------------------------------------------------------------------- #


def _content_tokens() -> Callable[[str], frozenset[str]]:
    # Imported lazily: autogate_eval depends on autogate_bench, not the reverse.
    from autogate_eval.baselines.rules import content_tokens

    return content_tokens


def token_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard of two content-token sets; 0 when either is empty (nothing to compare)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@dataclass(frozen=True, slots=True)
class RefUtterance:
    """An existing seed or paraphrase: its intent, seed id, split, kind and text."""

    intent: str
    sid: str
    split: str
    kind: str  # "seed" or "paraphrase"
    text: str


class TokenIndex:
    """Inverted index over content-token sets, for the highest Jaccard to a query."""

    def __init__(self, entries: Iterable[tuple[RefUtterance, frozenset[str]]]):
        self.entries = [(e, t) for e, t in entries if t]
        self.postings: dict[str, list[int]] = {}
        for i, (_, toks) in enumerate(self.entries):
            for t in toks:
                self.postings.setdefault(t, []).append(i)

    def __len__(self) -> int:
        return len(self.entries)

    def best(self, a: frozenset[str]) -> tuple[float, RefUtterance | None]:
        """The highest token Jaccard to ``a`` and its entry (first in index order on a tie)."""
        counts: Counter = Counter()
        for t in a:
            counts.update(self.postings.get(t, ()))
        best, best_i = 0.0, None
        for i in sorted(counts):
            j = token_jaccard(a, self.entries[i][1])
            if j > best:
                best, best_i = j, i
        return best, (self.entries[best_i][0] if best_i is not None else None)


@dataclass
class Reference:
    """Everything the validator compares against, built once from seeds and paraphrases."""

    seeds: Mapping[str, Sequence[Seed]]
    paraphrases: Mapping[str, Mapping[int, Sequence[Paraphrase]]]
    splits: dict[str, str]
    ood_intents: frozenset[str]
    tokens: Callable[[str], frozenset[str]]
    existing: dict[str, str]  # normalised text -> "seed of X" / "paraphrase of X"
    ood: TokenIndex
    valtest: TokenIndex
    known: TokenIndex  # every non-OOD seed and paraphrase, for the nearest intent

    @classmethod
    def build(
        cls,
        seeds: Mapping[str, Sequence[Seed]],
        paraphrases: Mapping[str, Mapping[int, Sequence[Paraphrase]]] | None = None,
        rng_seed: int = 0,
        ood_intents: Iterable[str] = OOD_INTENTS,
    ) -> Reference:
        paraphrases = paraphrases or {}
        ood = frozenset(ood_intents)
        ids = [seed_id(n, i) for n, items in seeds.items() for i in range(len(items))]
        splits = assign_splits(ids, random.Random(f"{rng_seed}/splits"), ood)
        ct = _content_tokens()
        refs: list[RefUtterance] = []
        existing: dict[str, str] = {}
        for intent, items in seeds.items():
            for i, s in enumerate(items):
                sid = seed_id(intent, i)
                refs.append(RefUtterance(intent, sid, splits[sid], "seed", s.text))
                existing.setdefault(normalize(s.text), f"a seed of {intent}")
        for intent, by_seed in paraphrases.items():
            for i, items in by_seed.items():
                sid = seed_id(intent, i)
                for p in items:
                    refs.append(
                        RefUtterance(intent, sid, splits.get(sid, TRAIN), "paraphrase", p.text)
                    )
                    existing.setdefault(normalize(p.text), f"a paraphrase of {intent}")
        toks = [(r, ct(r.text)) for r in refs]
        return cls(
            seeds=seeds,
            paraphrases=paraphrases,
            splits=splits,
            ood_intents=ood,
            tokens=ct,
            existing=existing,
            ood=TokenIndex((r, t) for r, t in toks if r.intent in ood),
            valtest=TokenIndex((r, t) for r, t in toks if r.split in (VAL, TEST)),
            known=TokenIndex((r, t) for r, t in toks if r.intent not in ood),
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Issue:
    """One error or warning, located by file, item (goal name or pair id) and line."""

    level: str
    file: str
    item: str
    line: int
    message: str

    def format(self) -> str:
        return f"{self.file}:{self.item}:{self.line}: {self.level}: {self.message}"


@dataclass
class ValidationReport:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, file: str, item: str, line: int, msg: str) -> None:
        self.errors.append(Issue("error", file, item, line, msg))

    def warn(self, file: str, item: str, line: int, msg: str) -> None:
        self.warnings.append(Issue("warning", file, item, line, msg))


def text_errors(text: str) -> list[str]:
    """The paraphrase spec's per-line text rules, plus slots from the seed slot set only."""
    # _format_errors also compares the slot set with the seed's; a seed with
    # the same text has the same slots, so only the text rules remain.
    errs = _format_errors(Paraphrase("", 0, text, "plain"), Seed("", text, SeedStyle.COMMAND))
    unknown = sorted({s for s in _PLACEHOLDER.findall(text) if s not in SLOTS})
    if unknown:
        errs.append(f"unknown slot {', '.join('{' + s + '}' for s in unknown)}")
    return errs


def _enum_errors(values: Mapping[str, str], enums: Mapping[str, type]) -> list[str]:
    out = []
    for name, enum in enums.items():
        try:
            enum(values[name])
        except ValueError:
            allowed = ", ".join(str(v) for v in enum)
            out.append(f"unknown {name} {values[name]!r} (one of {allowed})")
    return out


def ood_keyword_hits(text: str) -> list[str]:
    """``["fog_lights_on: fog", ...]`` for every OOD key-word entry found in ``text``."""
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    hits = []
    for intent, entries in OOD_KEYWORDS.items():
        for entry in entries:
            if all(any(w.startswith(k) for w in words) for k in entry):
                hits.append(f"{intent}: {' '.join(entry)}")
    return hits


def routes_differ(
    a: Intent, b: Intent, policy: Policy = DEFAULT_POLICY
) -> list[tuple[Context, bool]]:
    """Every (context, has_sensitive_spans) where the rulebook routes ``a`` and ``b`` apart."""
    return [
        (c, s)
        for c, s in itertools.product(ALL_CONTEXTS, (False, True))
        if route(a, c, s, policy).route != route(b, c, s, policy).route
    ]


def _utterance_errors(u: Utterance) -> list[str]:
    errs = [f"{m}: {u.text!r}" for m in text_errors(u.text)]
    try:
        SeedStyle(u.style)
    except ValueError:
        errs.append(f"unknown style {u.style!r} (one of {', '.join(SeedStyle)})")
    if u.variant not in WRITTEN_VARIANTS:
        allowed = ", ".join(str(v) for v in WRITTEN_VARIANTS)
        errs.append(f"unknown variant {u.variant!r} (one of {allowed})")
    return errs


def _tags_of(intent: Intent) -> dict[str, str]:
    return {t: str(getattr(intent, t)) for t in TAG_NAMES}


def validate(
    aug: Augmentation,
    ref: Reference,
    strict: bool = False,
    check_warnings: bool = True,
    policy: Policy = DEFAULT_POLICY,
) -> ValidationReport:
    """Check an augmentation set against the seeds and paraphrases in ``ref``.

    The rules are in the module docstring. ``check_warnings=False`` skips the
    warning-only checks (``generate`` does this).
    """
    rep = ValidationReport()
    seen: dict[str, str] = {}  # normalised augmentation text -> where it first appeared

    def check_text(file: str, item: str, u: Utterance, where: str) -> frozenset[str]:
        for msg in _utterance_errors(u):
            rep.error(file, item, u.line, msg)
        key = normalize(u.text)
        if key in ref.existing:
            rep.error(file, item, u.line, f"identical to {ref.existing[key]}: {u.text!r}")
        if key in seen:
            rep.error(file, item, u.line, f"duplicate of {seen[key]}: {u.text!r}")
        else:
            seen[key] = where
        toks = ref.tokens(u.text)
        j, hit = ref.valtest.best(toks)
        if j > VALTEST_MAX_JACCARD and hit is not None:
            rep.error(
                file,
                item,
                u.line,
                f"token Jaccard {j:.2f} > {VALTEST_MAX_JACCARD} with a {hit.split} {hit.kind} "
                f"of {hit.intent} ({hit.sid}) {hit.text!r}: {u.text!r}",
            )
        return toks

    # ---- goals ----------------------------------------------------------- #
    names: set[str] = set()
    for g in aug.goals:
        f, item = GOALS_FILE, g.name or "?"
        if not _SNAKE.fullmatch(g.name):
            rep.error(f, item, g.line, f"name {g.name!r} is not snake_case")
        if g.name in INTENTS_BY_NAME:
            rep.error(f, item, g.line, f"name {g.name!r} is an existing intent")
        if g.name in names:
            rep.error(f, item, g.line, f"duplicate goal name {g.name!r}")
        names.add(g.name)
        for key in ("description", "rationale"):
            if not getattr(g, key).strip():
                rep.error(f, item, g.line, f"empty {key}")
        enum_errs = _enum_errors(
            {"group": g.group, **g.tags()}, {"group": IntentGroup, **TAG_ENUMS}
        )
        for msg in enum_errs:
            rep.error(f, item, g.line, msg)
        intent: Intent | None = None
        if not enum_errs:
            try:
                intent = g.to_intent()
            except ValueError as e:
                rep.error(f, item, g.line, f"tags break an intent invariant: {e}")
            if intent is not None and intent.actuation is not ActuationClass.NONE:
                if intent.distraction is not DistractionLevel.LOW:
                    rep.error(f, item, g.line, "a vehicle command must be low distraction")
                    intent = None
        for hit in ood_keyword_hits(f"{g.name} {g.description}"):
            rep.error(f, item, g.line, f"name or description names an OOD goal ({hit})")

        n = len(g.utterances)
        styles = {u.style for u in g.utterances}
        mixed = any(u.variant in CODE_MIXED for u in g.utterances)
        quota = []
        if not MIN_UTTERANCES <= n <= MAX_UTTERANCES:
            quota.append(f"{n} utterances (want {MIN_UTTERANCES} to {MAX_UTTERANCES})")
        if len(styles) < MIN_STYLES:
            quota.append(f"{len(styles)} styles (want at least {MIN_STYLES})")
        if not mixed:
            quota.append("no hinglish or kannada_english utterance")
        for msg in quota:
            (rep.error if strict else rep.warn)(f, item, g.line, msg)
        if n == 0:
            rep.error(f, item, g.line, "no utterances")

        words = []
        for k, u in enumerate(g.utterances):
            toks = check_text(f, item, u, f"goal {g.name} utterance {k} (line {u.line})")
            j, hit = ref.ood.best(toks)
            if j > OOD_MAX_JACCARD and hit is not None:
                rep.error(
                    f,
                    item,
                    u.line,
                    f"token Jaccard {j:.2f} > {OOD_MAX_JACCARD} with OOD intent {hit.intent} "
                    f"{hit.kind} {hit.text!r}: {u.text!r}",
                )
            words.append(word_set(u.text))
            if not check_warnings:
                continue
            for m in range(k):
                wj = word_jaccard(words[k], words[m])
                if wj > NEAR_DUPLICATE and normalize(u.text) != normalize(g.utterances[m].text):
                    rep.warn(
                        f,
                        item,
                        u.line,
                        f"near-duplicate of line {g.utterances[m].line} ({wj:.2f}): {u.text!r}",
                    )
            if intent is not None:
                kj, near = ref.known.best(toks)
                shared = len(toks & ref.tokens(near.text)) if near is not None else 0
                if near is not None and kj >= CONFLICT_JACCARD and shared >= MIN_SHARED:
                    other = INTENTS_BY_NAME[near.intent]
                    diff = [t for t in ROUTING_TAGS if getattr(other, t) != getattr(intent, t)]
                    if diff:
                        rep.warn(
                            f,
                            item,
                            u.line,
                            f"looks like {near.intent} ({kj:.2f}, {near.text!r}) whose "
                            f"{', '.join(diff)} differ: {u.text!r}",
                        )

    # ---- pairs ----------------------------------------------------------- #
    ids: set[str] = set()
    for p in aug.pairs:
        f, item = PAIRS_FILE, p.id or "?"
        if not _SNAKE.fullmatch(p.id):
            rep.error(f, item, p.line, f"id {p.id!r} is not snake_case")
        if p.id in ids:
            rep.error(f, item, p.line, f"duplicate pair id {p.id!r}")
        ids.add(p.id)
        if not p.note.strip():
            rep.error(f, item, p.line, "empty note")
        bad_tags = sorted(set(p.differs_in) - set(TAG_NAMES))
        if bad_tags:
            rep.error(f, item, p.line, f"unknown tag in differs_in: {', '.join(bad_tags)}")
        intents: list[Intent | None] = []
        toks: list[frozenset[str]] = []
        for key, side in p.sides:
            u = side.utterance
            known = INTENTS_BY_NAME.get(side.intent)
            if known is None:
                rep.error(f, item, u.line, f"{key}: unknown intent {side.intent!r}")
            elif side.intent in ref.ood_intents:
                rep.error(f, item, u.line, f"{key}: {side.intent} is an OOD intent")
                known = None
            intents.append(known)
            toks.append(check_text(f, item, u, f"pair {p.id} {key} (line {u.line})"))
            if not check_warnings:
                continue
            j, hit = ref.ood.best(toks[-1])
            if j > OOD_MAX_JACCARD and hit is not None:
                rep.warn(
                    f,
                    item,
                    u.line,
                    f"{key}: token Jaccard {j:.2f} with OOD intent {hit.intent} {hit.text!r}",
                )
            if known is not None:
                kj, near = ref.known.best(toks[-1])
                shared = len(toks[-1] & ref.tokens(near.text)) if near is not None else 0
                if near is not None and near.intent != side.intent and shared >= MIN_SHARED:
                    rep.warn(
                        f,
                        item,
                        u.line,
                        f"{key}: nearest existing utterance is {near.intent} ({kj:.2f}, "
                        f"{near.text!r}), not {side.intent}: {u.text!r}",
                    )
        a, b = intents
        if p.a.intent == p.b.intent:
            rep.error(f, item, p.line, f"both sides are {p.a.intent}")
        pj = token_jaccard(toks[0], toks[1])
        if pj < PAIR_MIN_JACCARD:
            rep.error(f, item, p.line, f"the texts do not look alike: token Jaccard {pj:.2f} < 0.4")
        if a is None or b is None or a == b:
            continue
        actual = {t for t in TAG_NAMES if getattr(a, t) != getattr(b, t)}
        if set(p.differs_in) != actual or len(set(p.differs_in)) != len(p.differs_in):
            rep.error(
                f,
                item,
                p.line,
                f"differs_in [{', '.join(p.differs_in)}] but the tags that differ are "
                f"[{', '.join(t for t in TAG_NAMES if t in actual)}]",
            )
        if not routes_differ(a, b, policy):
            rep.error(
                f,
                item,
                p.line,
                f"{a.name} and {b.name} route the same in every context: not a hard negative",
            )
    return rep


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def goal_seed_id(name: str) -> str:
    return f"goal:{name}"


def pair_seed_id(pair_id: str) -> str:
    return f"pair:{pair_id}"


def augment_rows(
    aug: Augmentation,
    seeds: Mapping[str, Sequence[Seed]],
    paraphrases: Mapping[str, Mapping[int, Sequence[Paraphrase]]] | None = None,
    rng_seed: int = 0,
    ood_intents: Iterable[str] = OOD_INTENTS,
    contexts_per_utterance: int = 3,
    n_sensitive: int = 2,
    n_generic: int = 1,
    policy: Policy = DEFAULT_POLICY,
) -> list:
    """The train rows of an augmentation set: goals in file order, then pairs.

    Utterance ids: goal utterance ``k`` is ``goal:{name}/u{k}`` (seed id
    ``goal:{name}``), a pair side is ``pair:{id}/a`` or ``/b`` (seed id
    ``pair:{id}``). Each utterance draws from ``Random(f"{rng_seed}/{uid}")``.
    Slots are filled and contexts sampled exactly as for a seed (no ASR
    copies). A goal's route is ``rulebook.route_tags`` on its tags (through
    ``rulebook.route`` on the goal as an ``Intent``); a pair side is labelled
    as its intent. Raises ``ValueError`` if :func:`validate` finds an error.
    """
    from autogate_bench.dataset import _utterance_rows

    ood_intents = tuple(ood_intents)
    ref = Reference.build(seeds, paraphrases, rng_seed, ood_intents)
    rep = validate(aug, ref, check_warnings=False, policy=policy)
    if rep.errors:
        shown = "\n".join(e.format() for e in rep.errors[:20])
        raise ValueError(f"{len(rep.errors)} augmentation errors:\n{shown}")
    common = dict(
        contexts_per_utterance=contexts_per_utterance,
        n_sensitive=n_sensitive,
        n_generic=n_generic,
        policy=policy,
    )
    rows = []
    for g in aug.goals:
        intent = g.to_intent()
        sid = goal_seed_id(g.name)
        for k, u in enumerate(g.utterances):
            uid = f"{sid}/u{k}"
            seed = Seed(g.name, u.text, SeedStyle(u.style))
            rng = random.Random(f"{rng_seed}/{uid}")
            for r in _utterance_rows(
                intent, seed, sid, uid, rng, TRAIN, "synthetic_goal", u.variant, **common
            ):
                rows.append(replace(r, is_synthetic=True))
    for p in aug.pairs:
        sid = pair_seed_id(p.id)
        for key, side in p.sides:
            intent = INTENTS_BY_NAME[side.intent]
            u = side.utterance
            uid = f"{sid}/{key}"
            seed = Seed(side.intent, u.text, SeedStyle(u.style))
            rng = random.Random(f"{rng_seed}/{uid}")
            for r in _utterance_rows(
                intent, seed, sid, uid, rng, TRAIN, "minimal_pair", u.variant, **common
            ):
                rows.append(replace(r, pair_id=p.id))
    return rows


# --------------------------------------------------------------------------- #
# Stats and review
# --------------------------------------------------------------------------- #


def stats_lines(aug: Augmentation, ood_intents: Iterable[str] = OOD_INTENTS) -> list[str]:
    """Counts: goals and utterances, goals per tag value (beside the training
    intents with that value), pairs per ``differs_in`` tag and per intent pair."""
    ood = set(ood_intents)
    train_intents = [i for i in INTENTS if i.name not in ood]
    utts = [u for g in aug.goals for u in g.utterances]
    lines = [
        f"goals: {len(aug.goals)}, goal utterances: {len(utts)}, pairs: {len(aug.pairs)} "
        f"({2 * len(aug.pairs)} utterances)",
        "goal utterances by style: "
        + ", ".join(f"{k} {v}" for k, v in sorted(Counter(u.style for u in utts).items())),
        "goal utterances by variant: "
        + ", ".join(f"{k} {v}" for k, v in sorted(Counter(u.variant for u in utts).items())),
        "",
        "goals per tag value (training intents with that value in brackets):",
    ]
    for tag, enum in {"group": IntentGroup, **TAG_ENUMS}.items():
        for value in enum:
            gs = [g for g in aug.goals if getattr(g, tag) == str(value)]
            n_utt = sum(len(g.utterances) for g in gs)
            n_train = sum(str(getattr(i, tag)) == str(value) for i in train_intents)
            lines.append(
                f"  {tag:<12} {str(value):<17} {len(gs):>4} goals {n_utt:>5} utterances  "
                f"[{n_train}]"
            )
        unknown = Counter(
            getattr(g, tag) for g in aug.goals if getattr(g, tag) not in {str(v) for v in enum}
        )
        for value, n in sorted(unknown.items()):
            lines.append(f"  {tag:<12} {value:<17} {n:>4} goals (unknown value)")
    combos = Counter((g.capability, g.distraction, g.actuation, g.consequence) for g in aug.goals)
    lines += ["", "goal tag combinations (capability / distraction / actuation / consequence):"]
    lines += [f"  {n:>4}  {' / '.join(c)}" for c, n in sorted(combos.items(), key=lambda x: -x[1])]
    lines += ["", "pairs per differs_in tag:"]
    per_tag = Counter(t for p in aug.pairs for t in p.differs_in)
    lines += [f"  {t:<12} {per_tag.get(t, 0):>4}" for t in TAG_NAMES]
    per_combo = Counter(tuple(p.differs_in) for p in aug.pairs)
    lines += ["", "pairs per differs_in set:"]
    lines += [
        f"  {n:>4}  {', '.join(c)}" for c, n in sorted(per_combo.items(), key=lambda x: -x[1])
    ]
    per_intents = Counter(tuple(sorted((p.a.intent, p.b.intent))) for p in aug.pairs)
    lines += ["", "pairs per intent pair:"]
    lines += [
        f"  {n:>4}  {a} / {b}" for (a, b), n in sorted(per_intents.items(), key=lambda x: -x[1])
    ]
    return lines


def _fmt_tags(tags: Mapping[str, str]) -> str:
    return " / ".join(str(tags[t]) for t in TAG_NAMES)


@dataclass
class _UttReview:
    text: str
    style: str
    variant: str
    near: tuple[float, RefUtterance | None]
    ood: tuple[float, RefUtterance | None]
    valtest: tuple[float, RefUtterance | None]
    near_tags_differ: list[str]


def _review_utterance(
    u: Utterance, ref: Reference, tags: Mapping[str, str] | None, own_intent: str | None = None
) -> _UttReview:
    toks = ref.tokens(u.text)
    near = ref.known.best(toks)
    differ: list[str] = []
    if near[1] is not None:
        other = INTENTS_BY_NAME[near[1].intent]
        if tags is not None:
            differ = [t for t in TAG_NAMES if str(getattr(other, t)) != tags.get(t)]
        if own_intent is not None and near[1].intent != own_intent:
            differ = ["intent"]
    return _UttReview(
        u.text, u.style, u.variant, near, ref.ood.best(toks), ref.valtest.best(toks), differ
    )


def _risk(rs: Sequence[_UttReview], errors: int) -> float:
    """0..~1: the closest call among the leak guards and the label-conflict check."""
    score = 0.0
    for r in rs:
        score = max(score, r.ood[0] / OOD_MAX_JACCARD, r.valtest[0] / VALTEST_MAX_JACCARD)
        if r.near_tags_differ:
            score = max(score, r.near[0] / CONFLICT_JACCARD)
    return score + errors


def _ref_text(hit: tuple[float, RefUtterance | None]) -> str:
    j, r = hit
    if r is None:
        return "-"
    return f"{j:.2f} {r.intent} `{r.text}`"


def review_markdown(
    aug: Augmentation, ref: Reference, rep: ValidationReport, policy: Policy = DEFAULT_POLICY
) -> str:
    """The risk-sorted review page: every goal and pair, with what to look at first."""
    by_item: Counter = Counter((i.file, i.item) for i in rep.errors)
    warn_item: Counter = Counter((i.file, i.item) for i in rep.warnings)
    out = [
        "# Augmentation review",
        "",
        "Generated by `python -m autogate_bench.augment review`; do not edit by hand.",
        "Items are sorted by risk, highest first. Risk is the closest call among: token",
        f"Jaccard to an OOD utterance / {OOD_MAX_JACCARD}, to a val/test utterance /",
        f"{VALTEST_MAX_JACCARD}, and (when its tags differ) to the nearest existing intent /",
        f"{CONFLICT_JACCARD}; each validation error adds 1. Above 1.0 is past a limit.",
        "",
        f"{len(aug.goals)} goals, {len(aug.pairs)} pairs; validation: {len(rep.errors)} errors, "
        f"{len(rep.warnings)} warnings.",
        "",
        "Columns: **nearest** is the existing seed or paraphrase (non-OOD intents) with the",
        "highest token Jaccard (rules-baseline tokenizer); **tags differ** lists the tags in",
        "which that intent differs from the goal (for a pair side: `intent` if the nearest",
        "utterance belongs to another intent); **OOD** and **val/test** are the highest",
        "Jaccard to any OOD, or val/test, seed or paraphrase.",
        "",
    ]
    goals = []
    for g in aug.goals:
        tags = g.tags()
        rs = [_review_utterance(u, ref, tags) for u in g.utterances]
        goals.append((_risk(rs, by_item[(GOALS_FILE, g.name)]), g, rs))
    goals.sort(key=lambda x: -x[0])
    out += ["## Synthetic goals", ""]
    if not goals:
        out += ["None.", ""]
    for risk, g, rs in goals:
        flags = []
        if by_item[(GOALS_FILE, g.name)]:
            flags.append(f"{by_item[(GOALS_FILE, g.name)]} validation errors")
        if warn_item[(GOALS_FILE, g.name)]:
            flags.append(f"{warn_item[(GOALS_FILE, g.name)]} warnings")
        nearest = Counter(r.near[1].intent for r in rs if r.near[1] is not None)
        out += [
            f"### {g.name} (risk {risk:.2f})",
            "",
            f"- {g.description}",
            f"- group `{g.group}`; tags {_fmt_tags(g.tags())}",
            f"- rationale: {g.rationale}",
        ]
        if nearest:
            top, n = nearest.most_common(1)[0]
            other = INTENTS_BY_NAME[top]
            otags = _tags_of(other)
            same = otags == {t: str(v) for t, v in g.tags().items()}
            out.append(
                f"- nearest existing intent (most utterances): `{top}` ({n}/{len(rs)}), tags "
                + ("identical" if same else _fmt_tags(otags) + " **(differ)**")
            )
        if flags:
            out.append(f"- **attention:** {'; '.join(flags)}")
        out += [
            "",
            "| # | utterance | style | variant | nearest | tags differ | OOD | val/test |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for k, r in enumerate(rs):
            out.append(
                f"| {k} | `{r.text}` | {r.style} | {r.variant} | {_ref_text(r.near)} "
                f"| {', '.join(r.near_tags_differ) or '-'} | {_ref_text(r.ood)} "
                f"| {_ref_text(r.valtest)} |"
            )
        out.append("")

    pairs = []
    for p in aug.pairs:
        rs = [_review_utterance(s.utterance, ref, None, s.intent) for _, s in p.sides]
        pairs.append((_risk(rs, by_item[(PAIRS_FILE, p.id)]), p, rs))
    pairs.sort(key=lambda x: -x[0])
    out += ["## Minimal pairs", ""]
    if not pairs:
        out += ["None.", ""]
    for risk, p, rs in pairs:
        a, b = INTENTS_BY_NAME.get(p.a.intent), INTENTS_BY_NAME.get(p.b.intent)
        pj = token_jaccard(ref.tokens(p.a.utterance.text), ref.tokens(p.b.utterance.text))
        out += [f"### {p.id} (risk {risk:.2f})", ""]
        out.append(f"- differs_in: {', '.join(p.differs_in)}; token Jaccard a/b {pj:.2f}")
        if a is not None and b is not None:
            diff = routes_differ(a, b, policy)
            out.append(
                f"- routes differ in {len(diff)} of {2 * len(ALL_CONTEXTS)} (context, sensitive) "
                "cases"
                + (
                    f", e.g. {diff[0][0].to_prefix()}: {route(a, *diff[0]).route} vs "
                    f"{route(b, *diff[0]).route}"
                    if diff
                    else ""
                )
            )
        out.append(f"- note: {p.note}")
        n_err, n_warn = by_item[(PAIRS_FILE, p.id)], warn_item[(PAIRS_FILE, p.id)]
        if n_err or n_warn:
            out.append(f"- **attention:** {n_err} validation errors; {n_warn} warnings")
        out += [
            "",
            "| side | intent | tags | utterance | style | nearest | OOD | val/test |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for (key, side), r in zip(p.sides, rs, strict=True):
            intent = INTENTS_BY_NAME.get(side.intent)
            tags = _fmt_tags(_tags_of(intent)) if intent else "?"
            near = _ref_text(r.near) + (" **(other intent)**" if r.near_tags_differ else "")
            out.append(
                f"| {key} | `{side.intent}` | {tags} | `{r.text}` | {r.style} | {near} "
                f"| {_ref_text(r.ood)} | {_ref_text(r.valtest)} |"
            )
        out.append("")
    if rep.errors or rep.warnings:
        out += ["## Validation messages", ""]
        out += [f"- {i.format()}" for i in rep.errors + rep.warnings]
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load(a: argparse.Namespace) -> tuple[Augmentation, Reference] | None:
    try:
        aug = load_augmentation(a.dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return None
    seeds = load_seeds(a.seeds)
    paraphrases = load_paraphrases(a.paraphrases) if a.paraphrases else {}
    return aug, Reference.build(seeds, paraphrases, a.rng_seed)


def _cmd_validate(a: argparse.Namespace) -> int:
    loaded = _load(a)
    if loaded is None:
        return 2
    aug, ref = loaded
    if not aug:
        print(f"no augmentation files (or empty ones) in {a.dir}")
        return 0
    rep = validate(aug, ref, strict=a.strict, check_warnings=not a.no_warnings)
    n_utt = sum(len(g.utterances) for g in aug.goals)
    print(f"{len(aug.goals)} goals ({n_utt} utterances), {len(aug.pairs)} pairs in {a.dir}")
    for issue in rep.errors + rep.warnings:
        print(issue.format())
    print(
        f"\n{len(rep.errors)} errors, {len(rep.warnings)} warnings{' (strict)' if a.strict else ''}"
    )
    return 1 if rep.errors else 0


def _cmd_stats(a: argparse.Namespace) -> int:
    try:
        aug = load_augmentation(a.dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n".join(stats_lines(aug)))
    return 0


def _cmd_review(a: argparse.Namespace) -> int:
    loaded = _load(a)
    if loaded is None:
        return 2
    aug, ref = loaded
    rep = validate(aug, ref, strict=a.strict)
    text = review_markdown(aug, ref, rep)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n", encoding="utf-8")
    print(
        f"wrote {a.out}: {len(aug.goals)} goals, {len(aug.pairs)} pairs, "
        f"{len(rep.errors)} errors, {len(rep.warnings)} warnings"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m autogate_bench.augment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, reference: bool = True) -> None:
        sp.add_argument("--dir", type=Path, default=Path("data/augment"))
        if reference:
            sp.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS_PATH)
            sp.add_argument(
                "--paraphrases",
                type=Path,
                default=DEFAULT_PARAPHRASES_DIR,
                help="paraphrase directory compared against (default data/paraphrases)",
            )
            sp.add_argument(
                "--rng-seed", type=int, default=0, help="generation seed that fixes the splits"
            )
            sp.add_argument("--strict", action="store_true", help="enforce the per-goal quota")

    v = sub.add_parser("validate", help="check the augmentation files")
    common(v)
    v.add_argument("--no-warnings", action="store_true", help="skip the warning checks")
    v.set_defaults(func=_cmd_validate)
    s = sub.add_parser("stats", help="goals per tag value, pairs per tag, counts")
    common(s, reference=False)
    s.set_defaults(func=_cmd_stats)
    r = sub.add_parser("review", help="write the risk-sorted review page")
    common(r)
    r.add_argument("--out", type=Path, default=Path("docs/augment-review.md"))
    r.set_defaults(func=_cmd_review)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
