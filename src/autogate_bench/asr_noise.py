"""Deterministic ASR-style noise for paraphrases.

At generation time each seed gets a few ``asr`` utterances, each derived from
one of its ``plain`` paraphrases by :func:`asr_noise`. The noise imitates what
a speech recogniser does to Indian-English in-car speech, and it is applied
in code rather than written by the paraphrase agents so it is free,
reproducible and consistent across the dataset.

One call picks one edit, and with probability ``SECOND_EDIT_P`` a second one
at a different position, from four kinds:

``homophone``
    Swap a word for one that sounds the same or nearly the same, from
    :data:`HOMOPHONES` (to/two, wipers/vipers, seat/sheet, defrost/the frost,
    ac/easy, ...). The table is directional: it maps what was said to what
    was heard.
``drop``
    Delete one function word from :data:`DROPPABLE` (the, a, to, my,
    please, can, you), as a recogniser does with unstressed words. Never
    the only word.
``number``
    Swap a spelled-out number for its homophone and back (one/won, two/too,
    four/for, eight/ate). Slot values are never involved.
``merge_split``
    Join a compound written as two words or split one written as one
    (lane assist/laneassist, seatbelt/seat belt, voicemail/voice mail).

The kind is drawn uniformly from those that apply to the text, then the
position uniformly within that kind, so a text with one droppable word and
ten homophones does not see only homophone swaps. A trailing ``?`` is
always removed (a recogniser does not punctuate). Text inside a ``{slot}``,
including a token such as ``{contact}'s``, is never touched, so the span
injector fills the output exactly as it fills the input. The output always
differs from the input; :func:`asr_noise` returns ``None`` when no edit
applies. Everything is drawn from the ``random.Random`` passed in.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

SECOND_EDIT_P = 0.25

# what was said -> what the recogniser heard
_HOMOPHONE_PAIRS: tuple[tuple[str, str], ...] = (
    # classic homophones, both directions
    ("to", "two"),
    ("to", "too"),
    ("for", "four"),
    ("there", "their"),
    ("their", "there"),
    ("right", "write"),
    ("brake", "break"),
    ("break", "brake"),
    ("hear", "here"),
    ("here", "hear"),
    ("weather", "whether"),
    ("road", "rode"),
    ("by", "buy"),
    ("no", "know"),
    ("know", "no"),
    ("new", "knew"),
    ("hour", "our"),
    ("our", "hour"),
    ("week", "weak"),
    ("meet", "meat"),
    ("way", "weigh"),
    ("wait", "weight"),
    ("where", "wear"),
    ("route", "root"),
    ("rain", "reign"),
    ("whole", "hole"),
    ("mail", "male"),
    # in-car near-homophones
    ("wipers", "vipers"),
    ("wiper", "viper"),
    ("heater", "eater"),
    ("seat", "sheet"),
    ("seats", "sheets"),
    ("call", "cool"),
    ("cool", "call"),
    ("text", "test"),
    ("defrost", "the frost"),
    ("lane", "line"),
    ("ac", "easy"),
    ("mode", "mood"),
    ("mood", "mode"),
    ("eco", "echo"),
    ("sport", "spot"),
    ("fan", "van"),
    ("air", "hair"),
    ("vent", "went"),
    ("vents", "events"),
    ("blower", "lower"),
    ("warm", "worm"),
    ("heat", "hit"),
    ("mist", "missed"),
    ("cruise", "crews"),
    ("boot", "boat"),
    ("petrol", "patrol"),
    ("bunk", "bank"),
    ("fuel", "full"),
    ("range", "change"),
    ("window", "widow"),
    ("doors", "drawers"),
    ("lock", "luck"),
    ("park", "bark"),
    ("parking", "barking"),
    ("pause", "paws"),
    ("play", "pray"),
    ("skip", "ship"),
    ("mute", "moot"),
    ("message", "massage"),
    ("reply", "rely"),
    ("reminder", "remainder"),
    ("note", "not"),
    ("stop", "shop"),
    ("cancel", "council"),
    ("left", "lift"),
    ("turn", "torn"),
    ("score", "store"),
    ("match", "much"),
    ("reach", "rich"),
    ("lights", "lites"),
    ("charging", "charting"),
    ("dickey", "dicky"),
)

HOMOPHONES: dict[str, tuple[str, ...]] = {}
for _said, _heard in _HOMOPHONE_PAIRS:
    HOMOPHONES[_said] = (*HOMOPHONES.get(_said, ()), _heard)

NUMBER_WORDS: dict[str, str] = {
    "one": "won",
    "won": "one",
    "two": "too",
    "four": "for",
    "eight": "ate",
    "ate": "eight",
}

DROPPABLE: frozenset[str] = frozenset({"the", "a", "to", "my", "please", "can", "you"})

_COMPOUNDS: tuple[tuple[str, str], ...] = (
    ("lane assist", "laneassist"),
    ("seat belt", "seatbelt"),
    ("head lights", "headlights"),
    ("head light", "headlight"),
    ("voice mail", "voicemail"),
    ("blue tooth", "bluetooth"),
    ("play list", "playlist"),
    ("hand brake", "handbrake"),
    ("wind shield", "windshield"),
    ("wind screen", "windscreen"),
    ("dash board", "dashboard"),
    ("pod cast", "podcast"),
    ("tail gate", "tailgate"),
    ("wash room", "washroom"),
    ("sun roof", "sunroof"),
    ("e mail", "email"),
    ("a c", "ac"),
    ("in to", "into"),
    ("can not", "cannot"),
    ("may be", "maybe"),
    ("any one", "anyone"),
    ("some where", "somewhere"),
    ("near by", "nearby"),
    ("to day", "today"),
    ("to night", "tonight"),
    ("to morrow", "tomorrow"),
    ("week end", "weekend"),
    ("de mist", "demist"),
    ("un lock", "unlock"),
    ("cruise control", "cruisecontrol"),
    ("fog lights", "foglights"),
    ("child lock", "childlock"),
)

# spoken form (one or two words) -> its merged or split counterpart
MERGE_SPLIT: dict[str, str] = {}
for _split, _merged in _COMPOUNDS:
    MERGE_SPLIT[_merged] = _split
    MERGE_SPLIT[_split] = _merged

KINDS: tuple[str, ...] = ("homophone", "drop", "number", "merge_split")


@dataclass(frozen=True, slots=True)
class Edit:
    """Replace ``n`` words starting at word ``start`` with ``replacement`` (may be empty)."""

    kind: str
    start: int
    n: int
    replacement: str


def _frozen(word: str) -> bool:
    return "{" in word or "}" in word


def candidate_edits(words: Sequence[str]) -> list[Edit]:
    """Every edit that applies to ``words``, in a fixed order."""
    out: list[Edit] = []
    free = [not _frozen(w) for w in words]
    for i, w in enumerate(words):
        if not free[i]:
            continue
        for heard in HOMOPHONES.get(w, ()):
            out.append(Edit("homophone", i, 1, heard))
        if w in DROPPABLE and len(words) > 1:
            out.append(Edit("drop", i, 1, ""))
        if w in NUMBER_WORDS:
            out.append(Edit("number", i, 1, NUMBER_WORDS[w]))
        if w in MERGE_SPLIT:
            out.append(Edit("merge_split", i, 1, MERGE_SPLIT[w]))
        if i + 1 < len(words) and free[i + 1]:
            pair = f"{w} {words[i + 1]}"
            if pair in HOMOPHONES:
                out.extend(Edit("homophone", i, 2, h) for h in HOMOPHONES[pair])
            if pair in MERGE_SPLIT:
                out.append(Edit("merge_split", i, 2, MERGE_SPLIT[pair]))
    return out


def _pick(edits: Sequence[Edit], rng: random.Random) -> Edit:
    kinds = [k for k in KINDS if any(e.kind == k for e in edits)]
    kind = kinds[rng.randrange(len(kinds))]
    pool = [e for e in edits if e.kind == kind]
    return pool[rng.randrange(len(pool))]


def _apply(words: Sequence[str], edits: Sequence[Edit]) -> str:
    out: list[str] = []
    by_start = {e.start: e for e in edits}
    i = 0
    while i < len(words):
        e = by_start.get(i)
        if e is None:
            out.append(words[i])
            i += 1
        else:
            if e.replacement:
                out.append(e.replacement)
            i += e.n
    return " ".join(out)


def asr_edits(text: str, rng: random.Random) -> list[Edit]:
    """The edits :func:`asr_noise` makes to ``text`` (empty if none applies).

    One edit, of a kind drawn uniformly from the kinds that apply; with
    probability ``SECOND_EDIT_P`` a second one at a position that does not
    overlap the first, unless it would delete every word.
    """
    words = text.removesuffix("?").split()
    edits = candidate_edits(words)
    if not edits:
        return []
    chosen = [_pick(edits, rng)]
    if rng.random() < SECOND_EDIT_P:
        a = chosen[0]
        rest = [e for e in edits if e.start >= a.start + a.n or e.start + e.n <= a.start]
        if rest:
            second = _pick(rest, rng)
            if _apply(words, [a, second]):
                chosen.append(second)
    return chosen


def asr_noise(text: str, rng: random.Random) -> str | None:
    """One noisy version of ``text``, or ``None`` if no edit applies.

    The result never equals ``text`` (after the trailing ``?`` is removed,
    at least one word is changed or dropped) and never alters a ``{slot}``.
    """
    edits = asr_edits(text, rng)
    if not edits:
        return None
    body = text.removesuffix("?").rstrip()
    out = _apply(body.split(), edits)
    if not out or out == body:
        return None
    return out


def asr_variants(text: str, rng: random.Random, k: int) -> list[str]:
    """Up to ``k`` distinct noisy versions of ``text`` (fewer if the text allows fewer)."""
    out: list[str] = []
    for _ in range(max(0, k) * 10):
        if len(out) >= k:
            break
        v = asr_noise(text, rng)
        if v is None:
            break
        if v not in out:
            out.append(v)
    return out
