"""Span injector: fill a seed's ``{slot}`` placeholders and record sensitive offsets.

Every value here is synthetic. The names are invented combinations of common
Indian first names and regional surnames (Kannada, Tamil, Hindi, Malayalam,
Bengali, Punjabi), the phone numbers, OTPs and card digits are generated from
the rng, and the addresses pair plausible house numbers with well-known
localities. None of it identifies a real person.

Only the five sensitive slots (``contact``, ``phone``, ``address``, ``otp``,
``card``) produce a :class:`Span`; everything else is filled from a small
non-sensitive pool and leaves no trace beyond the utterance text. All values
are lowercase ASCII, matching the seed text rules, so the utterances look
like ASR output.

Sensitive versus generic fills
------------------------------
``fill(seed, rng, sensitive=False)`` is only valid for seeds whose *only*
sensitive slot is ``{contact}``: the contact becomes a relation word ("my
wife", "amma") that names nobody, so the row has no spans and routes like a
plain request. A phone number, address, OTP or card number is sensitive
whatever it looks like, so seeds with those slots are always filled
sensitively and ``sensitive=False`` raises ``ValueError`` for them.

The ``{card}`` slot is filled with the last four digits only, because the
seeds already say "card ending {card}".
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from autogate_bench.seeds import _PLACEHOLDER, SENSITIVE_SLOTS, Seed

# --------------------------------------------------------------------------- #
# Sensitive pools
# --------------------------------------------------------------------------- #

NAMES: tuple[str, ...] = (
    # Kannada
    "anil hegde",
    "kavya shetty",
    "ravi gowda",
    "deepa kulkarni",
    "suresh bhat",
    "meghana rao",
    # Tamil
    "karthik subramanian",
    "divya raghavan",
    "arun venkatesan",
    "lakshmi iyer",
    "senthil murugan",
    "priya natarajan",
    "vignesh ramasamy",
    # Hindi
    "rahul sharma",
    "neha verma",
    "amit tiwari",
    "pooja mishra",
    "vikas yadav",
    "sunita chauhan",
    "rohit saxena",
    "nikhil joshi",
    # Malayalam
    "anjali nair",
    "vishnu menon",
    "sreeja pillai",
    "joseph kurian",
    "arjun panicker",
    "reshma namboothiri",
    # Bengali
    "sourav chatterjee",
    "moumita banerjee",
    "arindam mukherjee",
    "tanushree ghosh",
    "debashis sen",
    "riya dasgupta",
    # Punjabi
    "harpreet sandhu",
    "gurpreet gill",
    "manjeet dhillon",
    "simran grewal",
    "jaspreet sidhu",
    "navjot brar",
    # Urdu
    "farah qureshi",
)

# A relation word names nobody, so it is not a span.
RELATIONS: tuple[str, ...] = (
    "my wife",
    "my husband",
    "amma",
    "appa",
    "my boss",
    "my manager",
    "my brother",
    "my sister",
    "my son",
    "my daughter",
    "my mom",
    "my dad",
    "my friend",
    "my cousin",
    "my landlord",
    "nani",
    "my neighbour",
    "my colleague",
    "my uncle",
    "the driver",
)

ADDRESSES: tuple[str, ...] = (
    # Bengaluru
    "42 mg road",
    "flat 3b green acres whitefield",
    "12th cross indiranagar",
    "221 100 feet road koramangala 560034",
    "no 7 4th main jayanagar",
    # Chennai
    "18 radhakrishnan salai mylapore",
    "plot 56 anna nagar west",
    "door 9 2nd avenue besant nagar",
    "14 cathedral road teynampet 600086",
    # Hyderabad
    "flat 402 lake view residency kukatpally",
    "house 11 road no 12 banjara hills",
    "3 jubilee hills checkpost lane",
    # Mumbai
    "704 b wing sea breeze apartments bandra west",
    "15 linking road santacruz",
    "room 3 shanti niwas dadar east",
    "22 hill road bandra 400050",
    # Delhi
    "c 42 lajpat nagar 2",
    "house 118 sector 15 rohini",
    "a 7 greater kailash 1",
    "flat 12 pocket b vasant kunj",
)

# --------------------------------------------------------------------------- #
# Non-sensitive pools, one per remaining slot
# --------------------------------------------------------------------------- #
# Values avoid the command words of other intents' seeds ("call", "home",
# "time", "next") so a fill never reads as a different request.

POOLS: dict[str, tuple[str, ...]] = {
    "place": (
        "coffee shop",
        "south indian restaurant",
        "biryani place",
        "dhaba",
        "pharmacy",
        "petrol bunk",
        "bakery",
        "tea stall",
        "supermarket",
        "pizza place",
    ),
    "city": (
        "bengaluru",
        "chennai",
        "hyderabad",
        "mumbai",
        "delhi",
        "pune",
        "kochi",
        "mysuru",
        "goa",
        "jaipur",
        "kolkata",
        "ooty",
    ),
    "temperature": ("18", "20", "21", "22", "23", "24", "25", "26"),
    "fan_level": ("low", "medium", "high", "max", "level 2", "level 3", "auto"),
    "song": (
        "tum hi ho",
        "chaiyya chaiyya",
        "kal ho naa ho",
        "munbe vaa",
        "kun faya kun",
        "jai ho",
        "rowdy baby",
        "malare",
    ),
    "artist": (
        "a r rahman",
        "arijit singh",
        "shreya ghoshal",
        "ilaiyaraaja",
        "kishore kumar",
        "lata mangeshkar",
        "anirudh",
        "sid sriram",
        "prateek kuhad",
    ),
    "station": (
        "radio city",
        "radio mirchi",
        "big fm",
        "red fm",
        "fm rainbow",
        "vividh bharati",
        "radio indigo",
    ),
    "podcast": (
        "the history hour",
        "cricket banter",
        "tech this week",
        "the morning brief",
        "startup stories",
        "desi science",
        "mind matters",
    ),
    "time": (
        "5 pm",
        "6 30",
        "7 am",
        "noon",
        "8 15 pm",
        "half past nine",
        "10 tonight",
        "4 in the evening",
    ),
    "date": (
        "monday",
        "friday",
        "saturday",
        "the 14th",
        "the 21st",
        "march 3rd",
        "thursday",
        "diwali",
    ),
    "duration": (
        "10 minutes",
        "15 minutes",
        "half an hour",
        "an hour",
        "two hours",
        "20 minutes",
    ),
    "team": (
        "india",
        "rcb",
        "csk",
        "mumbai indians",
        "kkr",
        "bengaluru fc",
        "mohun bagan",
    ),
    "topic": (
        "quantum computing",
        "inflation",
        "black holes",
        "the stock market",
        "electric vehicles",
        "climate change",
        "the union budget",
        "quarterly sales",
    ),
    "note_text": (
        "buy milk",
        "pay the electricity bill",
        "book the car service",
        "water the plants",
        "renew the insurance",
        "collect the laundry",
        "get vegetables",
    ),
    "message_text": (
        "i'm running late",
        "on my way",
        "reached the office",
        "stuck in traffic",
        "see you at dinner",
        "start without me",
        "be there soon",
    ),
    "event_title": (
        "team standup",
        "dentist appointment",
        "parent teacher meeting",
        "project review",
        "gym session",
        "car service",
    ),
    "drive_mode": ("eco", "sport", "normal", "comfort", "sport plus"),
    "color": ("blue", "red", "purple", "warm white", "green", "amber", "orange"),
    "speed": ("60", "80", "90", "100", "110", "120"),
    "seat": ("driver", "passenger", "rear left", "rear right", "front"),
    "window": ("driver", "passenger", "rear left", "rear right", "front", "back"),
    "number": ("2", "3", "4", "5", "6", "8", "10", "12"),
}

# A few slots read differently in one intent; these pools replace the
# generic one there.
INTENT_POOLS: dict[tuple[str, str], tuple[str, ...]] = {
    ("plan_multi_day_trip", "duration"): (
        "two day",
        "three day",
        "four day",
        "five day",
        "week long",
    ),
    ("tune_radio_station", "number"): ("91.1", "93.5", "98.3", "102.4", "104.8"),
}

GENERIC_CONTACT_ONLY = frozenset({"contact"})


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def gen_phone(rng: random.Random) -> str:
    """An Indian mobile number in one of four written forms."""
    number = rng.choice("99998876") + _digits(rng, 9)
    form = rng.randrange(4)
    if form == 0:
        return f"+91 {number[:5]} {number[5:]}"
    if form == 1:
        return f"+91 {number}"
    if form == 2:
        return number
    return f"{number[:5]} {number[5:]}"


def gen_otp(rng: random.Random) -> str:
    return _digits(rng, rng.choice((4, 6)))


def gen_card(rng: random.Random) -> str:
    """The last four digits of a card (the seeds say "card ending {card}")."""
    return _digits(rng, 4)


# --------------------------------------------------------------------------- #
# Spans and fills
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Span:
    """A sensitive substring: ``utterance[start:end] == text``."""

    start: int
    end: int
    label: str
    text: str

    def to_dict(self) -> dict[str, int | str]:
        return {"start": self.start, "end": self.end, "label": self.label, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> Span:
        return cls(int(d["start"]), int(d["end"]), str(d["label"]), str(d["text"]))


@dataclass(frozen=True, slots=True)
class Filled:
    """One filled seed: the utterance and its sensitive spans."""

    seed: Seed
    utterance: str
    spans: tuple[Span, ...]

    @property
    def has_sensitive_spans(self) -> bool:
        return bool(self.spans)


def _sensitive_slots(seed: Seed) -> set[str]:
    return {s for s in seed.slots if s in SENSITIVE_SLOTS}


def _value(slot: str, intent: str, rng: random.Random, sensitive: bool) -> str:
    if slot == "contact":
        return rng.choice(NAMES if sensitive else RELATIONS)
    if slot == "phone":
        return gen_phone(rng)
    if slot == "otp":
        return gen_otp(rng)
    if slot == "card":
        return gen_card(rng)
    if slot == "address":
        return rng.choice(ADDRESSES)
    return rng.choice(INTENT_POOLS.get((intent, slot), POOLS[slot]))


def fill(seed: Seed, rng: random.Random, sensitive: bool = True) -> Filled:
    """Fill every placeholder in ``seed`` and record the sensitive spans.

    A slot that appears twice gets the same value both times. With
    ``sensitive=False`` the seed's only sensitive slot must be ``{contact}``;
    it is filled with a relation word and no span is recorded.
    """
    sens = _sensitive_slots(seed)
    if not sensitive and sens - GENERIC_CONTACT_ONLY:
        raise ValueError(
            f"{seed.text!r}: sensitive=False is only valid when contact is the only "
            f"sensitive slot (found {sorted(sens)})"
        )
    values: dict[str, str] = {}
    for slot in seed.slots:
        if slot not in values:
            values[slot] = _value(slot, seed.intent, rng, sensitive)

    parts: list[str] = []
    spans: list[Span] = []
    pos = 0  # position in the output utterance
    last = 0  # position in the seed text
    for m in _PLACEHOLDER.finditer(seed.text):
        literal = seed.text[last : m.start()]
        parts.append(literal)
        pos += len(literal)
        slot = m.group(1)
        value = values[slot]
        if sensitive and slot in SENSITIVE_SLOTS:
            spans.append(Span(pos, pos + len(value), slot, value))
        parts.append(value)
        pos += len(value)
        last = m.end()
    parts.append(seed.text[last:])
    utterance = "".join(parts)
    return Filled(seed=seed, utterance=utterance, spans=tuple(spans))


def _distinct(seed: Seed, rng: random.Random, n: int, sensitive: bool) -> list[Filled]:
    """Up to ``n`` fills with pairwise different utterances."""
    out: list[Filled] = []
    seen: set[str] = set()
    for _ in range(n * 20):
        if len(out) == n:
            break
        f = fill(seed, rng, sensitive)
        if f.utterance not in seen:
            seen.add(f.utterance)
            out.append(f)
    return out


def fills_for_seed(
    seed: Seed, rng: random.Random, n_sensitive: int = 2, n_generic: int = 1
) -> list[Filled]:
    """All fills of one seed for the pilot.

    * A seed with sensitive slots gives ``n_sensitive`` sensitive fills, plus
      ``n_generic`` generic fills (relation word, no spans) when ``{contact}``
      is its only sensitive slot.
    * A seed with only non-sensitive slots gives 2 fills with different values.
    * A seed with no slots gives 1 fill (the seed text itself).
    """
    sens = _sensitive_slots(seed)
    if sens:
        out = _distinct(seed, rng, n_sensitive, sensitive=True)
        if sens == GENERIC_CONTACT_ONLY:
            out += _distinct(seed, rng, n_generic, sensitive=False)
        return out
    if seed.slots:
        return _distinct(seed, rng, 2, sensitive=True)
    return [fill(seed, rng)]


# --------------------------------------------------------------------------- #
# Tokens and BIO tags for the span head
# --------------------------------------------------------------------------- #

Tokenizer = Callable[[str], list[tuple[int, int]]]

_WORD = re.compile(r"\w+(?:[.:]\w+)*|[^\w\s]")


def word_tokenize(text: str) -> list[tuple[int, int]]:
    """Token offsets: whitespace-separated words, with punctuation split off.

    This is the default rather than a bare whitespace split because a name
    followed by a possessive ("priya's") would otherwise share a token with
    the "'s", and the span would no longer fall on token boundaries.
    Decimal and clock numbers ("93.5", "6:30") stay one token.
    """
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def whitespace_tokenize(text: str) -> list[tuple[int, int]]:
    """Token offsets from a plain whitespace split."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def tokens(utterance: str, tokenizer: Tokenizer = word_tokenize) -> list[str]:
    return [utterance[a:b] for a, b in tokenizer(utterance)]


def bio_tags(
    utterance: str, spans: Sequence[Span], tokenizer: Tokenizer = word_tokenize
) -> list[str]:
    """Per-token ``B-label`` / ``I-label`` / ``O`` tags.

    A token is inside a span if it overlaps it; the first such token is B.
    """
    tags: list[str] = []
    open_span: Span | None = None
    for a, b in tokenizer(utterance):
        hit = next((s for s in spans if a < s.end and b > s.start), None)
        if hit is None:
            tags.append("O")
            open_span = None
        elif hit is open_span:
            tags.append(f"I-{hit.label}")
        else:
            tags.append(f"B-{hit.label}")
            open_span = hit
    return tags


def spans_from_bio(
    utterance: str, tags: Sequence[str], tokenizer: Tokenizer = word_tokenize
) -> list[Span]:
    """Invert :func:`bio_tags`: merge B/I runs back into character spans."""
    offsets = tokenizer(utterance)
    if len(offsets) != len(tags):
        raise ValueError(f"{len(tags)} tags for {len(offsets)} tokens")
    out: list[Span] = []
    cur: list[int] | None = None  # [start, end]
    label = ""
    for (a, b), tag in zip(offsets, tags, strict=True):
        if tag.startswith("I-") and cur is not None and tag[2:] == label:
            cur[1] = b
            continue
        if cur is not None:
            out.append(Span(cur[0], cur[1], label, utterance[cur[0] : cur[1]]))
            cur = None
        if tag.startswith(("B-", "I-")):
            cur, label = [a, b], tag[2:]
    if cur is not None:
        out.append(Span(cur[0], cur[1], label, utterance[cur[0] : cur[1]]))
    return out
