"""ASR noise: slots untouched, output always differs, deterministic."""

import random
import re
from collections import Counter

import pytest

from autogate_bench.asr_noise import (
    DROPPABLE,
    HOMOPHONES,
    KINDS,
    MERGE_SPLIT,
    asr_edits,
    asr_noise,
    asr_variants,
    candidate_edits,
)
from autogate_bench.paraphrases import load_paraphrases
from autogate_bench.seeds import load_seeds

SLOT = re.compile(r"\{[^{}]*\}(?:'s)?")

TEXTS = [
    "turn the wipers on",
    "can you call {contact} please",
    "navigate to {contact}'s house",
    "turn off lane assist for two minutes?",
    "text {contact} {message_text}",
    "{contact} ge video call maadu",
    "demist",
    "set the temperature to {temperature} please",
]
CORPUS = [s.text for items in load_seeds().values() for s in items] + [
    p.text
    for by in load_paraphrases("tests/fixtures/paraphrases").values()
    for items in by.values()
    for p in items
]


def test_tables_are_large_enough():
    assert sum(len(v) for v in HOMOPHONES.values()) >= 60
    assert {
        "to",
        "for",
        "there",
        "wipers",
        "heater",
        "seat",
        "call",
        "text",
        "defrost",
        "lane",
        "ac",
    } <= set(HOMOPHONES)
    assert DROPPABLE == {"the", "a", "to", "my", "please", "can", "you"}
    assert MERGE_SPLIT["lane assist"] == "laneassist" and MERGE_SPLIT["seatbelt"] == "seat belt"


@pytest.mark.parametrize("text", CORPUS)
def test_slots_untouched_and_output_differs(text):
    rng = random.Random(text)
    for _ in range(5):
        out = asr_noise(text, rng)
        if out is None:
            assert candidate_edits(text.removesuffix("?").split()) == []
            return
        assert out != text and out != text.removesuffix("?")
        assert SLOT.findall(out) == SLOT.findall(text)
        assert not out.endswith("?") and out.strip() == out and "  " not in out


def test_deterministic():
    a = [asr_noise(t, random.Random(7)) for t in TEXTS]
    b = [asr_noise(t, random.Random(7)) for t in TEXTS]
    assert a == b
    assert asr_variants(TEXTS[0], random.Random(1), 3) == asr_variants(
        TEXTS[0], random.Random(1), 3
    )


def test_none_when_nothing_applies():
    assert asr_noise("sos", random.Random(0)) is None
    assert asr_noise("{contact} {phone}", random.Random(0)) is None
    assert asr_variants("sos", random.Random(0), 3) == []


def test_never_edits_inside_a_slot():
    # "to" and "call" are editable outside the slot; the slot names are not
    for e in candidate_edits("call {call} to {to}".split()):
        assert e.start in (0, 2)


def test_every_kind_is_used_and_chosen_evenly():
    text = "can you play the one podcast to me"  # every kind applies
    assert {e.kind for e in candidate_edits(text.split())} == set(KINDS)
    rng = random.Random(0)
    first = Counter(asr_edits(text, rng)[0].kind for _ in range(800))
    # more homophone candidates than droppable words, yet each kind is drawn ~1/4 of the time
    assert all(150 < first[k] < 250 for k in KINDS), first
    rng = random.Random(0)
    n_edits = Counter(len(asr_edits(text, rng)) for _ in range(800))
    assert set(n_edits) == {1, 2} and n_edits[1] > n_edits[2]


def test_variants_are_distinct():
    out = asr_variants("turn the wipers on to the max please", random.Random(3), 5)
    assert len(out) == len(set(out)) == 5
