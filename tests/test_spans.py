"""Span injector: offsets, generic fills, BIO round-trip, pools, determinism."""

import random
import re

import pytest

from autogate_bench import SENSITIVE_SLOTS, SLOTS, Seed, SeedStyle, load_seeds
from autogate_bench.spans import (
    ADDRESSES,
    INTENT_POOLS,
    NAMES,
    POOLS,
    RELATIONS,
    Span,
    bio_tags,
    fill,
    fills_for_seed,
    gen_otp,
    gen_phone,
    spans_from_bio,
    tokens,
    whitespace_tokenize,
)

SEEDS = [s for seeds in load_seeds().values() for s in seeds]
SENSITIVE = [s for s in SEEDS if s.has_sensitive_slot]
CONTACT_ONLY = [s for s in SENSITIVE if {x for x in s.slots if x in SENSITIVE_SLOTS} == {"contact"}]
OTHER_SENSITIVE = [s for s in SENSITIVE if s not in CONTACT_ONLY]


def _all_fills(rng_seed: int = 0):
    rng = random.Random(rng_seed)
    return [f for s in SEEDS for f in fills_for_seed(s, rng, n_sensitive=2, n_generic=1)]


FILLS = _all_fills()


def test_every_span_matches_its_offsets():
    spans = [(f, s) for f in FILLS for s in f.spans]
    assert spans
    for f, s in spans:
        assert f.utterance[s.start : s.end] == s.text
        assert s.label in SENSITIVE_SLOTS


def test_no_placeholders_left():
    for f in FILLS:
        assert "{" not in f.utterance and "}" not in f.utterance
        assert f.utterance == f.utterance.lower()
        assert f.utterance.isascii()


def test_only_sensitive_slots_become_spans():
    for f in FILLS:
        n_sensitive = sum(1 for x in f.seed.slots if x in SENSITIVE_SLOTS)
        if f.spans:
            assert len(f.spans) == n_sensitive
        else:  # a generic fill, or a seed with no sensitive slot
            assert f.seed in CONTACT_ONLY or n_sensitive == 0


@pytest.mark.parametrize("seed", CONTACT_ONLY, ids=lambda s: s.text)
def test_generic_fill_has_no_spans(seed):
    f = fill(seed, random.Random(1), sensitive=False)
    assert f.spans == () and not f.has_sensitive_spans
    assert any(r in f.utterance for r in RELATIONS)


@pytest.mark.parametrize("seed", OTHER_SENSITIVE, ids=lambda s: s.text)
def test_generic_fill_rejected_for_non_contact_sensitive_slots(seed):
    with pytest.raises(ValueError, match="only valid"):
        fill(seed, random.Random(1), sensitive=False)


def test_fill_counts_per_seed():
    rng = random.Random(0)
    for s in SEEDS:
        fs = fills_for_seed(s, rng, n_sensitive=2, n_generic=1)
        if s in CONTACT_ONLY:
            assert [f.has_sensitive_spans for f in fs] == [True, True, False]
        elif s.has_sensitive_slot:
            assert [f.has_sensitive_spans for f in fs] == [True, True]
        elif s.slots:
            assert len(fs) == 2 and fs[0].utterance != fs[1].utterance
        else:
            assert [f.utterance for f in fs] == [s.text]


def test_repeated_slot_gets_one_value():
    seed = Seed("call_contact", "call {contact} no really call {contact}", SeedStyle.COMMAND)
    f = fill(seed, random.Random(3))
    assert len(f.spans) == 2 and f.spans[0].text == f.spans[1].text


def test_fills_are_deterministic():
    a = [(f.utterance, f.spans) for f in _all_fills(7)]
    b = [(f.utterance, f.spans) for f in _all_fills(7)]
    c = [(f.utterance, f.spans) for f in _all_fills(8)]
    assert a == b
    assert a != c


# ---- pools ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pool",
    [NAMES, RELATIONS, ADDRESSES, *POOLS.values(), *INTENT_POOLS.values()],
    ids=lambda p: p[0],
)
def test_pools_have_no_duplicates_and_are_lowercase_ascii(pool):
    assert len(pool) == len(set(pool))
    for v in pool:
        assert v == v.lower() and v.isascii() and v == v.strip()


def test_every_non_sensitive_slot_has_a_pool():
    assert set(POOLS) == SLOTS - SENSITIVE_SLOTS
    assert all(5 <= len(p) <= 15 for p in POOLS.values())


def test_pool_sizes():
    assert len(NAMES) >= 40 and len(RELATIONS) >= 20 and len(ADDRESSES) >= 20


@pytest.mark.parametrize("slot", ["message_text", "note_text"])
def test_free_text_pools_carry_no_pii(slot):
    for v in POOLS[slot]:
        assert not re.search(r"\d", v)
        assert not any(name.split()[0] in v.split() for name in NAMES)


def test_generated_numbers():
    rng = random.Random(0)
    for _ in range(200):
        digits = re.sub(r"\D", "", gen_phone(rng))
        assert len(digits) in (10, 12)
        assert digits[-10] in "6789"
        otp = gen_otp(rng)
        assert otp.isdigit() and len(otp) in (4, 6)


# ---- BIO ------------------------------------------------------------------- #


def test_bio_round_trips_on_every_fill():
    for f in FILLS:
        tags = bio_tags(f.utterance, f.spans)
        assert len(tags) == len(tokens(f.utterance))
        assert spans_from_bio(f.utterance, tags) == list(f.spans)


def test_bio_example():
    u = "navigate to priya natarajan's house"
    span = Span(12, 27, "contact", "priya natarajan")
    assert u[span.start : span.end] == span.text
    assert tokens(u) == ["navigate", "to", "priya", "natarajan", "'", "s", "house"]
    assert bio_tags(u, [span]) == ["O", "O", "B-contact", "I-contact", "O", "O", "O"]


def test_bio_adjacent_spans_stay_separate():
    u = "reply 1234 5678"
    spans = [Span(6, 10, "otp", "1234"), Span(11, 15, "otp", "5678")]
    assert bio_tags(u, spans) == ["O", "B-otp", "B-otp"]
    assert spans_from_bio(u, bio_tags(u, spans)) == spans


def test_bio_with_whitespace_tokenizer():
    u = "dial +91 98450 12345 now"
    span = Span(5, 20, "phone", "+91 98450 12345")
    tags = bio_tags(u, [span], tokenizer=whitespace_tokenize)
    assert tags == ["O", "B-phone", "I-phone", "I-phone", "O"]
    assert spans_from_bio(u, tags, tokenizer=whitespace_tokenize) == [span]
