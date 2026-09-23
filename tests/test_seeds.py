"""Structural checks on the seed phrasings in ``data/seeds.yaml``."""

from collections import Counter

import pytest

from autogate_bench import (
    SENSITIVE_SLOTS,
    SLOTS,
    ActuationClass,
    DistractionLevel,
    Seed,
    SeedStyle,
    load_seeds,
)
from autogate_bench.intents import INTENTS, INTENTS_BY_NAME

SEEDS = load_seeds()
ALL_SEEDS = [s for seeds in SEEDS.values() for s in seeds]

PERSON_TARGETING = (
    "call_contact",
    "lookup_contact_info",
    "send_text_message",
    "reply_to_message",
    "start_video_call",
    "compose_long_email",
    "navigate_to_contact_address",
)

RECEIVED_ONLY = ("read_messages", "summarize_group_chat", "play_voicemail")

MAX_WORDS = 12


def test_every_intent_has_seeds_and_nothing_else():
    assert set(SEEDS) == {i.name for i in INTENTS}


def test_file_order_matches_intents():
    assert list(SEEDS) == [i.name for i in INTENTS]


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_three_to_five_seeds(intent):
    assert 3 <= len(SEEDS[intent.name]) <= 5


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_style_coverage(intent):
    styles = {s.style for s in SEEDS[intent.name]}
    assert SeedStyle.COMMAND in styles
    assert len(styles) >= 3


@pytest.mark.parametrize(
    "intent",
    [i for i in INTENTS if i.actuation is ActuationClass.RESTRICTED],
    ids=lambda i: i.name,
)
def test_restricted_intents_have_a_question(intent):
    assert SeedStyle.QUESTION in {s.style for s in SEEDS[intent.name]}


@pytest.mark.parametrize(
    "intent",
    [i for i in INTENTS if i.distraction is not DistractionLevel.LOW],
    ids=lambda i: i.name,
)
def test_long_response_intents_have_an_implicit_seed(intent):
    styles = {s.style for s in SEEDS[intent.name]}
    assert styles & {SeedStyle.OUTCOME, SeedStyle.INDIRECT}


def test_texts_are_unique_across_the_file():
    counts = Counter(s.text.strip().lower() for s in ALL_SEEDS)
    assert [t for t, c in counts.items() if c > 1] == []


@pytest.mark.parametrize("seed", ALL_SEEDS, ids=lambda s: s.text)
def test_text_form(seed):
    assert len(seed.text.split()) <= MAX_WORDS
    assert seed.text.isascii()
    assert seed.text == seed.text.lower()
    assert seed.text == seed.text.strip()
    assert "  " not in seed.text
    assert not seed.text.endswith((".", ",", "!", ";", ":"))


@pytest.mark.parametrize("seed", ALL_SEEDS, ids=lambda s: s.text)
def test_only_known_slots(seed):
    assert set(seed.slots) <= SLOTS


@pytest.mark.parametrize("name", PERSON_TARGETING)
def test_person_targeting_intents_have_a_contact_seed(name):
    assert name in INTENTS_BY_NAME
    assert any("contact" in s.slots for s in SEEDS[name])


@pytest.mark.parametrize("name", RECEIVED_ONLY)
def test_no_dictated_body_in_received_message_intents(name):
    assert all("message_text" not in s.slots for s in SEEDS[name])


def test_sensitive_slots_are_used():
    used = {slot for s in ALL_SEEDS for slot in s.slots}
    assert SENSITIVE_SLOTS <= used


def test_seed_properties():
    s = Seed("send_text_message", "text {contact} the otp is {otp} at {time}", SeedStyle.COMMAND)
    assert s.slots == ("contact", "otp", "time")
    assert s.has_sensitive_slot
    assert not Seed("play_music", "play {song}", SeedStyle.COMMAND).has_sensitive_slot


# ---- loader rejects bad input --------------------------------------------- #


def _write(tmp_path, body: str):
    p = tmp_path / "seeds.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loader_accepts_a_minimal_file(tmp_path):
    p = _write(tmp_path, 'call_contact:\n  - {text: "call {contact}", style: command}\n')
    seeds = load_seeds(p)
    assert seeds == {"call_contact": (Seed("call_contact", "call {contact}", SeedStyle.COMMAND),)}


def test_loader_rejects_unknown_style(tmp_path):
    p = _write(tmp_path, 'call_contact:\n  - {text: "call {contact}", style: shout}\n')
    with pytest.raises(ValueError, match="unknown style"):
        load_seeds(p)


def test_loader_rejects_unknown_slot(tmp_path):
    p = _write(tmp_path, 'call_contact:\n  - {text: "call {person}", style: command}\n')
    with pytest.raises(ValueError, match="unknown slot"):
        load_seeds(p)


def test_loader_rejects_unknown_intent(tmp_path):
    p = _write(tmp_path, 'order_pizza:\n  - {text: "order a pizza", style: command}\n')
    with pytest.raises(ValueError, match="unknown intent"):
        load_seeds(p)


@pytest.mark.parametrize(
    "body",
    [
        "call_contact:\n  - call someone\n",
        'call_contact:\n  - {text: "call {contact}"}\n',
        'call_contact:\n  - {text: "call {contact}", style: command, extra: 1}\n',
        'call_contact:\n  - {text: "call {contact", style: command}\n',
        "call_contact: []\n",
        "- just a list\n",
    ],
)
def test_loader_rejects_malformed_items(tmp_path, body):
    with pytest.raises(ValueError):
        load_seeds(_write(tmp_path, body))
