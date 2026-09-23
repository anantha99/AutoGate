"""Hand-written seed utterances, the anchors of the AutoGate data pipeline.

Each intent has 3 to 5 seeds in ``data/seeds.yaml``. A paraphraser later
expands every seed into many variants, and the train/val/test split is made
by seed, so seeds within one intent are distinct ways of asking rather than
synonyms. Each seed carries one phrasing style and may contain ``{slot}``
placeholders; the sensitive ones are filled from synthetic pools by the span
injector, which records their offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from autogate_bench.intents import INTENTS_BY_NAME

DEFAULT_SEEDS_PATH = Path(__file__).resolve().parents[2] / "data" / "seeds.yaml"


class SeedStyle(StrEnum):
    """How a seed phrases its request."""

    COMMAND = "command"  # an imperative: "set the temperature to {temperature}"
    OUTCOME = "outcome"  # a need or situation, no action named: "it's boiling in here"
    QUESTION = "question"  # a request as a question: "can you make it a bit cooler"
    INDIRECT = "indirect"  # conversational, action implied: "the kids are complaining"
    FRAGMENT = "fragment"  # the terse form: "ac off"


SENSITIVE_SLOTS: frozenset[str] = frozenset({"contact", "phone", "address", "otp", "card"})

SLOTS: frozenset[str] = SENSITIVE_SLOTS | frozenset(
    {
        "place",
        "city",
        "temperature",
        "fan_level",
        "song",
        "artist",
        "station",
        "podcast",
        "time",
        "date",
        "duration",
        "team",
        "topic",
        "note_text",
        "message_text",
        "event_title",
        "drive_mode",
        "color",
        "speed",
        "seat",
        "window",
        "number",
    }
)

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


def _parse_slots(text: str) -> tuple[str, ...]:
    return tuple(m.group(1) for m in _PLACEHOLDER.finditer(text))


@dataclass(frozen=True, slots=True)
class Seed:
    """One hand-written utterance for one intent."""

    intent: str
    text: str
    style: SeedStyle

    @property
    def slots(self) -> tuple[str, ...]:
        """Slot names in the text, in order of appearance (repeats kept)."""
        return _parse_slots(self.text)

    @property
    def has_sensitive_slot(self) -> bool:
        return any(s in SENSITIVE_SLOTS for s in self.slots)


def _parse_item(intent: str, index: int, item: object) -> Seed:
    where = f"{intent}[{index}]"
    if not isinstance(item, dict) or set(item) != {"text", "style"}:
        raise ValueError(f"{where}: expected a mapping with exactly 'text' and 'style'")
    text, style = item["text"], item["style"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{where}: 'text' must be a non-empty string")
    if not isinstance(style, str):
        raise ValueError(f"{where}: 'style' must be a string")
    try:
        seed_style = SeedStyle(style)
    except ValueError:
        raise ValueError(f"{where}: unknown style {style!r}") from None
    for slot in _parse_slots(text):
        if slot not in SLOTS:
            raise ValueError(f"{where}: unknown slot {{{slot}}} in {text!r}")
    if "{" in _PLACEHOLDER.sub("", text) or "}" in _PLACEHOLDER.sub("", text):
        raise ValueError(f"{where}: unbalanced brace in {text!r}")
    return Seed(intent=intent, text=text, style=seed_style)


def load_seeds(path: str | Path = DEFAULT_SEEDS_PATH) -> dict[str, tuple[Seed, ...]]:
    """Parse the seeds file into ``{intent name: seeds}``, in file order.

    Raises ``ValueError`` on an unknown intent, style, or slot, or on any
    malformed entry. Completeness against ``INTENTS`` is a test, not a load
    error, so partial files can be loaded.
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of intent name to seed list")
    seeds: dict[str, tuple[Seed, ...]] = {}
    for intent, items in raw.items():
        if intent not in INTENTS_BY_NAME:
            raise ValueError(f"unknown intent {intent!r}")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{intent}: expected a non-empty list of seeds")
        seeds[intent] = tuple(_parse_item(intent, i, item) for i, item in enumerate(items))
    return seeds
