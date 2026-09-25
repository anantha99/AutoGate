"""Fixed label vocabularies for the router's three heads.

The index of a label in these tuples is the index of its logit, so they are
written to ``labels.json`` beside every checkpoint and read back from there
at prediction time. A checkpoint therefore never depends on the order of
``Route`` or ``INTENTS`` in whatever version of the code loads it.

* ``ROUTES``: the five routes in ``autogate_bench.schema.Route`` order.
* ``INTENTS``: the 65 intent names in ``autogate_bench.intents.INTENTS`` order.
* ``BIO_TAGS``: ``O`` then ``B-``/``I-`` for each sensitive span label, in
  the fixed order contact, phone, address, otp, card (11 tags).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from autogate_bench.intents import INTENTS as _INTENTS
from autogate_bench.schema import Route
from autogate_bench.seeds import SENSITIVE_SLOTS

LABELS_FILE = "labels.json"

ROUTES: tuple[str, ...] = tuple(str(r) for r in Route)
INTENTS: tuple[str, ...] = tuple(i.name for i in _INTENTS)
SPAN_LABELS: tuple[str, ...] = ("contact", "phone", "address", "otp", "card")
BIO_TAGS: tuple[str, ...] = ("O", *(f"{p}-{lab}" for lab in SPAN_LABELS for p in ("B", "I")))

assert set(SPAN_LABELS) == set(SENSITIVE_SLOTS), "span labels drifted from SENSITIVE_SLOTS"

IGNORE_INDEX = -100


@dataclass(frozen=True)
class Labels:
    """The three vocabularies plus name -> index maps."""

    routes: tuple[str, ...] = ROUTES
    intents: tuple[str, ...] = INTENTS
    bio_tags: tuple[str, ...] = BIO_TAGS
    route_index: dict[str, int] = field(init=False, repr=False, compare=False)
    intent_index: dict[str, int] = field(init=False, repr=False, compare=False)
    bio_index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, values in (
            ("routes", self.routes),
            ("intents", self.intents),
            ("bio_tags", self.bio_tags),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate entries in {name}")
        if not self.bio_tags or self.bio_tags[0] != "O":
            raise ValueError("bio_tags must start with 'O'")
        object.__setattr__(self, "route_index", {r: i for i, r in enumerate(self.routes)})
        object.__setattr__(self, "intent_index", {r: i for i, r in enumerate(self.intents)})
        object.__setattr__(self, "bio_index", {r: i for i, r in enumerate(self.bio_tags)})

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "routes": list(self.routes),
            "intents": list(self.intents),
            "bio_tags": list(self.bio_tags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Labels:
        return cls(tuple(d["routes"]), tuple(d["intents"]), tuple(d["bio_tags"]))


DEFAULT_LABELS = Labels()


def save_labels(directory: str | Path, labels: Labels = DEFAULT_LABELS) -> Path:
    """Write ``labels.json`` into ``directory`` (created if needed)."""
    path = Path(directory) / LABELS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_labels(directory: str | Path) -> Labels:
    """Read ``labels.json`` from ``directory`` (or from the file path itself)."""
    path = Path(directory)
    if path.is_dir():
        path = path / LABELS_FILE
    return Labels.from_dict(json.loads(path.read_text(encoding="utf-8")))
