"""Fixed label vocabularies for the router's heads.

The index of a label in these tuples is the index of its logit, so they are
written to ``labels.json`` beside every checkpoint and read back from there
at prediction time. A checkpoint therefore never depends on the order of
``Route`` or ``INTENTS`` in whatever version of the code loads it.

* ``ROUTES``: the five routes in ``autogate_bench.schema.Route`` order.
* ``INTENTS``: the intent names (81) in ``autogate_bench.intents.INTENTS`` order.
* ``BIO_TAGS``: ``O`` then ``B-``/``I-`` for each sensitive span label, in
  the fixed order contact, phone, address, otp, card (11 tags).
* ``CAPABILITIES``, ``DISTRACTIONS``, ``ACTUATIONS``, ``CONSEQUENCES``: the
  four policy tags, each in its ``autogate_bench.schema`` enum order
  (``Capability``, ``DistractionLevel``, ``ActuationClass``,
  ``Consequence``). ``TAG_NAMES`` names them in the order the tag heads are
  built: capability, distraction, actuation, consequence.

``labels.json`` written before the tag heads existed has no tag keys; it
still loads, with ``Labels.tags`` empty, and the model is then built
without tag heads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from autogate_bench.intents import INTENTS as _INTENTS
from autogate_bench.schema import ActuationClass, Capability, Consequence, DistractionLevel, Route
from autogate_bench.seeds import SENSITIVE_SLOTS

LABELS_FILE = "labels.json"

ROUTES: tuple[str, ...] = tuple(str(r) for r in Route)
INTENTS: tuple[str, ...] = tuple(i.name for i in _INTENTS)
SPAN_LABELS: tuple[str, ...] = ("contact", "phone", "address", "otp", "card")
BIO_TAGS: tuple[str, ...] = ("O", *(f"{p}-{lab}" for lab in SPAN_LABELS for p in ("B", "I")))

CAPABILITIES: tuple[str, ...] = tuple(str(c) for c in Capability)
DISTRACTIONS: tuple[str, ...] = tuple(str(d) for d in DistractionLevel)
ACTUATIONS: tuple[str, ...] = tuple(str(a) for a in ActuationClass)
CONSEQUENCES: tuple[str, ...] = tuple(str(c) for c in Consequence)
TAG_NAMES: tuple[str, ...] = ("capability", "distraction", "actuation", "consequence")
TAG_VOCABS: dict[str, tuple[str, ...]] = {
    "capability": CAPABILITIES,
    "distraction": DISTRACTIONS,
    "actuation": ACTUATIONS,
    "consequence": CONSEQUENCES,
}

assert set(SPAN_LABELS) == set(SENSITIVE_SLOTS), "span labels drifted from SENSITIVE_SLOTS"

IGNORE_INDEX = -100


@dataclass(frozen=True)
class Labels:
    """The vocabularies plus name -> index maps.

    ``tags`` maps each tag name (``TAG_NAMES`` order) to its vocabulary; it
    is empty for labels saved before the tag heads existed.
    """

    routes: tuple[str, ...] = ROUTES
    intents: tuple[str, ...] = INTENTS
    bio_tags: tuple[str, ...] = BIO_TAGS
    tags: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(TAG_VOCABS))
    route_index: dict[str, int] = field(init=False, repr=False, compare=False)
    intent_index: dict[str, int] = field(init=False, repr=False, compare=False)
    bio_index: dict[str, int] = field(init=False, repr=False, compare=False)
    tag_index: dict[str, dict[str, int]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        tags = {k: tuple(v) for k, v in self.tags.items()}
        unknown = set(tags) - set(TAG_NAMES)
        if unknown:
            raise ValueError(f"unknown tag names {sorted(unknown)}")
        if tags and set(tags) != set(TAG_NAMES):
            raise ValueError(f"tags must be all of {TAG_NAMES} or none")
        tags = {k: tags[k] for k in TAG_NAMES if k in tags}
        object.__setattr__(self, "tags", tags)
        for name, values in (
            ("routes", self.routes),
            ("intents", self.intents),
            ("bio_tags", self.bio_tags),
            *tags.items(),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate entries in {name}")
        if not self.bio_tags or self.bio_tags[0] != "O":
            raise ValueError("bio_tags must start with 'O'")
        object.__setattr__(self, "route_index", {r: i for i, r in enumerate(self.routes)})
        object.__setattr__(self, "intent_index", {r: i for i, r in enumerate(self.intents)})
        object.__setattr__(self, "bio_index", {r: i for i, r in enumerate(self.bio_tags)})
        object.__setattr__(
            self,
            "tag_index",
            {k: {v: i for i, v in enumerate(vals)} for k, vals in tags.items()},
        )

    @property
    def has_tags(self) -> bool:
        return bool(self.tags)

    def to_dict(self) -> dict[str, list[str]]:
        out = {
            "routes": list(self.routes),
            "intents": list(self.intents),
            "bio_tags": list(self.bio_tags),
        }
        for name, key in zip(TAG_NAMES, _TAG_KEYS, strict=True):
            if name in self.tags:
                out[key] = list(self.tags[name])
        return out

    @classmethod
    def from_dict(cls, d: dict) -> Labels:
        tags = {
            name: tuple(d[key]) for name, key in zip(TAG_NAMES, _TAG_KEYS, strict=True) if key in d
        }
        return cls(tuple(d["routes"]), tuple(d["intents"]), tuple(d["bio_tags"]), tags)

    def without_tags(self) -> Labels:
        return Labels(self.routes, self.intents, self.bio_tags, {})


# keys in labels.json, one per tag name
_TAG_KEYS: tuple[str, ...] = ("capabilities", "distractions", "actuations", "consequences")

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
