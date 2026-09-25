"""Train / val / test / ood assignment, by seed.

Paraphrases of one seed are near-duplicates, so the split unit is the seed:
every row generated from a seed lands in that seed's split. The six intents
in the "Suggested OOD holdout" of ``docs/taxonomy.md`` go entirely to
``ood``. The remaining seeds are split 80/10/10 within each intent group, so
every group appears in train, val and test.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.schema import IntentGroup

TRAIN, VAL, TEST, OOD = "train", "val", "test", "ood"
SPLITS: tuple[str, ...] = (TRAIN, VAL, TEST, OOD)

OOD_INTENTS: tuple[str, ...] = (
    "disable_child_locks",
    "add_stop_along_route",
    "summarize_group_chat",
    "vehicle_manual_question",
    "plan_charging_stops",
    "add_calendar_event",
)


def seed_id(intent: str, index: int) -> str:
    """The stable id of a seed: its intent and its position in ``seeds.yaml``."""
    return f"{intent}/{index}"


def intent_of(sid: str) -> str:
    return sid.rsplit("/", 1)[0]


def assign_splits(
    seed_ids: Iterable[str],
    rng: random.Random,
    ood_intents: Iterable[str] = OOD_INTENTS,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    min_train_per_intent: int = 2,
) -> dict[str, str]:
    """Map every seed id to a split.

    Within each intent group the non-OOD seeds are shuffled and val and
    test each take ``round(fraction * n)`` of them (at least one). A seed
    is only moved out of train if its intent keeps at least
    ``min_train_per_intent`` training seeds, so no in-distribution intent
    is accidentally unseen in training. Deterministic given the rng state
    and the set of ids (input order does not matter).
    """
    ids = sorted(set(seed_ids))
    ood = set(ood_intents)
    unknown = ood - set(INTENTS_BY_NAME)
    if unknown:
        raise ValueError(f"unknown OOD intents: {sorted(unknown)}")
    out: dict[str, str] = {}
    by_group: dict[IntentGroup, list[str]] = {}
    for sid in ids:
        intent = intent_of(sid)
        if intent in ood:
            out[sid] = OOD
        else:
            by_group.setdefault(INTENTS_BY_NAME[intent].group, []).append(sid)

    for group in IntentGroup:
        members = by_group.get(group, [])
        if not members:
            continue
        rng.shuffle(members)
        n = len(members)
        want = {TEST: max(1, round(test_fraction * n)), VAL: max(1, round(val_fraction * n))}
        train_left: dict[str, int] = {}
        for sid in members:
            train_left[intent_of(sid)] = train_left.get(intent_of(sid), 0) + 1
        for sid in members:
            intent = intent_of(sid)
            target = next((s for s in (TEST, VAL) if want[s] > 0), TRAIN)
            if target != TRAIN and train_left[intent] > min_train_per_intent:
                want[target] -= 1
                train_left[intent] -= 1
                out[sid] = target
            else:
                out[sid] = TRAIN
        if want[TEST] or want[VAL]:
            raise ValueError(f"group {group} is too small to fill val and test")
    return out
