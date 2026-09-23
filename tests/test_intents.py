"""Structural checks on the intent taxonomy in ``autogate_bench.intents``."""

import re
from collections import Counter

import pytest

from autogate_bench import (
    ActuationClass,
    Capability,
    Connectivity,
    Context,
    Decision,
    DistractionLevel,
    DrivingDemand,
    IntentGroup,
    Reason,
    Route,
    SpeedBucket,
    Workload,
    route,
)
from autogate_bench.intents import INTENTS, INTENTS_BY_NAME

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def test_intent_count_in_range():
    assert 58 <= len(INTENTS) <= 62


def test_names_are_unique():
    counts = Counter(i.name for i in INTENTS)
    assert [n for n, c in counts.items() if c > 1] == []
    assert len(INTENTS_BY_NAME) == len(INTENTS)


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_name_is_snake_case(intent):
    assert SNAKE_CASE.match(intent.name)


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_description_is_one_plain_line(intent):
    assert intent.description.strip() == intent.description
    assert intent.description and "\n" not in intent.description


def test_every_enum_value_is_covered():
    assert {i.group for i in INTENTS} == set(IntentGroup)
    assert {i.capability for i in INTENTS} == set(Capability)
    assert {i.distraction for i in INTENTS} == set(DistractionLevel)
    assert {i.actuation for i in INTENTS} == set(ActuationClass)


def test_every_group_has_at_least_four_intents():
    counts = Counter(i.group for i in INTENTS)
    assert {g: counts[g] for g in IntentGroup if counts[g] < 4} == {}


def test_vehicle_control_intents_are_low_distraction():
    bad = [
        i.name
        for i in INTENTS
        if i.group is IntentGroup.VEHICLE_CONTROL and i.distraction is not DistractionLevel.LOW
    ]
    assert bad == []


def test_restricted_and_distraction_refusals_are_disjoint():
    bad = [
        i.name
        for i in INTENTS
        if i.actuation is ActuationClass.RESTRICTED and i.distraction is not DistractionLevel.LOW
    ]
    assert bad == []


def test_context_flipping_cells_are_populated():
    """Each tag that makes the label depend on context has several intents behind it."""
    assert sum(i.distraction is DistractionLevel.MEDIUM for i in INTENTS) >= 5
    assert sum(i.distraction is DistractionLevel.HIGH for i in INTENTS) >= 5
    assert sum(i.actuation is ActuationClass.RESTRICTED for i in INTENTS) >= 4
    assert sum(i.capability is Capability.NEEDS_CLOUD for i in INTENTS) >= 5
    assert sum(i.capability is Capability.NEEDS_SMALL_LOCAL for i in INTENTS) >= 5


PARKED_GOOD = Context(SpeedBucket.PARKED, Connectivity.GOOD, Workload.LOW)


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_every_intent_routes_when_parked_with_good_connectivity(intent):
    d = route(intent, PARKED_GOOD)
    assert isinstance(d, Decision)
    assert d.route in set(Route) and d.reason in set(Reason)
    assert d.demand is DrivingDemand.PARKED
    # Parked with signal, nothing is refused or deferred.
    assert d.route in (Route.LOCAL, Route.CLOUD)
