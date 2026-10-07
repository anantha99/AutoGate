"""Structural checks on the intent taxonomy in ``autogate_bench.intents``."""

import re
from collections import Counter

import pytest

from autogate_bench import (
    ActuationClass,
    Capability,
    Connectivity,
    Consequence,
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
from autogate_bench.policy import DEFAULT_POLICY
from autogate_bench.schema import Intent

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def test_intent_count_in_range():
    assert 58 <= len(INTENTS) <= 66


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
    assert set(DEFAULT_POLICY.capability.values()) == set(Capability)
    assert {i.distraction for i in INTENTS} == set(DistractionLevel)
    assert {i.actuation for i in INTENTS} == set(ActuationClass)
    assert {i.consequence for i in INTENTS} == set(Consequence)


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
    capabilities = [DEFAULT_POLICY.capability_of(i) for i in INTENTS]
    assert capabilities.count(Capability.NEEDS_CLOUD) >= 5
    assert capabilities.count(Capability.NEEDS_SMALL_LOCAL) >= 5


PARKED_GOOD = Context(SpeedBucket.PARKED, Connectivity.GOOD, Workload.LOW)


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_every_intent_routes_when_parked_with_good_connectivity(intent):
    d = route(intent, PARKED_GOOD)
    assert isinstance(d, Decision)
    assert d.route in set(Route) and d.reason in set(Reason)
    assert d.demand is DrivingDemand.PARKED
    # Parked with signal, nothing is refused or deferred.
    assert d.route in (Route.LOCAL, Route.CLOUD)


# ---- consequence tag ------------------------------------------------------ #


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_vehicle_commands_have_a_side_effect(intent):
    if intent.actuation is not ActuationClass.NONE:
        assert intent.consequence is not Consequence.NONE


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_information_intents_are_read_only(intent):
    if intent.group is IntentGroup.INFORMATION:
        assert intent.consequence is Consequence.NONE


def test_irreversible_intents_are_the_ones_another_party_sees():
    irreversible = {i.name for i in INTENTS if i.consequence is Consequence.IRREVERSIBLE}
    assert irreversible == {
        "emergency_call",
        "call_contact",
        "send_text_message",
        "reply_to_message",
        "start_video_call",
        "compose_long_email",
        "book_restaurant",
        "add_calendar_event",
    }


def test_consequence_never_changes_the_route():
    """Two intents identical except for consequence route identically everywhere."""
    base = INTENTS_BY_NAME["send_text_message"]
    twin = Intent(
        base.name,
        base.group,
        base.distraction,
        base.actuation,
        base.description,
        consequence=Consequence.NONE,
    )
    for speed in SpeedBucket:
        for conn in Connectivity:
            for work in Workload:
                ctx = Context(speed, conn, work)
                assert route(base, ctx) == route(twin, ctx)


def test_schema_rejects_vehicle_command_without_side_effect():
    with pytest.raises(ValueError, match="side effect"):
        Intent(
            "x",
            IntentGroup.VEHICLE_CONTROL,
            DistractionLevel.LOW,
            ActuationClass.COMFORT,
            consequence=Consequence.NONE,
        )
