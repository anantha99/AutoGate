"""Table 1, 2 and 3: every cell pinned, plus the validator's invariants."""

import json

import pytest

from autogate_bench import (
    ActuationClass,
    Capability,
    Connectivity,
    Context,
    DistractionLevel,
    DrivingDemand,
    Policy,
    Route,
    SpeedBucket,
    Workload,
)
from autogate_bench.contexts import ALL_CONTEXTS
from autogate_bench.intents import INTENTS
from autogate_bench.intents import INTENTS_BY_NAME as I
from autogate_bench.policy import (
    DEFAULT_ACTUATION_GATE,
    DEFAULT_CAPABILITY,
    DEFAULT_DEMAND_TABLE,
    DEFAULT_DISTRACTION_GATE,
)
from autogate_bench.rulebook import route

C = Capability
D = DrivingDemand
S = SpeedBucket
W = Workload
L = DistractionLevel
A = ActuationClass


# ---- Table 1: driving demand -------------------------------------------- #


@pytest.mark.parametrize(
    ("speed", "workload", "expected"),
    [
        (S.PARKED, W.LOW, D.PARKED),
        (S.PARKED, W.MEDIUM, D.PARKED),
        (S.PARKED, W.HIGH, D.PARKED),
        (S.LOW, W.LOW, D.LOW),
        (S.LOW, W.MEDIUM, D.MEDIUM),
        (S.LOW, W.HIGH, D.HIGH),
        (S.MEDIUM, W.LOW, D.MEDIUM),
        (S.MEDIUM, W.MEDIUM, D.MEDIUM),
        (S.MEDIUM, W.HIGH, D.HIGH),
        (S.HIGH, W.LOW, D.MEDIUM),
        (S.HIGH, W.MEDIUM, D.HIGH),
        (S.HIGH, W.HIGH, D.HIGH),
    ],
)
def test_demand_table_cell(speed, workload, expected):
    assert Policy().demand(speed, workload) is expected


def test_demand_table_is_complete():
    assert set(DEFAULT_DEMAND_TABLE) == {(s, w) for s in S for w in W}


# ---- Table 2: distraction gate ------------------------------------------ #


@pytest.mark.parametrize(
    ("level", "demand", "allowed"),
    [
        (L.LOW, D.PARKED, True),
        (L.LOW, D.LOW, True),
        (L.LOW, D.MEDIUM, True),
        (L.LOW, D.HIGH, True),
        (L.MEDIUM, D.PARKED, True),
        (L.MEDIUM, D.LOW, True),
        (L.MEDIUM, D.MEDIUM, True),
        (L.MEDIUM, D.HIGH, False),
        (L.HIGH, D.PARKED, True),
        (L.HIGH, D.LOW, False),
        (L.HIGH, D.MEDIUM, False),
        (L.HIGH, D.HIGH, False),
    ],
)
def test_distraction_gate_cell(level, demand, allowed):
    assert Policy().distraction_allowed(level, demand) is allowed


def test_distraction_gate_is_complete():
    assert set(DEFAULT_DISTRACTION_GATE) == {(lv, d) for lv in L for d in D}


# ---- Table 3: actuation gate -------------------------------------------- #


@pytest.mark.parametrize("demand", list(D))
def test_safety_critical_always_allowed(demand):
    assert Policy().actuation_allowed(A.SAFETY_CRITICAL, demand) is True


@pytest.mark.parametrize("demand", list(D))
def test_comfort_always_allowed(demand):
    assert Policy().actuation_allowed(A.COMFORT, demand) is True


@pytest.mark.parametrize(
    ("demand", "allowed"),
    [(D.PARKED, True), (D.LOW, False), (D.MEDIUM, False), (D.HIGH, False)],
)
def test_restricted_only_when_parked(demand, allowed):
    assert Policy().actuation_allowed(A.RESTRICTED, demand) is allowed


def test_actuation_gate_is_complete():
    gated = {c for c in A if c is not A.NONE}
    assert set(DEFAULT_ACTUATION_GATE) == {(c, d) for c in gated for d in D}


# ---- Table 4: capability -------------------------------------------------- #


def test_capability_table_covers_the_taxonomy():
    assert set(DEFAULT_CAPABILITY) == {i.name for i in INTENTS}


def test_default_capability_counts():
    """Conservative default: only what a tiny head-unit model handles is local_ok."""
    counts = {c: list(DEFAULT_CAPABILITY.values()).count(c) for c in C}
    assert counts == {
        C.LOCAL_OK: 43,
        C.NEEDS_SMALL_LOCAL: 8,
        C.NEEDS_CLOUD: 10,
        C.CLOUD_PREFERRED: 4,
    }


def test_every_vehicle_command_is_local_ok_by_default():
    commands = [i for i in INTENTS if i.actuation is not A.NONE]
    assert {Policy().capability_of(i) for i in commands} == {C.LOCAL_OK}


def test_capability_override_changes_routes():
    """A stronger onboard model: messages are read locally even on the tiny tier."""
    ctx = Context(S.PARKED, Connectivity.GOOD, W.LOW)  # local_model_tier defaults to tiny
    strong = Policy(capability={**DEFAULT_CAPABILITY, "read_messages": C.LOCAL_OK})
    assert route(I["read_messages"], ctx).route is Route.CLOUD
    assert route(I["read_messages"], ctx, policy=strong).route is Route.LOCAL


# ---- Locked rules ----------------------------------------------------------- #


@pytest.mark.parametrize("name", ["wipers_on", "unlock_doors", "set_cabin_temperature"])
@pytest.mark.parametrize("capability", [C.NEEDS_CLOUD, C.NEEDS_SMALL_LOCAL, C.CLOUD_PREFERRED])
def test_validator_rejects_vehicle_command_off_the_car(name, capability):
    with pytest.raises(ValueError, match="vehicle commands must be local_ok"):
        Policy(capability={**DEFAULT_CAPABILITY, name: capability})


def test_validator_rejects_missing_capability():
    table = dict(DEFAULT_CAPABILITY)
    del table["weather_now"]
    with pytest.raises(ValueError, match="capability missing"):
        Policy(capability=table)


def test_validator_rejects_unknown_capability_intent():
    with pytest.raises(ValueError, match="unknown intents"):
        Policy(capability={**DEFAULT_CAPABILITY, "launch_rocket": C.LOCAL_OK})


@pytest.mark.parametrize("extreme", [C.LOCAL_OK, C.NEEDS_CLOUD])
def test_no_capability_list_moves_safety_or_restricted_routes(extreme):
    """Widen or narrow every non-command intent: safety-critical stays LOCAL and
    restricted commands stay refused while moving, in every context."""
    table = {
        name: (cap if I[name].actuation is not A.NONE else extreme)
        for name, cap in DEFAULT_CAPABILITY.items()
    }
    policy = Policy(capability=table)
    for intent in INTENTS:
        if intent.actuation is A.NONE:
            continue
        for ctx in ALL_CONTEXTS:
            for sensitive in (False, True):
                assert route(intent, ctx, sensitive, policy) == route(intent, ctx, sensitive)
        if intent.actuation is A.SAFETY_CRITICAL:
            assert {route(intent, c, policy=policy).route for c in ALL_CONTEXTS} == {Route.LOCAL}
        if intent.actuation is A.RESTRICTED:
            moving = [c for c in ALL_CONTEXTS if c.speed_bucket is not S.PARKED]
            assert {route(intent, c, policy=policy).route for c in moving} == {Route.REFUSE}


# ---- Validator ----------------------------------------------------------- #


def test_validator_rejects_missing_cell():
    table = dict(DEFAULT_DEMAND_TABLE)
    del table[(S.HIGH, W.HIGH)]
    with pytest.raises(ValueError, match="demand_table missing"):
        Policy(demand_table=table)


def test_validator_rejects_blocked_safety_critical():
    gate = dict(DEFAULT_ACTUATION_GATE)
    gate[(A.SAFETY_CRITICAL, D.HIGH)] = False
    with pytest.raises(ValueError, match="safety_critical"):
        Policy(actuation_gate=gate)


def test_validator_rejects_parked_with_demand():
    table = dict(DEFAULT_DEMAND_TABLE)
    table[(S.PARKED, W.HIGH)] = D.HIGH
    with pytest.raises(ValueError, match="PARKED"):
        Policy(demand_table=table)


def test_validator_rejects_non_monotone_demand():
    gate = dict(DEFAULT_DISTRACTION_GATE)
    gate[(L.MEDIUM, D.MEDIUM)] = False  # refused at MEDIUM but allowed at HIGH? no, HIGH is False
    gate[(L.MEDIUM, D.HIGH)] = True  # now allowed at HIGH but not at MEDIUM
    with pytest.raises(ValueError, match="monotone in demand"):
        Policy(distraction_gate=gate)


def test_validator_rejects_non_monotone_level():
    gate = dict(DEFAULT_DISTRACTION_GATE)
    gate[(L.LOW, D.HIGH)] = False
    gate[(L.MEDIUM, D.HIGH)] = True
    with pytest.raises(ValueError, match="monotone in level"):
        Policy(distraction_gate=gate)


def test_validator_rejects_strict_privacy_to_cloud():
    with pytest.raises(ValueError, match="strict privacy"):
        Policy(strict_privacy_route=Route.CLOUD_MASKED)


# ---- Serialization -------------------------------------------------------- #


def test_policy_round_trips_through_json(tmp_path):
    policy = Policy(
        passenger_relax_steps=0,
        cloud_connectivity=frozenset({Connectivity.GOOD}),
        strict_privacy_route=Route.REFUSE,
    )
    path = tmp_path / "policy.json"
    policy.save(path)
    loaded = Policy.load(path)
    assert loaded == policy
    # And the file is plain JSON an OEM can edit by hand.
    data = json.loads(path.read_text())
    assert data["demand_table"]["high|low"] == "medium"
    assert data["distraction_gate"]["high|parked"] is True


def test_policy_file_may_list_only_widened_capabilities():
    data = Policy().to_dict()
    data["capability"] = {"read_messages": "local_ok"}
    policy = Policy.from_dict(data)
    assert policy.capability_of("read_messages") is C.LOCAL_OK
    assert policy.capability_of("weather_now") is C.NEEDS_CLOUD


def test_policy_file_without_capability_uses_the_default():
    data = Policy().to_dict()
    del data["capability"]
    assert Policy.from_dict(data) == Policy()


def test_capability_round_trips_through_json(tmp_path):
    policy = Policy(capability={**DEFAULT_CAPABILITY, "general_knowledge_question": C.LOCAL_OK})
    path = tmp_path / "policy.json"
    policy.save(path)
    assert Policy.load(path) == policy
    assert json.loads(path.read_text())["capability"]["general_knowledge_question"] == "local_ok"
