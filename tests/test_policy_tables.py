"""Table 1, 2 and 3: every cell pinned, plus the validator's invariants."""

import json

import pytest

from autogate_bench import (
    ActuationClass,
    Connectivity,
    DistractionLevel,
    DrivingDemand,
    Policy,
    Route,
    SpeedBucket,
    Workload,
)
from autogate_bench.policy import (
    DEFAULT_ACTUATION_GATE,
    DEFAULT_DEMAND_TABLE,
    DEFAULT_DISTRACTION_GATE,
)

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
