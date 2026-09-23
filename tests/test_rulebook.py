"""One test per branch of the tree, then precedence, then exhaustive invariants."""

import itertools

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
    Intent,
    IntentGroup,
    LocalModelTier,
    Policy,
    PrivacyMode,
    Reason,
    Route,
    SpeedBucket,
    Workload,
    route,
)
from autogate_bench.intents import INTENTS
from autogate_bench.intents import INTENTS_BY_NAME as I
from autogate_bench.rulebook import distraction_demand, driving_demand

D = DrivingDemand


# ---- Branch 1: safety-critical -> LOCAL --------------------------------- #


def test_safety_critical_is_local_when_busy(busy):
    assert route(I["defrost_windshield"], busy) == Decision(
        Route.LOCAL, Reason.SAFETY_CRITICAL, D.HIGH
    )


def test_safety_critical_is_local_even_offline_and_strict():
    ctx = Context(SpeedBucket.HIGH, Connectivity.NONE, Workload.HIGH, PrivacyMode.STRICT)
    d = route(I["hazard_lights_on"], ctx, has_sensitive_spans=True)
    assert d.route is Route.LOCAL and d.reason is Reason.SAFETY_CRITICAL


# ---- Branch 2a: restricted actuation -> REFUSE -------------------------- #


def test_restricted_actuation_refused_while_moving(cruising):
    d = route(I["disable_lane_assist"], cruising)
    assert d == Decision(Route.REFUSE, Reason.RESTRICTED_ACTUATION, D.MEDIUM)


def test_restricted_actuation_allowed_when_parked(parked):
    d = route(I["unlock_doors"], parked)
    assert d == Decision(Route.LOCAL, Reason.LOCAL_CAPABLE, D.PARKED)


def test_passenger_never_unlocks_restricted_actuation():
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW, passenger_present=True)
    d = route(I["open_trunk"], ctx)
    assert d.route is Route.REFUSE and d.reason is Reason.RESTRICTED_ACTUATION


# ---- Branch 2b: distraction -> REFUSE ------------------------------------ #


def test_medium_distraction_refused_at_high_demand(busy):
    assert route(I["read_messages"], busy) == Decision(Route.REFUSE, Reason.DISTRACTION, D.HIGH)


def test_medium_distraction_allowed_while_cruising(cruising):
    ctx = Context(
        SpeedBucket.HIGH, Connectivity.GOOD, Workload.LOW, local_model_tier=LocalModelTier.SMALL
    )
    assert route(I["read_messages"], ctx) == Decision(Route.LOCAL, Reason.LOCAL_CAPABLE, D.MEDIUM)


def test_high_distraction_refused_at_any_speed():
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW)
    assert route(I["play_video"], ctx) == Decision(Route.REFUSE, Reason.DISTRACTION, D.LOW)


def test_high_distraction_allowed_when_parked(parked):
    assert route(I["show_photos"], parked).route is Route.LOCAL


def test_passenger_relaxes_distraction_gate(busy):
    with_passenger = Context(
        SpeedBucket.HIGH, Connectivity.GOOD, Workload.HIGH, passenger_present=True
    )
    assert route(I["calendar_today"], busy).route is Route.REFUSE
    d = route(I["calendar_today"], with_passenger)
    assert d == Decision(Route.LOCAL, Reason.LOCAL_CAPABLE, D.MEDIUM)


def test_passenger_relaxation_never_reaches_parked():
    """HIGH distraction needs PARKED; a passenger at low speed must not grant it."""
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW, passenger_present=True)
    assert distraction_demand(ctx) is D.LOW
    assert route(I["play_video"], ctx).route is Route.REFUSE


def test_passenger_relaxation_can_be_disabled(busy):
    with_passenger = Context(
        SpeedBucket.HIGH, Connectivity.GOOD, Workload.HIGH, passenger_present=True
    )
    policy = Policy(passenger_relax_steps=0)
    assert route(I["calendar_today"], with_passenger, policy=policy).route is Route.REFUSE


# ---- Branch 3: locally capable -> LOCAL ---------------------------------- #


def test_local_ok_intent_is_local_even_offline(offline):
    assert route(I["play_music"], offline) == Decision(Route.LOCAL, Reason.LOCAL_CAPABLE, D.LOW)


def test_needs_small_local_depends_on_tier(offline):
    tiny = offline
    small = Context(
        SpeedBucket.LOW, Connectivity.NONE, Workload.LOW, local_model_tier=LocalModelTier.SMALL
    )
    assert route(I["general_knowledge_question"], tiny).route is Route.DEFER
    assert route(I["general_knowledge_question"], small).route is Route.LOCAL


def test_local_ok_with_sensitive_spans_stays_local(cruising):
    d = route(I["call_contact"], cruising, has_sensitive_spans=True)
    assert d.route is Route.LOCAL and d.reason is Reason.LOCAL_CAPABLE


# ---- Branch 4: no connectivity -> DEFER ---------------------------------- #


def test_cloud_task_offline_is_deferred(offline):
    assert route(I["weather_now"], offline) == Decision(Route.DEFER, Reason.NO_CONNECTIVITY, D.LOW)


def test_weak_connectivity_counts_as_available_by_default():
    ctx = Context(SpeedBucket.LOW, Connectivity.WEAK, Workload.LOW)
    assert route(I["weather_now"], ctx).route is Route.CLOUD


def test_weak_connectivity_can_be_excluded_by_policy():
    ctx = Context(SpeedBucket.LOW, Connectivity.WEAK, Workload.LOW)
    policy = Policy(cloud_connectivity=frozenset({Connectivity.GOOD}))
    assert route(I["weather_now"], ctx, policy=policy).route is Route.DEFER


def test_defer_beats_masking(offline):
    d = route(I["plan_multi_day_trip"], offline, has_sensitive_spans=True)
    assert d.route is Route.DEFER


# ---- Branch 5: sensitive spans -------------------------------------------- #


def test_sensitive_spans_are_masked(cruising):
    d = route(I["navigate_to_contact_address"], cruising, has_sensitive_spans=True)
    assert d == Decision(Route.CLOUD_MASKED, Reason.SENSITIVE_SPANS, D.MEDIUM)


def test_strict_privacy_keeps_sensitive_spans_local():
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW, PrivacyMode.STRICT)
    d = route(I["plan_multi_day_trip"], ctx, has_sensitive_spans=True)
    assert d == Decision(Route.LOCAL, Reason.STRICT_PRIVACY, D.LOW)


def test_strict_privacy_route_is_configurable():
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW, PrivacyMode.STRICT)
    policy = Policy(strict_privacy_route=Route.REFUSE)
    d = route(I["plan_multi_day_trip"], ctx, has_sensitive_spans=True, policy=policy)
    assert d.route is Route.REFUSE and d.reason is Reason.STRICT_PRIVACY


def test_strict_privacy_without_sensitive_spans_goes_to_cloud():
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW, PrivacyMode.STRICT)
    assert route(I["plan_multi_day_trip"], ctx).route is Route.CLOUD


# ---- Branch 6: plain cloud ---------------------------------------------------- #


def test_cloud_default(cruising):
    assert route(I["plan_multi_day_trip"], cruising) == Decision(
        Route.CLOUD, Reason.CLOUD_DEFAULT, D.MEDIUM
    )


# ---- PRD examples, verbatim ------------------------------------------------- #


def test_prd_example_refuse_read_whatsapp_at_120():
    """'Disable lane assist and read me my WhatsApp' at 120 km/h -> REFUSE, both halves."""
    ctx = Context(SpeedBucket.HIGH, Connectivity.GOOD, Workload.MEDIUM)
    assert route(I["disable_lane_assist"], ctx).route is Route.REFUSE
    assert route(I["read_messages"], ctx).route is Route.REFUSE


def test_prd_example_defer_cloud_task_at_high_speed_no_signal():
    ctx = Context(SpeedBucket.HIGH, Connectivity.NONE, Workload.LOW)
    assert route(I["plan_multi_day_trip"], ctx).route is Route.DEFER


# ---- Precedence ---------------------------------------------------------------- #


def test_refuse_beats_defer(offline):
    """A distracting task with no signal is REFUSE, not DEFER: don't queue what you'd refuse."""
    busy_offline = Context(SpeedBucket.HIGH, Connectivity.NONE, Workload.HIGH)
    assert route(I["explain_topic_in_depth"], busy_offline).reason is Reason.DISTRACTION


def test_restricted_actuation_beats_distraction():
    """If an intent were both RESTRICTED and HIGH distraction, actuation fires first.

    The taxonomy keeps the two REFUSE mechanisms disjoint (see test_intents.py), so this
    uses a synthetic intent to pin the precedence of the tree itself.
    """
    both = Intent(
        "synthetic_restricted_and_distracting",
        IntentGroup.VEHICLE_CONTROL,
        Capability.LOCAL_OK,
        DistractionLevel.HIGH,
        ActuationClass.RESTRICTED,
        consequence=Consequence.REVERSIBLE,
    )
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW)
    assert route(both, ctx).reason is Reason.RESTRICTED_ACTUATION


def test_screen_heavy_setup_is_refused_by_distraction():
    """pair_bluetooth_device is a HIGH-distraction setup flow, not an actuation."""
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW)
    assert route(I["pair_bluetooth_device"], ctx).reason is Reason.DISTRACTION


# ---- Exhaustive invariants over every intent x every context ------------ #


def _all_contexts():
    for speed, conn, work, priv, pax, tier in itertools.product(
        SpeedBucket, Connectivity, Workload, PrivacyMode, (False, True), LocalModelTier
    ):
        yield Context(speed, conn, work, priv, pax, tier)


ALL_CONTEXTS = list(_all_contexts())


def test_context_space_size():
    assert len(ALL_CONTEXTS) == 4 * 3 * 3 * 2 * 2 * 2


@pytest.mark.parametrize("intent", INTENTS, ids=lambda i: i.name)
def test_invariants_hold_for_every_context(intent):
    for ctx in ALL_CONTEXTS:
        for sensitive in (False, True):
            d = route(intent, ctx, sensitive)
            # Safety-critical never leaves LOCAL.
            if intent.actuation is ActuationClass.SAFETY_CRITICAL:
                assert d.route is Route.LOCAL
            # Vehicle commands never go to the cloud or wait on it.
            if intent.actuation is not ActuationClass.NONE:
                assert d.route in (Route.LOCAL, Route.REFUSE)
            # Sensitive spans never reach the cloud unmasked.
            if sensitive:
                assert d.route is not Route.CLOUD
            # Strict privacy never sends sensitive spans out at all.
            if sensitive and ctx.privacy_mode is PrivacyMode.STRICT:
                assert d.route not in (Route.CLOUD, Route.CLOUD_MASKED)
            # Nothing cloud-bound without connectivity.
            if ctx.connectivity is Connectivity.NONE:
                assert d.route not in (Route.CLOUD, Route.CLOUD_MASKED)
            # Parked with a local-ok, non-safety intent is always LOCAL.
            if ctx.speed_bucket is SpeedBucket.PARKED and intent.capability is Capability.LOCAL_OK:
                assert d.route is Route.LOCAL


def test_demand_helpers_agree_with_policy():
    ctx = Context(SpeedBucket.HIGH, Connectivity.GOOD, Workload.HIGH, passenger_present=True)
    assert driving_demand(ctx) is D.HIGH
    assert distraction_demand(ctx) is D.MEDIUM


# ---- Intent schema guards ------------------------------------------------------ #


def test_intent_rejects_actuation_outside_vehicle_control():
    with pytest.raises(ValueError, match="only vehicle_control"):
        Intent(
            "x",
            IntentGroup.MEDIA,
            Capability.LOCAL_OK,
            DistractionLevel.LOW,
            ActuationClass.COMFORT,
            consequence=Consequence.REVERSIBLE,
        )


def test_intent_requires_actuation_for_vehicle_control():
    with pytest.raises(ValueError, match="need an actuation class"):
        Intent(
            "x",
            IntentGroup.VEHICLE_CONTROL,
            Capability.LOCAL_OK,
            DistractionLevel.LOW,
            consequence=Consequence.REVERSIBLE,
        )


def test_intent_rejects_cloud_vehicle_command():
    with pytest.raises(ValueError, match="local_ok"):
        Intent(
            "x",
            IntentGroup.VEHICLE_CONTROL,
            Capability.NEEDS_CLOUD,
            DistractionLevel.LOW,
            ActuationClass.COMFORT,
            consequence=Consequence.REVERSIBLE,
        )


def test_context_prefix_serialization():
    ctx = Context(SpeedBucket.HIGH, Connectivity.NONE, Workload.HIGH)
    assert ctx.to_prefix() == (
        "[speed=high][conn=none][workload=high][privacy=standard][passenger=no][local=tiny]"
    )
