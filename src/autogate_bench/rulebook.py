"""The precedence tree: (intent, context, sensitive?) -> route.

Plain-text order, from the PRD:

    safety-critical command        -> LOCAL
    unsafe for driving state       -> REFUSE   (restricted actuation, then distraction)
    no cloud capability needed     -> LOCAL
    no connectivity                -> DEFER
    sensitive spans, strict mode   -> policy.strict_privacy_route (LOCAL by default)
    sensitive spans                -> CLOUD_MASKED
    otherwise                      -> CLOUD

Safety and driving state come before capability and privacy so a safety
command never waits on a cloud round-trip and a distracting task never runs
just because the network is good. The function returns the route and the
branch that fired, so the labeler can record why and the degradation study
can check that routes shift for the right reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.schema import (
    ActuationClass,
    Capability,
    Context,
    DrivingDemand,
    Intent,
    LocalModelTier,
    Route,
)


class Reason(StrEnum):
    """Which branch of the tree produced the route."""

    SAFETY_CRITICAL = "safety_critical"
    RESTRICTED_ACTUATION = "restricted_actuation"
    DISTRACTION = "distraction"
    LOCAL_CAPABLE = "local_capable"
    NO_CONNECTIVITY = "no_connectivity"
    STRICT_PRIVACY = "strict_privacy"
    SENSITIVE_SPANS = "sensitive_spans"
    CLOUD_DEFAULT = "cloud_default"


@dataclass(frozen=True, slots=True)
class Decision:
    route: Route
    reason: Reason
    demand: DrivingDemand
    """Demand actually used by the distraction gate (after passenger relaxation)."""


def driving_demand(context: Context, policy: Policy = DEFAULT_POLICY) -> DrivingDemand:
    """Table 1: the raw demand, before any passenger relaxation."""
    return policy.demand(context.speed_bucket, context.driver_workload)


def distraction_demand(context: Context, policy: Policy = DEFAULT_POLICY) -> DrivingDemand:
    """Demand as seen by the distraction gate: relaxed if a passenger is present."""
    demand = driving_demand(context, policy)
    if context.passenger_present and policy.passenger_relax_steps:
        demand = demand.relaxed(policy.passenger_relax_steps)
    return demand


def locally_capable(intent: Intent, context: Context) -> bool:
    """Can the head unit answer this intent well enough on its own?"""
    if intent.capability is Capability.LOCAL_OK:
        return True
    if intent.capability is Capability.NEEDS_SMALL_LOCAL:
        return context.local_model_tier is LocalModelTier.SMALL
    return False


def route(
    intent: Intent,
    context: Context,
    has_sensitive_spans: bool = False,
    policy: Policy = DEFAULT_POLICY,
) -> Decision:
    """Apply the precedence tree and return the route with its reason."""
    raw_demand = driving_demand(context, policy)
    demand = distraction_demand(context, policy)

    # 1. Safety-critical vehicle command: local, immediately, no matter what.
    if intent.actuation is ActuationClass.SAFETY_CRITICAL:
        return Decision(Route.LOCAL, Reason.SAFETY_CRITICAL, demand)

    # 2a. Restricted actuation uses the raw demand: a passenger never unlocks doors.
    if intent.actuation is not ActuationClass.NONE and not policy.actuation_allowed(
        intent.actuation, raw_demand
    ):
        return Decision(Route.REFUSE, Reason.RESTRICTED_ACTUATION, demand)

    # 2b. Distraction gate uses the passenger-relaxed demand.
    if not policy.distraction_allowed(intent.distraction, demand):
        return Decision(Route.REFUSE, Reason.DISTRACTION, demand)

    # 3. No cloud capability needed.
    if locally_capable(intent, context):
        return Decision(Route.LOCAL, Reason.LOCAL_CAPABLE, demand)

    # 4. Cloud needed but unreachable.
    if context.connectivity not in policy.cloud_connectivity:
        return Decision(Route.DEFER, Reason.NO_CONNECTIVITY, demand)

    # 5. Sensitive spans.
    if has_sensitive_spans:
        if context.privacy_mode.value == "strict":
            return Decision(policy.strict_privacy_route, Reason.STRICT_PRIVACY, demand)
        return Decision(Route.CLOUD_MASKED, Reason.SENSITIVE_SPANS, demand)

    # 6. Plain cloud.
    return Decision(Route.CLOUD, Reason.CLOUD_DEFAULT, demand)
