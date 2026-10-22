"""Routes from predicted policy tags, and the combined (more cautious) decision.

The router's tag heads predict the four intent tags the rulebook reads
(capability, distraction, actuation; consequence is reported but no branch
of the tree uses it). One forward pass therefore gives two routes:

* **e2e**: the route head's argmax.
* **tags**: :func:`route_from_tags`, the rulebook's precedence tree
  (``autogate_bench.rulebook.route_tags``) applied to the decoded tags, the
  row's context, and whether the span head found any sensitive span.

and a third, **combined**, from :func:`combine`.

Conservative decoding (:func:`decode_tags`)
    A tag miss is not symmetric: calling a safety-critical command
    ``comfort`` can send it to the cloud, calling a restricted one ``none``
    lets it run while moving, and calling a high-distraction task ``low``
    lets it through at speed. So by default the dangerous classes win at
    probabilities well below 0.5:

    * actuation: ``safety_critical`` if p >= tau_sc (0.3), else
      ``restricted`` if p >= tau_restricted (0.3), else argmax;
    * distraction: ``high`` if p >= tau_high (0.3), else ``medium`` if
      p(medium) + p(high) >= tau_medium_high (0.5), else argmax;
    * capability and consequence: argmax.

    ``conservative=False`` is plain argmax for all four.

Combining e2e and tags (:func:`combine`)
    1. tag actuation is ``safety_critical``  -> the tag route (LOCAL);
    2. either route is REFUSE                -> REFUSE;
    3. either route is DEFER                 -> DEFER;
    4. one is CLOUD_MASKED, the other CLOUD  -> CLOUD_MASKED;
    5. otherwise                             -> the tag route.

    Rule 1 keeps a safety command off the network whatever the route head
    says; rules 2 to 4 take the more cautious of the two; rule 5 trusts the
    tree when the two disagree in a way none of the above covers (LOCAL
    against CLOUD, say), since the tree is exact once its inputs are right.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.rulebook import Decision, route_tags
from autogate_bench.schema import (
    ActuationClass,
    Capability,
    Connectivity,
    Context,
    DistractionLevel,
    LocalModelTier,
    PrivacyMode,
    Route,
    SpeedBucket,
    Workload,
)

ROUTE_SOURCES = ("e2e", "tags", "combined")


@dataclass(frozen=True)
class TagThresholds:
    """Thresholds for conservative tag decoding (see the module docstring)."""

    safety_critical: float = 0.3
    restricted: float = 0.3
    high: float = 0.3
    medium_high: float = 0.5


DEFAULT_THRESHOLDS = TagThresholds()


def _argmax(p: Mapping[str, float]) -> str:
    return max(p, key=p.__getitem__)


def decode_tags(
    probs: Mapping[str, Mapping[str, float]],
    conservative: bool = True,
    thresholds: TagThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, str]:
    """One value per tag from per-tag probabilities ``{tag: {value: p}}``."""
    out = {name: _argmax(p) for name, p in probs.items()}
    if not conservative:
        return out
    act = probs.get("actuation")
    if act is not None:
        if act.get(ActuationClass.SAFETY_CRITICAL, 0.0) >= thresholds.safety_critical:
            out["actuation"] = str(ActuationClass.SAFETY_CRITICAL)
        elif act.get(ActuationClass.RESTRICTED, 0.0) >= thresholds.restricted:
            out["actuation"] = str(ActuationClass.RESTRICTED)
    dis = probs.get("distraction")
    if dis is not None:
        high = dis.get(DistractionLevel.HIGH, 0.0)
        medium = dis.get(DistractionLevel.MEDIUM, 0.0)
        if high >= thresholds.high:
            out["distraction"] = str(DistractionLevel.HIGH)
        elif medium + high >= thresholds.medium_high:
            out["distraction"] = str(DistractionLevel.MEDIUM)
    return out


def route_from_tags(
    capability: Capability | str,
    distraction: DistractionLevel | str,
    actuation: ActuationClass | str,
    context: Context,
    has_sensitive_spans: bool = False,
    policy: Policy = DEFAULT_POLICY,
) -> Decision:
    """The rulebook's decision for these tags; equals ``rulebook.route`` on an intent's tags."""
    return route_tags(capability, distraction, actuation, context, has_sensitive_spans, policy)


def combine(
    e2e_route: Route | str, tag_route: Route | str, tag_actuation: ActuationClass | str
) -> Route:
    """The more cautious of the two routes (rules in the module docstring)."""
    e2e, tag = Route(e2e_route), Route(tag_route)
    if ActuationClass(tag_actuation) is ActuationClass.SAFETY_CRITICAL:
        return tag
    pair = {e2e, tag}
    if Route.REFUSE in pair:
        return Route.REFUSE
    if Route.DEFER in pair:
        return Route.DEFER
    if pair == {Route.CLOUD_MASKED, Route.CLOUD}:
        return Route.CLOUD_MASKED
    return tag


def row_context(row: Mapping[str, Any]) -> Context:
    """The context of a benchmark row (its six context fields)."""
    return Context(
        SpeedBucket(row["speed_bucket"]),
        Connectivity(row["connectivity"]),
        Workload(row["driver_workload"]),
        PrivacyMode(row["privacy_mode"]),
        bool(row["passenger_present"]),
        LocalModelTier(row["local_model_tier"]),
    )
