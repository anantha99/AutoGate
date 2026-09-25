"""Context sampler: pair each utterance with contexts that exercise its label.

The six context fields take 4 x 3 x 3 x 2 x 2 x 2 = 288 values. For one
utterance (an intent plus whether it has sensitive spans) the rulebook maps
each of them to a route. If all 288 give the same route the utterance is
context-invariant and any context will do; otherwise the sampler makes sure
the chosen contexts land on at least two different routes, leaning toward
the rarer ones, so a router cannot score well by ignoring the context.
"""

from __future__ import annotations

import itertools
import random

from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.rulebook import Decision, route
from autogate_bench.schema import (
    Connectivity,
    Context,
    Intent,
    LocalModelTier,
    PrivacyMode,
    Route,
    SpeedBucket,
    Workload,
)

ALL_CONTEXTS: tuple[Context, ...] = tuple(
    Context(speed, conn, workload, privacy, passenger, tier)
    for speed, conn, workload, privacy, passenger, tier in itertools.product(
        SpeedBucket, Connectivity, Workload, PrivacyMode, (False, True), LocalModelTier
    )
)
"""All 288 contexts, in enum order (speed slowest-varying, local tier fastest)."""

_CACHE: dict[tuple[str, bool], dict[Context, Decision]] = {}


def _intent(intent: Intent | str) -> Intent:
    return INTENTS_BY_NAME[intent] if isinstance(intent, str) else intent


def route_map(
    intent: Intent | str, has_sensitive_spans: bool, policy: Policy = DEFAULT_POLICY
) -> dict[Context, Decision]:
    """The rulebook's decision for this utterance under every context."""
    intent = _intent(intent)
    key = (intent.name, has_sensitive_spans)
    if policy is DEFAULT_POLICY and key in _CACHE:
        return _CACHE[key]
    result = {c: route(intent, c, has_sensitive_spans, policy) for c in ALL_CONTEXTS}
    if policy is DEFAULT_POLICY:
        _CACHE[key] = result
    return result


def flip_profile(
    intent: Intent | str, has_sensitive_spans: bool, policy: Policy = DEFAULT_POLICY
) -> frozenset[Route]:
    """The distinct routes this utterance takes over all 288 contexts."""
    return frozenset(d.route for d in route_map(intent, has_sensitive_spans, policy).values())


def _weighted_order(routes: list[Route], counts: dict[Route, int], rng: random.Random) -> list:
    """A random order of ``routes`` drawn without replacement, weight 1/count."""
    remaining = list(routes)
    order: list[Route] = []
    while remaining:
        weights = [1.0 / counts[r] for r in remaining]
        pick = rng.choices(remaining, weights=weights, k=1)[0]
        order.append(pick)
        remaining.remove(pick)
    return order


def sample_contexts(
    intent: Intent | str,
    has_sensitive_spans: bool,
    rng: random.Random,
    k: int = 3,
    policy: Policy = DEFAULT_POLICY,
) -> list[Context]:
    """Choose ``k`` distinct contexts for one utterance.

    Context-invariant utterance: ``k`` contexts uniformly at random. Otherwise
    the distinct routes are put in a random order that favours rare routes
    (weight 1 / number of contexts giving that route), the ``k`` slots are
    dealt round-robin over that order, and each slot takes a uniform context
    of its route not already chosen. With ``k >= 2`` at least two routes are
    covered. Deterministic given the rng state.
    """
    if not 1 <= k <= len(ALL_CONTEXTS):
        raise ValueError(f"k must be in 1..{len(ALL_CONTEXTS)}")
    rmap = route_map(intent, has_sensitive_spans, policy)
    by_route: dict[Route, list[Context]] = {}
    for c in ALL_CONTEXTS:  # fixed order, so the draw is reproducible
        by_route.setdefault(rmap[c].route, []).append(c)
    if len(by_route) == 1:
        return rng.sample(ALL_CONTEXTS, k)

    routes = [r for r in Route if r in by_route]
    counts = {r: len(by_route[r]) for r in routes}
    order = _weighted_order(routes, counts, rng)
    pools = {r: list(by_route[r]) for r in routes}
    chosen: list[Context] = []
    i = 0
    while len(chosen) < k:
        r = order[i % len(order)]
        i += 1
        if pools[r]:
            chosen.append(pools[r].pop(rng.randrange(len(pools[r]))))
    return chosen
