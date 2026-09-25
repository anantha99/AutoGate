"""Context sampler: the 288-context grid, route maps, and flip-covering samples."""

import random

import pytest

from autogate_bench import Route, route
from autogate_bench.contexts import ALL_CONTEXTS, flip_profile, route_map, sample_contexts
from autogate_bench.intents import INTENTS
from autogate_bench.intents import INTENTS_BY_NAME as I


def test_all_contexts_are_288_unique():
    assert len(ALL_CONTEXTS) == 288
    assert len(set(ALL_CONTEXTS)) == 288


def test_route_map_matches_rulebook():
    rmap = route_map("find_nearby_place", True)
    assert len(rmap) == 288
    for c, d in rmap.items():
        assert d == route(I["find_nearby_place"], c, True)


def test_flip_profiles():
    assert flip_profile("defrost_windshield", False) == {Route.LOCAL}
    assert flip_profile("unlock_doors", False) == {Route.LOCAL, Route.REFUSE}
    assert flip_profile("traffic_on_route", False) == {Route.CLOUD, Route.DEFER}
    assert flip_profile("navigate_to_contact_address", True) == {
        Route.LOCAL,
        Route.CLOUD_MASKED,
    }


FLIPPABLE = [(i, s) for i in INTENTS for s in (False, True) if len(flip_profile(i, s)) > 1]
INVARIANT = [(i, s) for i in INTENTS for s in (False, True) if len(flip_profile(i, s)) == 1]


@pytest.mark.parametrize(
    ("intent", "sensitive"), FLIPPABLE, ids=lambda x: str(getattr(x, "name", x))
)
def test_flippable_samples_cover_two_routes(intent, sensitive):
    rmap = route_map(intent, sensitive)
    for seed in range(20):
        ctxs = sample_contexts(intent, sensitive, random.Random(seed), k=3)
        assert len(ctxs) == 3 and len(set(ctxs)) == 3
        assert len({rmap[c].route for c in ctxs}) >= 2


@pytest.mark.parametrize(
    ("intent", "sensitive"), INVARIANT, ids=lambda x: str(getattr(x, "name", x))
)
def test_invariant_samples_work(intent, sensitive):
    ctxs = sample_contexts(intent, sensitive, random.Random(0), k=3)
    assert len(set(ctxs)) == 3
    assert all(c in ALL_CONTEXTS for c in ctxs)


def test_sampler_prefers_rare_routes():
    # unlock_doors: LOCAL on 72 parked contexts, REFUSE on the 216 moving ones.
    rmap = route_map("unlock_doors", False)
    local = sum(
        rmap[c].route is Route.LOCAL
        for seed in range(200)
        for c in sample_contexts("unlock_doors", False, random.Random(seed), k=3)
    )
    assert local / 600 > 72 / 288


def test_sampler_is_deterministic():
    a = sample_contexts("read_messages", True, random.Random(5), k=4)
    b = sample_contexts("read_messages", True, random.Random(5), k=4)
    assert a == b


def test_k_out_of_range():
    with pytest.raises(ValueError):
        sample_contexts("read_messages", False, random.Random(0), k=0)
