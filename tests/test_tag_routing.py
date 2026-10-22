"""rulebook.route_tags, tag decoding, routing from tags, and the combine rule (no torch)."""

import hashlib
import re
from pathlib import Path

import pytest

from autogate_bench.contexts import ALL_CONTEXTS
from autogate_bench.intents import INTENTS
from autogate_bench.rulebook import Reason, route, route_tags
from autogate_bench.schema import (
    ActuationClass,
    Capability,
    Connectivity,
    Context,
    DistractionLevel,
    PrivacyMode,
    Route,
    SpeedBucket,
    Workload,
)
from autogate_router.tag_routing import (
    TagThresholds,
    combine,
    decode_tags,
    route_from_tags,
    row_context,
)

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_route_tags_equals_route_everywhere():
    n = 0
    for intent in INTENTS:
        for ctx in ALL_CONTEXTS:
            for sensitive in (False, True):
                want = route(intent, ctx, sensitive)
                got = route_tags(
                    intent.capability, intent.distraction, intent.actuation, ctx, sensitive
                )
                assert got == want, (intent.name, ctx, sensitive)
                assert (
                    route_from_tags(
                        str(intent.capability),
                        str(intent.distraction),
                        str(intent.actuation),
                        ctx,
                        sensitive,
                    )
                    == want
                )
                n += 1
    assert n == len(INTENTS) * 288 * 2


def test_pilot_labels_unchanged(tmp_path):
    """Regenerating the pilot after the refactor gives the sha256 recorded in docs/pilot.md."""
    from autogate_bench.generate import main as generate_main

    want = re.search(r"sha256\s+`([0-9a-f]{64})`", (DOCS / "pilot.md").read_text()).group(1)
    generate_main(["--out", str(tmp_path / "pilot"), "--rng-seed", "0"])
    got = hashlib.sha256((tmp_path / "pilot" / "rows.jsonl").read_bytes()).hexdigest()
    assert got == want


def test_route_tags_accepts_combinations_no_intent_allows():
    ctx = Context(SpeedBucket.HIGH, Connectivity.GOOD, Workload.LOW)
    # actuation on a cloud-capability "intent": Intent() would reject this
    d = route_tags(Capability.NEEDS_CLOUD, DistractionLevel.LOW, ActuationClass.COMFORT, ctx)
    assert d.route is Route.CLOUD and d.reason is Reason.CLOUD_DEFAULT
    d = route_tags("needs_cloud", "low", "restricted", ctx)
    assert d.route is Route.REFUSE and d.reason is Reason.RESTRICTED_ACTUATION
    d = route_tags("needs_cloud", "high", "safety_critical", ctx, True)
    assert d.route is Route.LOCAL and d.reason is Reason.SAFETY_CRITICAL
    with pytest.raises(ValueError):
        route_tags("teleport", "low", "none", ctx)


def test_row_context_matches_prefix(pilot_rows):
    for r in pilot_rows[:50]:
        assert row_context(r).to_prefix() == r["context_prefix"]


# --------------------------------------------------------------------------- #
# Conservative decoding
# --------------------------------------------------------------------------- #

CAP = {"local_ok": 0.7, "needs_small_local": 0.1, "needs_cloud": 0.1, "cloud_preferred": 0.1}
CONS = {"none": 0.2, "reversible": 0.3, "irreversible": 0.5}


def _probs(actuation, distraction):
    return {
        "capability": CAP,
        "distraction": distraction,
        "actuation": actuation,
        "consequence": CONS,
    }


def _act(none, sc, comfort, restricted):
    return {"none": none, "safety_critical": sc, "comfort": comfort, "restricted": restricted}


@pytest.mark.parametrize(
    ("actuation", "conservative", "plain"),
    [
        (_act(0.6, 0.3, 0.1, 0.0), "safety_critical", "none"),
        (_act(0.5, 0.29, 0.0, 0.21), "none", "none"),
        (_act(0.4, 0.0, 0.25, 0.35), "restricted", "none"),
        # both dangerous classes over threshold: safety_critical wins
        (_act(0.0, 0.3, 0.0, 0.7), "safety_critical", "restricted"),
        (_act(0.1, 0.1, 0.8, 0.0), "comfort", "comfort"),
    ],
)
def test_decode_actuation(actuation, conservative, plain):
    low = {"low": 0.9, "medium": 0.05, "high": 0.05}
    assert decode_tags(_probs(actuation, low))["actuation"] == conservative
    assert decode_tags(_probs(actuation, low), conservative=False)["actuation"] == plain


@pytest.mark.parametrize(
    ("distraction", "conservative", "plain"),
    [
        ({"low": 0.7, "medium": 0.0, "high": 0.3}, "high", "low"),
        ({"low": 0.5, "medium": 0.25, "high": 0.25}, "medium", "low"),
        ({"low": 0.51, "medium": 0.3, "high": 0.19}, "low", "low"),
        ({"low": 0.2, "medium": 0.75, "high": 0.05}, "medium", "medium"),
        ({"low": 0.1, "medium": 0.2, "high": 0.7}, "high", "high"),
    ],
)
def test_decode_distraction(distraction, conservative, plain):
    act = {"none": 1.0, "safety_critical": 0.0, "comfort": 0.0, "restricted": 0.0}
    assert decode_tags(_probs(act, distraction))["distraction"] == conservative
    assert decode_tags(_probs(act, distraction), conservative=False)["distraction"] == plain


def test_decode_argmax_tags_and_custom_thresholds():
    act = {"none": 0.75, "safety_critical": 0.2, "comfort": 0.05, "restricted": 0.0}
    dis = {"low": 0.8, "medium": 0.1, "high": 0.1}
    got = decode_tags(_probs(act, dis))
    assert got == {
        "capability": "local_ok",
        "distraction": "low",
        "actuation": "none",
        "consequence": "irreversible",
    }
    strict = TagThresholds(safety_critical=0.15, high=0.1)
    got = decode_tags(_probs(act, dis), thresholds=strict)
    assert got["actuation"] == "safety_critical" and got["distraction"] == "high"


# --------------------------------------------------------------------------- #
# combine
# --------------------------------------------------------------------------- #

L, C, CM, D, R = (Route.LOCAL, Route.CLOUD, Route.CLOUD_MASKED, Route.DEFER, Route.REFUSE)


@pytest.mark.parametrize(
    ("e2e", "tag", "actuation", "want"),
    [
        (C, L, "safety_critical", L),  # 1. safety-critical tags: the tag route, LOCAL
        (R, L, "safety_critical", L),
        (L, R, "none", R),  # 2. any REFUSE
        (R, C, "comfort", R),
        (R, D, "none", R),
        (D, CM, "none", D),  # 3. any DEFER
        (L, D, "none", D),
        (CM, C, "none", CM),  # 4. CLOUD_MASKED over CLOUD
        (C, CM, "none", CM),
        (L, C, "none", C),  # 5. otherwise the tag route
        (C, L, "none", L),
        (CM, L, "restricted", L),
        (L, L, "none", L),
        (C, C, "none", C),
    ],
)
def test_combine_rule_table(e2e, tag, actuation, want):
    assert combine(e2e, tag, actuation) is want
    assert combine(str(e2e), str(tag), actuation) is want


def test_combine_never_less_cautious_than_both_on_refuse_and_defer():
    for e2e in Route:
        for tag in Route:
            for act in ActuationClass:
                got = combine(e2e, tag, act)
                if act is ActuationClass.SAFETY_CRITICAL:
                    assert got is tag
                elif R in (e2e, tag):
                    assert got is R
                else:
                    assert got in (e2e, tag)


def test_strict_privacy_from_tags():
    ctx = Context(SpeedBucket.LOW, Connectivity.GOOD, Workload.LOW, PrivacyMode.STRICT)
    assert route_from_tags("needs_cloud", "low", "none", ctx, True).route is Route.LOCAL
    assert route_from_tags("needs_cloud", "low", "none", ctx, False).route is Route.CLOUD
