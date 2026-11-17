"""From answer probabilities and a context to a route, a risk score and an abstention.

Tables (built once, :class:`RouteTables`)
    Every context (288: speed 4 x connectivity 3 x workload 3 x privacy 2 x
    passenger 2 x local tier 2) and every combination of the answers the
    rulebook reads (96: capability 4 x distraction 3 x actuation 4 x
    sensitive 2) go through ``autogate_bench.rulebook.route_tags`` once.
    ``cost[ctx, combo, r]`` is the cost-matrix cost of predicting route ``r``
    when the truth is that combination in that context, and
    ``critical[ctx, combo, r]`` whether that prediction is a critical miss
    (the definition in ``autogate_eval.metrics``).

Combination probabilities (:func:`combo_probs`)
    ``product``: the product of the four marginals (capability, distraction,
    actuation, sensitive). Tags are fixed per intent, so the product puts
    mass on combinations no intent has when the heads hesitate between two
    intents. ``joint``: a head over the (capability, distraction, actuation)
    combinations seen in training, times p(sensitive). ``mix``: lambda *
    joint + (1 - lambda) * product, lambda chosen on held-out intents.

Decoders
    ``conservative``: ``autogate_router.tag_routing.decode_tags`` (actuation
    safety_critical / restricted at p >= 0.3, distraction high at 0.3,
    medium + high at 0.5) on the weighted heads, sensitive at p >= 0.5, then
    the rulebook. This is the pre-registered Gate 2 decoder.
    ``expected_cost``: the route with the lowest expected cost under the
    combination probabilities, restricted to routes the rulebook can produce
    in that context (so it never answers DEFER with good signal).

Risk score
    ``p_critical``: the probability, under the combination probabilities,
    that the chosen route is a critical miss. Abstaining ("ask the driver")
    on the riskiest rows trades coverage for critical misses;
    :func:`abstention_scan` reports that trade and :func:`pick_threshold`
    chooses the operating point on val.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from autogate_bench.rulebook import route_tags
from autogate_bench.schema import (
    Connectivity,
    Context,
    LocalModelTier,
    PrivacyMode,
    Route,
    SpeedBucket,
    Workload,
)
from autogate_decide.data import QUESTIONS
from autogate_eval.cost_matrix import CostMatrix, default_cost_matrix

ROUTES: tuple[str, ...] = tuple(str(r) for r in Route)
R_IDX = {r: i for i, r in enumerate(ROUTES)}
COMBO_QUESTIONS: tuple[str, ...] = ("capability", "distraction", "actuation", "sensitive")
COMBOS: tuple[tuple[str, ...], ...] = tuple(
    itertools.product(*(QUESTIONS[q] for q in COMBO_QUESTIONS))
)
COMBO_IDX = {c: i for i, c in enumerate(COMBOS)}
CONTEXTS: tuple[Context, ...] = tuple(
    Context(s, c, w, p, bool(pa), t)
    for s, c, w, p, pa, t in itertools.product(
        SpeedBucket, Connectivity, Workload, PrivacyMode, (False, True), LocalModelTier
    )
)
CTX_IDX = {c: i for i, c in enumerate(CONTEXTS)}


def context_index(row: Mapping[str, Any]) -> int:
    from autogate_router.tag_routing import row_context

    return CTX_IDX[row_context(row)]


@dataclass(frozen=True)
class RouteTables:
    route: np.ndarray  # [ctx, combo] route index
    cost: np.ndarray  # [ctx, combo, route]
    critical: np.ndarray  # [ctx, combo, route] bool
    feasible: np.ndarray  # [ctx, route] bool


@lru_cache(maxsize=4)
def route_tables(cost_matrix: CostMatrix | None = None) -> RouteTables:
    cm = cost_matrix or default_cost_matrix()
    n_ctx, n_combo, n_r = len(CONTEXTS), len(COMBOS), len(ROUTES)
    route = np.zeros((n_ctx, n_combo), dtype=np.int64)
    cost = np.zeros((n_ctx, n_combo, n_r))
    critical = np.zeros((n_ctx, n_combo, n_r), dtype=bool)
    for i, ctx in enumerate(CONTEXTS):
        for j, (cap, dis, act, sens) in enumerate(COMBOS):
            true = str(route_tags(cap, dis, act, ctx, sens == "yes").route)
            sc = act == "safety_critical"
            route[i, j] = R_IDX[true]
            for k, r in enumerate(ROUTES):
                cost[i, j, k] = cm.cost(true, r, sc)
                to_cloud = r in ("CLOUD", "CLOUD_MASKED")
                critical[i, j, k] = (true == "LOCAL" and sc and to_cloud) or (
                    true == "REFUSE" and (to_cloud or r == "LOCAL")
                )
    feasible = np.zeros((n_ctx, n_r), dtype=bool)
    for i in range(n_ctx):
        feasible[i, np.unique(route[i])] = True
    return RouteTables(route, cost, critical, feasible)


def combo_probs(
    probs: Mapping[str, np.ndarray],
    mode: str = "product",
    joint: np.ndarray | None = None,
    joint_vocab: Sequence[tuple[str, str, str]] = (),
    lam: float = 0.0,
) -> np.ndarray:
    """``[n, 96]`` probabilities over :data:`COMBOS` from per-question ``[n, k]`` probabilities."""
    pc, pd, pa, ps = (probs[q] for q in COMBO_QUESTIONS)
    n = len(pc)
    prod = np.einsum("ni,nj,nk,nl->nijkl", pc, pd, pa, ps).reshape(n, -1)
    if mode == "product":
        return prod
    if joint is None:
        raise ValueError(f"mode {mode!r} needs joint probabilities")
    j3 = np.zeros((n, *(len(QUESTIONS[q]) for q in ("capability", "distraction", "actuation"))))
    for c, (cap, dis, act) in enumerate(joint_vocab):
        j3[
            :,
            QUESTIONS["capability"].index(cap),
            QUESTIONS["distraction"].index(dis),
            QUESTIONS["actuation"].index(act),
        ] = joint[:, c]
    jp = np.einsum("nijk,nl->nijkl", j3, ps).reshape(n, -1)
    if mode == "joint":
        return jp
    if mode == "mix":
        return lam * jp + (1.0 - lam) * prod
    raise ValueError(f"unknown mode {mode!r}")


def expected_costs(pc: np.ndarray, ctx: np.ndarray, tables: RouteTables) -> np.ndarray:
    """``[n, 5]`` expected cost of each route; infeasible routes are +inf."""
    e = np.einsum("nc,ncr->nr", pc, tables.cost[ctx])
    return np.where(tables.feasible[ctx], e, np.inf)


def p_critical(pc: np.ndarray, ctx: np.ndarray, route: np.ndarray, t: RouteTables) -> np.ndarray:
    """Probability that ``route`` is a critical miss, per row."""
    crit = t.critical[ctx, :, route]  # [n, combo]
    return (pc * crit).sum(-1)


def decode_expected_cost(pc: np.ndarray, ctx: np.ndarray, t: RouteTables) -> np.ndarray:
    return expected_costs(pc, ctx, t).argmin(-1)


def conservative_combo(
    probs: Mapping[str, np.ndarray], sensitive_threshold: float = 0.5
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Combination index and decoded tags per row, by ``tag_routing.decode_tags``."""
    from autogate_router.tag_routing import decode_tags

    tag_names = ("capability", "distraction", "actuation", "consequence")
    n = len(probs["capability"])
    idx = np.zeros(n, dtype=np.int64)
    decoded = []
    for i in range(n):
        per = {q: dict(zip(QUESTIONS[q], probs[q][i].tolist(), strict=True)) for q in tag_names}
        tags = decode_tags(per)
        sens = "yes" if probs["sensitive"][i, 1] >= sensitive_threshold else "no"
        idx[i] = COMBO_IDX[(tags["capability"], tags["distraction"], tags["actuation"], sens)]
        decoded.append({**tags, "sensitive": sens})
    return idx, decoded


def decode_conservative(combo: np.ndarray, ctx: np.ndarray, t: RouteTables) -> np.ndarray:
    return t.route[ctx, combo]


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #


def wilson_upper(k: int, n: int, z: float = 1.96) -> float | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre + rad) / den)


def abstention_scan(
    rows: Sequence[Mapping[str, Any]],
    routes: Sequence[str],
    risk: np.ndarray,
    coverages: Sequence[float] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5),
) -> list[dict[str, Any]]:
    """Covered-row metrics when the ``1 - coverage`` riskiest rows are abstained on."""
    from autogate_eval.metrics import _L, _R, _critical, _encode, evaluate

    y, p, sc = _encode(rows, routes)
    eligible, miss = _critical(y, p, sc)
    order = np.sort(risk)
    out = []
    for c in coverages:
        k = int(round(c * len(rows)))
        thr = float(order[k - 1]) if k > 0 else -np.inf
        cov = risk <= thr
        m = evaluate(
            [r for r, k_ in zip(rows, cov, strict=True) if k_],
            [x for x, k_ in zip(routes, cov, strict=True) if k_],
            n_boot=20,
        )
        n_el = int((eligible & cov).sum())
        n_miss = int((miss & cov).sum())
        out.append(
            {
                "target_coverage": c,
                "threshold": thr,
                "coverage": float(cov.mean()),
                "macro_f1": m.get("macro_f1"),
                "swe": m.get("swe"),
                "critical_miss_rate": n_miss / n_el if n_el else None,
                "critical_miss_ucb95": wilson_upper(n_miss, n_el),
                "critical_misses": n_miss,
                "n_critical": n_el,
                "abstained_eligible_rate": float((eligible & ~cov).sum() / max(eligible.sum(), 1)),
                "abstained_safety_critical_rate": float(
                    ((y == _L) & sc & ~cov).sum() / max(((y == _L) & sc).sum(), 1)
                ),
                "abstained_refuse_rate": float(((y == _R) & ~cov).sum() / max((y == _R).sum(), 1)),
            }
        )
    return out


def apply_threshold(
    rows: Sequence[Mapping[str, Any]], routes: Sequence[str], risk: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Metrics on the rows with ``risk <= threshold``, plus what abstention cost."""
    from autogate_eval.metrics import _L, _critical, _encode, evaluate

    y, p, sc = _encode(rows, routes)
    eligible, miss = _critical(y, p, sc)
    cov = risk <= threshold
    m = evaluate(
        [r for r, k in zip(rows, cov, strict=True) if k],
        [x for x, k in zip(routes, cov, strict=True) if k],
        n_boot=200,
    )
    n_el, n_miss = int((eligible & cov).sum()), int((miss & cov).sum())
    sc_rows = (y == _L) & sc
    return {
        "threshold": threshold,
        "coverage": float(cov.mean()),
        "macro_f1": m.get("macro_f1"),
        "swe": m.get("swe"),
        "critical_miss_rate": n_miss / n_el if n_el else None,
        "critical_miss_ucb95": wilson_upper(n_miss, n_el),
        "critical_misses": n_miss,
        "n_critical": n_el,
        "critical_miss_rate_all_rows": float(miss.sum() / max(eligible.sum(), 1)),
        "abstained_eligible_rate": float((eligible & ~cov).sum() / max(eligible.sum(), 1)),
        "abstained_safety_critical_rate": float((sc_rows & ~cov).sum() / max(sc_rows.sum(), 1)),
    }


def pick_threshold(
    rows: Sequence[Mapping[str, Any]],
    routes: Sequence[str],
    risk: np.ndarray,
    target: float = 0.01,
) -> float:
    """The largest-coverage risk threshold whose covered critical miss rate is <= ``target``."""
    from autogate_eval.metrics import _critical, _encode

    y, p, sc = _encode(rows, routes)
    eligible, miss = _critical(y, p, sc)
    order = np.argsort(risk, kind="stable")
    el = np.cumsum(eligible[order])
    ms = np.cumsum(miss[order])
    rate = np.where(el > 0, ms / np.maximum(el, 1), 0.0)
    ok = np.flatnonzero(rate <= target)
    if len(ok) == 0:
        return -np.inf
    k = ok[-1]
    thr = float(risk[order[k]])
    # ties: rows with risk equal to thr are all covered; back off if that breaks the target
    while True:
        cov = risk <= thr
        n_el = (eligible & cov).sum()
        if n_el == 0 or (miss & cov).sum() / n_el <= target:
            return thr
        smaller = risk[risk < thr]
        if len(smaller) == 0:
            return -np.inf
        thr = float(smaller.max())
