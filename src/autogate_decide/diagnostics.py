"""Diagnostics reported with every A7 run (plan section 6, Phase 2).

* :func:`by_variant`: per language variant (plain, hinglish, kannada_english,
  disfluent, asr), utterance-level accuracy of each routing answer and the
  row-level critical miss rate, with counts.
* :func:`sensitive_scores`: recall and precision of the sensitive answer,
  per utterance.
* :func:`intent_bootstrap`: 95% intervals for SWE, macro-F1 and critical
  miss from resampling **intents** (all rows of a drawn intent together).
  Rows of one intent are strongly correlated, so the row-level intervals
  of ``autogate_eval.metrics`` are too narrow, most of all on OOD, which
  has eight intents.
* :func:`per_intent`: route metrics per intent (used for OOD).
* :func:`minimal_pair_probe`: over the minimal pairs of ``data/augment``
  (never trained on in A7), the share of pairs where every utterance on
  both sides gets capability, distraction and actuation right.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from autogate_eval.metrics import _critical, _encode, evaluate


def by_variant(
    rows: Sequence[Mapping[str, Any]],
    routes: Sequence[str],
    utt_of_row: np.ndarray,
    correct: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """``correct[q]`` is a per-utterance boolean array for question ``q``."""
    y, p, sc = _encode(rows, routes)
    eligible, miss = _critical(y, p, sc)
    variants = np.array([r.get("variant", "plain") for r in rows])
    out = {}
    for v in sorted(set(variants)):
        m = variants == v
        utts = np.unique(utt_of_row[m])
        n_el = int(eligible[m].sum())
        out[v] = {
            "n_rows": int(m.sum()),
            "n_utterances": len(utts),
            **{f"acc_{q}": float(c[utts].mean()) for q, c in correct.items()},
            "critical_miss_rate": float(miss[m].sum() / n_el) if n_el else None,
            "critical_misses": int(miss[m].sum()),
            "n_critical": n_el,
        }
    return out


def sensitive_scores(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tp = int((true & pred).sum())
    n_true, n_pred = int(true.sum()), int(pred.sum())
    return {
        "recall": tp / n_true if n_true else None,
        "precision": tp / n_pred if n_pred else None,
        "n_true": n_true,
        "n_pred": n_pred,
    }


def intent_bootstrap(
    rows: Sequence[Mapping[str, Any]], routes: Sequence[str], n_boot: int = 500, seed: int = 0
) -> dict[str, list[float] | None]:
    from autogate_eval.cost_matrix import default_cost_matrix
    from autogate_eval.metrics import _row_costs, confusion, macro_f1_from_confusion

    y, p, sc = _encode(rows, routes)
    costs = _row_costs(y, p, sc, default_cost_matrix())
    eligible, miss = _critical(y, p, sc)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["intent"]].append(i)
    members = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    swe, f1, crit = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[j] for j in pick])
        swe.append(costs[idx].mean())
        f1.append(macro_f1_from_confusion(confusion(y[idx], p[idx])))
        el = eligible[idx].sum()
        if el:
            crit.append(miss[idx].sum() / el)

    def ci(a: list[float]) -> list[float] | None:
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else None

    return {"swe": ci(swe), "macro_f1": ci(f1), "critical_miss_rate": ci(crit)}


def per_intent(rows: Sequence[Mapping[str, Any]], routes: Sequence[str]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["intent"]].append(i)
    out = {}
    for name, idx in sorted(groups.items()):
        m = evaluate([rows[i] for i in idx], [routes[i] for i in idx], n_boot=20)
        out[name] = {k: m[k] for k in ("n", "accuracy", "swe", "critical_miss_rate", "n_critical")}
    return out


def minimal_pair_probe(
    pair_rows: Sequence[Mapping[str, Any]],
    predicted: Mapping[str, Mapping[str, str]],
    differs_in: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """``predicted[utterance text]`` -> decoded tags. A pair passes if every one of its
    utterances, on both sides, has all three routing tags right."""
    from autogate_decide.data import ROUTING_TAGS

    ok: dict[str, bool] = {}
    by_axis: dict[str, list[bool]] = defaultdict(list)
    for r in pair_rows:
        pid = r["pair_id"]
        tags = predicted[r["utterance"]]
        right = all(tags[q] == r[q] for q in ROUTING_TAGS)
        ok[pid] = ok.get(pid, True) and right
    for pid, passed in ok.items():
        for ax in (differs_in or {}).get(pid) or ["unknown"]:
            by_axis[ax].append(passed)
    return {
        "n_pairs": len(ok),
        "both_sides_right": float(np.mean(list(ok.values()))) if ok else None,
        "by_differing_tag": {a: float(np.mean(v)) for a, v in sorted(by_axis.items())},
    }
