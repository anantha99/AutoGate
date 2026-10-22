"""Score any predictions file against benchmark rows.

    python -m autogate_eval.score --rows data/generated/full/rows.jsonl --preds predictions.jsonl

``predictions.jsonl`` has one JSON object per row::

    {"id": str, "route": str, "route_probs": [5 floats] | null,
     "intent": str | null, "intent_probs_top3": [[intent, p], ...] | null,
     "spans": [{"start", "end", "label", "text"}] | null, "confidence": float | null}

Only ``id`` and ``route`` are required. Predictions are joined to rows on
``id``; rows without a prediction are left out (so a file covering only
test and ood can be scored against the full rows file), and a prediction
whose id is not in the rows is an error.

Reported:

* the route report from ``autogate_eval.metrics.report`` (overall, per split,
  per slice, per class, confusion), rendered with ``format_report``;
* intent accuracy, overall and per split, when predictions carry ``intent``;
* when predictions carry ``spans``: exact-match span precision, recall and
  F1 (same start, end and label), and the **leak rate**: over rows whose
  *predicted* route is CLOUD or CLOUD_MASKED (the ones whose text leaves
  the car), the fraction of true sensitive spans not covered by a predicted
  span. A true span is covered when every non-space character in it lies
  inside some predicted span, whatever its label.
* when predictions carry ``tags`` (a router with tag heads), per tag
  (capability, distraction, actuation, consequence), overall and per split
  (so OOD is reported on its own):

  - ``accuracy`` and ``macro_f1`` (mean F1 over the classes that occur in
    the labels or the predictions);
  - ``dangerous_miss_rate``: over rows whose true class is dangerous
    (actuation safety_critical or restricted; distraction high;
    consequence irreversible; capability has none, so ``None``), the
    fraction predicted as a non-dangerous class;
  - ``route_changing_errors``: rows where this tag is wrong and correcting
    only this tag (the other tags as predicted, the row's context, and
    sensitivity = any predicted span) changes the rulebook's route.
    ``route_changing_cost`` is the summed cost-matrix cost of the route
    from the predicted tags on those rows, ``route_changing_cost_if_corrected``
    the same with the tag corrected. Consequence is not read by the
    rulebook, so it never changes the route.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autogate_eval.metrics import SPLIT_ORDER, format_report, report

CLOUD_ROUTES = frozenset({"CLOUD", "CLOUD_MASKED"})
TAG_NAMES = ("capability", "distraction", "actuation", "consequence")
DANGEROUS: dict[str, frozenset[str]] = {
    "capability": frozenset(),
    "distraction": frozenset({"high"}),
    "actuation": frozenset({"safety_critical", "restricted"}),
    "consequence": frozenset({"irreversible"}),
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(str(path)).to_pylist()
    return _read_jsonl(path)


def join(
    rows: Sequence[Mapping[str, Any]], preds: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Rows that have a prediction, and those predictions, in row order."""
    by_id: dict[str, Mapping[str, Any]] = {}
    for p in preds:
        if p["id"] in by_id:
            raise ValueError(f"duplicate prediction for id {p['id']!r}")
        by_id[p["id"]] = p
    row_ids = {r["id"] for r in rows}
    unknown = [i for i in by_id if i not in row_ids]
    if unknown:
        raise ValueError(f"{len(unknown)} predictions have ids not in the rows, e.g. {unknown[:3]}")
    kept = [r for r in rows if r["id"] in by_id]
    return kept, [by_id[r["id"]] for r in kept]


def _covered(span: Mapping[str, Any], text: str, predicted: Sequence[Mapping[str, Any]]) -> bool:
    for i in range(span["start"], span["end"]):
        if text[i].isspace():
            continue
        if not any(p["start"] <= i < p["end"] for p in predicted):
            return False
    return True


def span_scores(rows: Sequence[Mapping[str, Any]], preds: Sequence[Mapping[str, Any]]) -> dict:
    tp = n_pred = n_true = 0
    leaked = at_risk = 0
    for r, p in zip(rows, preds, strict=True):
        true = [(s["start"], s["end"], s["label"]) for s in r["spans"]]
        pred_spans = p.get("spans") or []
        pred = {(s["start"], s["end"], s["label"]) for s in pred_spans}
        tp += len(set(true) & pred)
        n_pred += len(pred)
        n_true += len(true)
        if p["route"] in CLOUD_ROUTES:
            for s in r["spans"]:
                at_risk += 1
                leaked += not _covered(s, r["utterance"], pred_spans)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_true": n_true,
        "n_pred": n_pred,
        "leak_rate": leaked / at_risk if at_risk else None,
        "leaked_spans": leaked,
        "spans_sent_to_cloud": at_risk,
    }


def intent_scores(rows: Sequence[Mapping[str, Any]], preds: Sequence[Mapping[str, Any]]) -> dict:
    def acc(pairs: list[tuple[Mapping, Mapping]]) -> float | None:
        return sum(p["intent"] == r["intent"] for r, p in pairs) / len(pairs) if pairs else None

    pairs = list(zip(rows, preds, strict=True))
    present = {r["split"] for r in rows}
    splits = [s for s in SPLIT_ORDER if s in present] + sorted(present - set(SPLIT_ORDER))
    return {
        "accuracy": acc(pairs),
        "by_split": {s: acc([(r, p) for r, p in pairs if r["split"] == s]) for s in splits},
    }


def _macro_f1(true: Sequence[str], pred: Sequence[str]) -> float | None:
    classes = sorted(set(true) | set(pred))
    if not classes:
        return None
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(true, pred, strict=True))
        n_pred = sum(p == c for p in pred)
        n_true = sum(t == c for t in true)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def _tag_routes(row: Mapping[str, Any], tags: Mapping[str, str], sensitive: bool) -> str:
    from autogate_bench.rulebook import route_tags
    from autogate_eval.baselines.rules import row_context

    return str(
        route_tags(
            tags["capability"], tags["distraction"], tags["actuation"], row_context(row), sensitive
        ).route
    )


def _one_tag(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    routes: Sequence[str],
    cm: Any,
) -> dict[str, Any]:
    true = [str(r[name]) for r in rows]
    pred = [str(p["tags"][name]) for p in preds]
    n = len(rows)
    danger = DANGEROUS[name]
    at_risk = [(t, q) for t, q in zip(true, pred, strict=True) if t in danger]
    missed = sum(q not in danger for _, q in at_risk)
    changing = 0
    cost = cost_fixed = 0.0
    for r, p, route_pred, t, q in zip(rows, preds, routes, true, pred, strict=True):
        if t == q:
            continue
        fixed = _tag_routes(r, {**p["tags"], name: t}, bool(p.get("spans")))
        if fixed != route_pred:
            sc = r["actuation"] == "safety_critical"
            changing += 1
            cost += cm.cost(r["route"], route_pred, sc)
            cost_fixed += cm.cost(r["route"], fixed, sc)
    return {
        "n": n,
        "accuracy": sum(t == q for t, q in zip(true, pred, strict=True)) / n if n else None,
        "macro_f1": _macro_f1(true, pred),
        "errors": sum(t != q for t, q in zip(true, pred, strict=True)),
        "n_dangerous": len(at_risk),
        "dangerous_misses": missed,
        "dangerous_miss_rate": missed / len(at_risk) if at_risk else None,
        "route_changing_errors": changing,
        "route_changing_cost": cost,
        "route_changing_cost_if_corrected": cost_fixed,
    }


def tag_scores(
    rows: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    cost_matrix: Any = None,
) -> dict[str, Any]:
    """The per-tag report in the module docstring: overall and per split."""
    from autogate_eval.cost_matrix import default_cost_matrix

    cm = cost_matrix or default_cost_matrix()
    names = [t for t in TAG_NAMES if all(t in p["tags"] for p in preds)]
    routes = [
        _tag_routes(r, p["tags"], bool(p.get("spans"))) for r, p in zip(rows, preds, strict=True)
    ]

    def block(idx: Sequence[int]) -> dict[str, Any]:
        rs, ps, rt = [rows[i] for i in idx], [preds[i] for i in idx], [routes[i] for i in idx]
        return {name: _one_tag(name, rs, ps, rt, cm) for name in names}

    present = {r["split"] for r in rows}
    splits = [s for s in SPLIT_ORDER if s in present] + sorted(present - set(SPLIT_ORDER))
    return {
        "overall": block(range(len(rows))),
        "by_split": {s: block([i for i, r in enumerate(rows) if r["split"] == s]) for s in splits},
    }


def format_tag_scores(tags: Mapping[str, Any]) -> str:
    lines = [
        "| subset | tag | n | accuracy | macro-F1 | dangerous miss | route-changing errors "
        "| their cost (if corrected) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    blocks = [("overall", tags["overall"])] + [
        (f"split: {s}", b) for s, b in tags.get("by_split", {}).items()
    ]
    for label, block in blocks:
        for name, m in block.items():
            dm = (
                f"{_f(m['dangerous_miss_rate'])} ({m['dangerous_misses']}/{m['n_dangerous']})"
                if m["n_dangerous"]
                else "-"
            )
            lines.append(
                f"| {label} | {name} | {m['n']} | {_f(m['accuracy'])} | {_f(m['macro_f1'])} "
                f"| {dm} | {m['route_changing_errors']} | {m['route_changing_cost']:.1f} "
                f"({m['route_changing_cost_if_corrected']:.1f}) |"
            )
    return "\n".join(lines)


def score(
    rows: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    n_boot: int = 1000,
    cost_matrix: Any = None,
) -> dict[str, Any]:
    """Everything in the module docstring, as a dict."""
    n_rows = len(rows)
    rows, preds = join(rows, preds)
    if not rows:
        raise ValueError("no prediction matches a row id")
    out: dict[str, Any] = {
        "n_rows": n_rows,
        "n_scored": len(rows),
        "route": report(rows, [p["route"] for p in preds], n_boot=n_boot, cost_matrix=cost_matrix),
    }
    if all(p.get("intent") is not None for p in preds):
        out["intent"] = intent_scores(rows, preds)
    if all(p.get("spans") is not None for p in preds):
        out["spans"] = span_scores(rows, preds)
    if all(p.get("tags") for p in preds):
        out["tags"] = tag_scores(rows, preds, cost_matrix)
    return out


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def format_score(result: Mapping[str, Any]) -> str:
    lines = [f"scored {result['n_scored']} of {result['n_rows']} rows", ""]
    intent = result.get("intent")
    if intent:
        by = ", ".join(f"{s} {_f(v)}" for s, v in intent["by_split"].items())
        lines.append(f"intent accuracy: {_f(intent['accuracy'])} ({by})")
    spans = result.get("spans")
    if spans:
        lines.append(
            f"spans (exact match): precision {_f(spans['precision'])}, recall "
            f"{_f(spans['recall'])}, F1 {_f(spans['f1'])} ({spans['n_true']} true, "
            f"{spans['n_pred']} predicted)"
        )
        lines.append(
            f"leak rate: {_f(spans['leak_rate'])} ({spans['leaked_spans']} of "
            f"{spans['spans_sent_to_cloud']} true spans on rows routed CLOUD/CLOUD_MASKED "
            "not covered by a predicted span)"
        )
    if intent or spans:
        lines.append("")
    lines.append(format_report(result["route"]))
    if result.get("tags"):
        lines += ["", "tags (route-changing: fixing only that tag changes the tag route):", ""]
        lines.append(format_tag_scores(result["tags"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--rows", type=Path, required=True, help="rows.jsonl or rows.parquet")
    p.add_argument("--preds", type=Path, required=True, help="predictions.jsonl")
    p.add_argument("--split", default=None, help="comma-separated splits to score (default: all)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--json", type=Path, default=None, help="also write the full result here")
    a = p.parse_args(argv)

    rows = load_rows(a.rows)
    if a.split:
        keep = {s.strip() for s in a.split.split(",")}
        rows = [r for r in rows if r["split"] in keep]
    preds = _read_jsonl(a.preds)
    if a.split:
        ids = {r["id"] for r in rows}
        preds = [x for x in preds if x["id"] in ids]
    result = score(rows, preds, n_boot=a.n_boot)
    print(format_score(result))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
