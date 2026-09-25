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
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autogate_eval.metrics import SPLIT_ORDER, format_report, report

CLOUD_ROUTES = frozenset({"CLOUD", "CLOUD_MASKED"})


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
