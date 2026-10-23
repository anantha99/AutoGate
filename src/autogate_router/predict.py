r"""Run a router checkpoint over a rows file and write ``predictions.jsonl``.

    python -m autogate_router.predict --checkpoint checkpoints/qwen3-0.6b/best \
        --rows data/generated/full/rows.jsonl --split test,ood --out predictions.jsonl

One JSON object per row, in file order::

    {"id": ..., "route": "LOCAL", "route_probs": [5 floats, labels.json order],
     "intent": "...", "intent_probs_top3": [["intent", p], ...],
     "spans": [{"start", "end", "label", "text"}], "confidence": 0.97}

``route_probs`` is the plain softmax of the route logits. ``confidence`` is
the max route probability after temperature scaling when
``calibration.json`` (from ``autogate_router.calibrate``) sits beside the
checkpoint, and the plain max otherwise. ``spans`` are character offsets
into the utterance, rebuilt from the span head's subword BIO tags through
the tokenizer's offset mapping (continuation subwords merged). Score the
file with ``python -m autogate_eval.score``.

With a checkpoint that has tag heads, each line also carries::

    "tags": {"capability": ..., "distraction": ..., "actuation": ..., "consequence": ...},
    "tag_probs": {"capability": {"local_ok": 0.9, ...}, ...},
    "route_e2e": "...", "route_tags": "...", "route_combined": "..."

``tags`` are decoded conservatively by default (``--no-conservative-tags``
for plain argmax; thresholds ``--tau-*``; see
``autogate_router.tag_routing``). ``route_e2e`` is the route head's argmax,
``route_tags`` the rulebook on the decoded tags, the row's context and
whether any span was predicted, and ``route_combined`` the more cautious of
the two. ``--route-source {e2e,tags,combined}`` (default ``e2e``) picks which
of them is copied into ``route``, so the scorer needs no change;
``--write-all PREFIX`` writes ``PREFIX_e2e.jsonl``, ``PREFIX_tags.jsonl``
and ``PREFIX_combined.jsonl`` from one forward pass.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autogate_router.tag_routing import ROUTE_SOURCES

CALIBRATION_FILE = "calibration.json"
PREDICTION_FIELDS = (
    "id",
    "route",
    "route_probs",
    "intent",
    "intent_probs_top3",
    "spans",
    "confidence",
)
TAG_PREDICTION_FIELDS = ("tags", "tag_probs", "route_e2e", "route_tags", "route_combined")


def infer(model: Any, examples: Sequence[dict], batch_size: int, device: Any) -> dict[str, Any]:
    """Logits for every example: route [N, 5], intent [N, n_intents], per-example span tag ids,
    and ``tag_logits`` {tag: [N, n_classes]} when the model has tag heads."""
    import torch

    from autogate_router.data import collate

    pad_id = getattr(getattr(model.backbone, "config", None), "pad_token_id", None) or 0
    route, intent, span_tags = [], [], []
    tag_names = tuple(getattr(model, "tag_names", ()))
    tag_out: dict[str, list[np.ndarray]] = {name: [] for name in tag_names}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i : i + batch_size]
            batch = collate(chunk, pad_id)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                utterance_mask=batch["utterance_mask"].to(device),
            )
            route.append(out["route_logits"].float().cpu().numpy())
            intent.append(out["intent_logits"].float().cpu().numpy())
            for name in tag_names:
                tag_out[name].append(out[f"{name}_logits"].float().cpu().numpy())
            tags = out["span_logits"].argmax(-1).cpu().numpy()
            span_tags += [tags[j, : len(ex["input_ids"])].tolist() for j, ex in enumerate(chunk)]
    model.train(was_training)
    n_r, n_i = model.config.n_routes, model.config.n_intents
    result: dict[str, Any] = {
        "route_logits": np.concatenate(route) if route else np.zeros((0, n_r), np.float32),
        "intent_logits": np.concatenate(intent) if intent else np.zeros((0, n_i), np.float32),
        "span_tags": span_tags,
    }
    if tag_names:
        result["tag_logits"] = {
            name: np.concatenate(v)
            if v
            else np.zeros((0, len(model.labels.tags[name])), np.float32)
            for name, v in tag_out.items()
        }
    return result


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def load_temperature(checkpoint: str | Path) -> float | None:
    path = Path(checkpoint) / CALIBRATION_FILE
    if not path.is_file():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["temperature"])


def predictions_from_outputs(
    rows: Sequence[dict],
    examples: Sequence[dict],
    outputs: dict[str, Any],
    labels: Any,
    temperature: float | None = None,
    route_source: str = "e2e",
    conservative: bool = True,
    thresholds: Any = None,
) -> list[dict[str, Any]]:
    from autogate_router.data import spans_from_tags
    from autogate_router.tag_routing import (
        DEFAULT_THRESHOLDS,
        combine,
        decode_tags,
        route_from_tags,
        row_context,
    )

    tag_logits = outputs.get("tag_logits")
    if route_source not in ROUTE_SOURCES:
        raise ValueError(f"route_source must be one of {ROUTE_SOURCES}, got {route_source!r}")
    if route_source != "e2e" and not tag_logits:
        raise ValueError(f"route_source {route_source!r} needs a checkpoint with tag heads")
    thresholds = thresholds or DEFAULT_THRESHOLDS
    route_probs = softmax(outputs["route_logits"])
    conf_probs = softmax(outputs["route_logits"], temperature) if temperature else route_probs
    intent_probs = softmax(outputs["intent_logits"])
    tag_probs = {name: softmax(x) for name, x in (tag_logits or {}).items()}
    preds = []
    for k, (row, ex) in enumerate(zip(rows, examples, strict=True)):
        top3 = np.argsort(-intent_probs[k])[:3]
        bio = [labels.bio_tags[t] for t in outputs["span_tags"][k]]
        spans = spans_from_tags(row["utterance"], ex["offsets"], bio)
        e2e = labels.routes[int(route_probs[k].argmax())]
        pred: dict[str, Any] = {
            "id": row["id"],
            "route": e2e,
            "route_probs": [round(float(p), 6) for p in route_probs[k]],
            "intent": labels.intents[int(top3[0])],
            "intent_probs_top3": [
                [labels.intents[int(j)], round(float(intent_probs[k, j]), 6)] for j in top3
            ],
            "spans": spans,
            "confidence": round(float(conf_probs[k].max()), 6),
        }
        if tag_probs:
            probs = {
                name: {v: round(float(p), 6) for v, p in zip(labels.tags[name], tp[k], strict=True)}
                for name, tp in tag_probs.items()
            }
            tags = decode_tags(probs, conservative, thresholds)
            by_tags = route_from_tags(
                tags["capability"],
                tags["distraction"],
                tags["actuation"],
                row_context(row),
                bool(spans),
            ).route
            pred["tags"] = tags
            pred["tag_probs"] = probs
            pred["route_e2e"] = e2e
            pred["route_tags"] = str(by_tags)
            pred["route_combined"] = str(combine(e2e, by_tags, tags["actuation"]))
            pred["route"] = pred[f"route_{route_source}"]
        preds.append(pred)
    return preds


def with_route_source(preds: Sequence[dict], source: str) -> list[dict[str, Any]]:
    """Copies of ``preds`` whose ``route`` is ``route_<source>``."""
    if source not in ROUTE_SOURCES:
        raise ValueError(f"route source must be one of {ROUTE_SOURCES}, got {source!r}")
    out = []
    for p in preds:
        key = f"route_{source}"
        if key not in p:
            raise ValueError(f"prediction {p['id']!r} has no {key} (no tag heads?)")
        out.append({**p, "route": p[key]})
    return out


def write_all(preds: Sequence[dict], prefix: str | Path) -> list[Path]:
    """``PREFIX_e2e.jsonl``, ``PREFIX_tags.jsonl``, ``PREFIX_combined.jsonl``."""
    return [
        write_predictions(with_route_source(preds, src), Path(f"{prefix}_{src}.jsonl"))
        for src in ROUTE_SOURCES
    ]


def write_predictions(preds: Sequence[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=True) + "\n")
    return path


def predict(
    checkpoint: str | Path,
    rows: Sequence[dict],
    batch_size: int = 64,
    device: str = "auto",
    route_source: str = "e2e",
    conservative: bool = True,
    thresholds: Any = None,
) -> list[dict[str, Any]]:
    from autogate_router.data import build_examples
    from autogate_router.model import AutoGateRouter, checkpoint_tokenizer
    from autogate_router.train import resolve_device

    dev = resolve_device(device)
    model = AutoGateRouter.from_pretrained(checkpoint).to(dev)
    tokenizer = checkpoint_tokenizer(checkpoint)
    cfg = model.config
    examples = build_examples(rows, tokenizer, cfg.max_length, cfg.use_context, model.labels)
    outputs = infer(model, examples, batch_size, dev)
    return predictions_from_outputs(
        rows,
        examples,
        outputs,
        model.labels,
        load_temperature(checkpoint),
        route_source,
        conservative,
        thresholds,
    )


def main(argv: list[str] | None = None) -> Path | list[Path]:
    from autogate_router.data import load_rows
    from autogate_router.tag_routing import TagThresholds

    d = TagThresholds()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="a saved router directory")
    p.add_argument("--rows", type=Path, required=True, help="rows.jsonl or rows.parquet")
    p.add_argument("--split", default=None, help="comma-separated splits (default: all rows)")
    p.add_argument(
        "--out", type=Path, default=None, help="default: predictions.jsonl (unless --write-all)"
    )
    p.add_argument(
        "--route-source",
        choices=ROUTE_SOURCES,
        default="e2e",
        help="which route goes into the top-level 'route' field (default: e2e)",
    )
    p.add_argument(
        "--write-all",
        metavar="PREFIX",
        default=None,
        help="write PREFIX_e2e.jsonl, PREFIX_tags.jsonl and PREFIX_combined.jsonl",
    )
    p.add_argument(
        "--conservative-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="conservative tag decoding (default); --no-conservative-tags for argmax",
    )
    p.add_argument("--tau-safety-critical", type=float, default=d.safety_critical)
    p.add_argument("--tau-restricted", type=float, default=d.restricted)
    p.add_argument("--tau-high", type=float, default=d.high)
    p.add_argument(
        "--tau-medium-high", type=float, default=d.medium_high, help="on p(medium) + p(high)"
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-rows", type=int, default=None)
    a = p.parse_args(argv)
    thresholds = TagThresholds(
        a.tau_safety_critical, a.tau_restricted, a.tau_high, a.tau_medium_high
    )
    rows = load_rows(a.rows, a.split, a.limit_rows)
    preds = predict(
        a.checkpoint,
        rows,
        a.batch_size,
        a.device,
        a.route_source,
        a.conservative_tags,
        thresholds,
    )
    written: list[Path] = []
    if a.write_all:
        written += write_all(preds, a.write_all)
    if a.out is not None or not a.write_all:
        written.append(write_predictions(preds, a.out or Path("predictions.jsonl")))
    for path in written:
        print(f"wrote {len(preds)} predictions to {path}")
    return written[0] if len(written) == 1 else written


if __name__ == "__main__":
    main()
