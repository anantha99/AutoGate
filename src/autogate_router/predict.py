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
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

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


def infer(model: Any, examples: Sequence[dict], batch_size: int, device: Any) -> dict[str, Any]:
    """Logits for every example: route [N, 5], intent [N, 65], per-example span tag ids."""
    import torch

    from autogate_router.data import collate

    pad_id = getattr(getattr(model.backbone, "config", None), "pad_token_id", None) or 0
    route, intent, span_tags = [], [], []
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
            tags = out["span_logits"].argmax(-1).cpu().numpy()
            span_tags += [tags[j, : len(ex["input_ids"])].tolist() for j, ex in enumerate(chunk)]
    model.train(was_training)
    n_r, n_i = model.config.n_routes, model.config.n_intents
    return {
        "route_logits": np.concatenate(route) if route else np.zeros((0, n_r), np.float32),
        "intent_logits": np.concatenate(intent) if intent else np.zeros((0, n_i), np.float32),
        "span_tags": span_tags,
    }


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
) -> list[dict[str, Any]]:
    from autogate_router.data import spans_from_tags

    route_probs = softmax(outputs["route_logits"])
    conf_probs = softmax(outputs["route_logits"], temperature) if temperature else route_probs
    intent_probs = softmax(outputs["intent_logits"])
    preds = []
    for k, (row, ex) in enumerate(zip(rows, examples, strict=True)):
        top3 = np.argsort(-intent_probs[k])[:3]
        tags = [labels.bio_tags[t] for t in outputs["span_tags"][k]]
        preds.append(
            {
                "id": row["id"],
                "route": labels.routes[int(route_probs[k].argmax())],
                "route_probs": [round(float(p), 6) for p in route_probs[k]],
                "intent": labels.intents[int(top3[0])],
                "intent_probs_top3": [
                    [labels.intents[int(j)], round(float(intent_probs[k, j]), 6)] for j in top3
                ],
                "spans": spans_from_tags(row["utterance"], ex["offsets"], tags),
                "confidence": round(float(conf_probs[k].max()), 6),
            }
        )
    return preds


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
        rows, examples, outputs, model.labels, load_temperature(checkpoint)
    )


def main(argv: list[str] | None = None) -> Path:
    from autogate_router.data import load_rows

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="a saved router directory")
    p.add_argument("--rows", type=Path, required=True, help="rows.jsonl or rows.parquet")
    p.add_argument("--split", default=None, help="comma-separated splits (default: all rows)")
    p.add_argument("--out", type=Path, default=Path("predictions.jsonl"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-rows", type=int, default=None)
    a = p.parse_args(argv)
    rows = load_rows(a.rows, a.split, a.limit_rows)
    preds = predict(a.checkpoint, rows, a.batch_size, a.device)
    out = write_predictions(preds, a.out)
    print(f"wrote {len(preds)} predictions to {out}")
    return out


if __name__ == "__main__":
    main()
