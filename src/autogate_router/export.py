r"""Export a router checkpoint to ONNX (fp32) and dynamically quantised int8.

    python -m autogate_router.export --checkpoint checkpoints/qwen3-0.6b/best \
        --out-dir checkpoints/qwen3-0.6b/onnx --rows data/generated/full/rows.jsonl

Steps: load the checkpoint with eager attention, merge the LoRA adapter into
the backbone, trace the three-head model with the TorchScript exporter
(opset 17; inputs ``input_ids``, ``attention_mask``, ``utterance_mask``, all
int64 [batch, sequence]; outputs ``route_logits`` [batch, 5],
``intent_logits`` [batch, 65], ``span_logits`` [batch, sequence, 11]; batch
and sequence axes dynamic), then ``onnxruntime.quantization.quantize_dynamic``
to int8 weights (``--per-channel`` for per-channel scales). Both files are
checked against the torch logits on ``--n-examples`` rows from ``--rows``
(or four built-in examples) in two different padded shapes: fp32 must agree
within ``--atol`` (default 1e-3); for int8 the max difference and the
per-head argmax agreement are reported but not enforced, so check the int8
model on real rows before shipping it. The tokenizer, ``labels.json``,
``router_config.json`` and ``calibration.json`` (if any) are copied next to
the models so the directory is self-contained. Needs the ``export`` extra (onnx, onnxruntime).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

FP32_FILE = "router.onnx"
INT8_FILE = "router.int8.onnx"
REPORT_FILE = "export_report.json"
INPUT_NAMES = ("input_ids", "attention_mask", "utterance_mask")
OUTPUT_NAMES = ("route_logits", "intent_logits", "span_logits")

SAMPLE_ROWS = (
    {"utterance": "turn on the front defrost", "spans": []},
    {
        "utterance": "text arindam mukherjee that the otp is 2182",
        "spans": [
            {"start": 5, "end": 22, "label": "contact", "text": "arindam mukherjee"},
            {"start": 39, "end": 43, "label": "otp", "text": "2182"},
        ],
    },
    {"utterance": "whats the weather tomorrow", "spans": []},
    {"utterance": "unlock the back door please", "spans": []},
)
SAMPLE_PREFIXES = (
    "[speed=high][conn=none][workload=medium][privacy=strict][passenger=no][local=small]",
    "[speed=parked][conn=good][workload=low][privacy=standard][passenger=yes][local=tiny]",
)


def _file_size(path: Path) -> int:
    """Model size including its external-data file, if any."""
    data = path.with_name(path.name + ".data")
    return path.stat().st_size + (data.stat().st_size if data.exists() else 0)


def _mb(n: int) -> str:
    return f"{n / 2**20:.2f} MB"


class ExportWrapper(torch.nn.Module):
    """The router with positional inputs and a tuple of the three logits, for tracing."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, utterance_mask):
        out = self.m(input_ids, attention_mask, utterance_mask)
        return out["route_logits"], out["intent_logits"], out["span_logits"]


def sample_batches(
    tokenizer: Any, config: Any, rows: list[dict] | None = None, batch_size: int = 16
) -> list[dict[str, Any]]:
    """Input batches for tracing (the first) and verification (all of them)."""
    from autogate_router.data import build_examples, collate

    if not rows:
        rows = [
            {**r, "context_prefix": SAMPLE_PREFIXES[i % len(SAMPLE_PREFIXES)]}
            for i, r in enumerate(SAMPLE_ROWS)
        ]
    examples = build_examples(rows, tokenizer, config.max_length, config.use_context)
    batches = []
    for i in range(0, len(examples), batch_size):
        batch = collate(examples[i : i + batch_size], tokenizer.pad_token_id)
        batches.append({k: batch[k] for k in INPUT_NAMES})
    if len(examples) > 1:  # a second shape: batch minus its first row, re-padded
        batch = collate(examples[1 : min(len(examples), 4)], tokenizer.pad_token_id)
        batches.append({k: batch[k] for k in INPUT_NAMES})
    return batches


def compare(
    session: Any, wrapper: torch.nn.Module, batches: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Per output: max |onnx - torch| and argmax agreement (span logits on real tokens only)."""
    stats = {name: {"max_abs_diff": 0.0, "agree": 0, "n": 0} for name in OUTPUT_NAMES}
    for b in batches:
        feeds = {k: b[k].numpy().astype(np.int64) for k in INPUT_NAMES}
        with torch.no_grad():
            refs = [t.numpy() for t in wrapper(*(b[k] for k in INPUT_NAMES))]
        outs = session.run(list(OUTPUT_NAMES), feeds)
        mask = feeds["attention_mask"].astype(bool)
        for name, o, r in zip(OUTPUT_NAMES, outs, refs, strict=True):
            if name == "span_logits":
                o, r = o[mask], r[mask]
            st = stats[name]
            st["max_abs_diff"] = max(st["max_abs_diff"], float(np.abs(o - r).max()))
            st["agree"] += int((o.argmax(-1) == r.argmax(-1)).sum())
            st["n"] += int(o.shape[0])
    return {
        name: {"max_abs_diff": st["max_abs_diff"], "argmax_agreement": st["agree"] / st["n"]}
        for name, st in stats.items()
    }


def export(
    checkpoint: str | Path,
    out_dir: str | Path,
    rows: list[dict] | None = None,
    atol: float = 1e-3,
    opset: int = 17,
    quantize: bool = True,
    per_channel: bool = False,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    from autogate_router.model import (
        CONFIG_FILE,
        TOKENIZER_DIR,
        AutoGateRouter,
        checkpoint_tokenizer,
    )
    from autogate_router.predict import CALIBRATION_FILE

    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoGateRouter.from_pretrained(checkpoint, attn_implementation="eager", merge_lora=True)
    model = model.float().eval()
    tokenizer = checkpoint_tokenizer(checkpoint)
    wrapper = ExportWrapper(model).eval()
    batches = sample_batches(tokenizer, model.config, rows)
    args = tuple(batches[0][k] for k in INPUT_NAMES)

    n_params = sum(p.numel() for p in model.parameters())
    large = n_params * 4 > 1.8 * 2**30  # protobuf's 2 GB limit: use external data
    fp32 = out_dir / FP32_FILE
    dyn = {name: {0: "batch", 1: "sequence"} for name in INPUT_NAMES}
    dyn.update(
        route_logits={0: "batch"},
        intent_logits={0: "batch"},
        span_logits={0: "batch", 1: "sequence"},
    )
    tmp = out_dir / "_trace"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(tmp / FP32_FILE),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=dyn,
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    # re-save as one graph file, with the weights in one external file if too big for protobuf
    proto = onnx.load(str(tmp / FP32_FILE))
    for stale in (fp32, fp32.with_name(fp32.name + ".data")):
        stale.unlink(missing_ok=True)
    if large:
        onnx.save_model(
            proto,
            str(fp32),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=FP32_FILE + ".data",
        )
    else:
        onnx.save_model(proto, str(fp32))
    del proto
    shutil.rmtree(tmp, ignore_errors=True)
    onnx.checker.check_model(str(fp32))

    paths = {"fp32": fp32}
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8 = out_dir / INT8_FILE
        quantize_dynamic(
            str(fp32),
            str(int8),
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            use_external_data_format=large,
        )
        paths["int8"] = int8

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "opset": opset,
        "n_verify_examples": sum(len(b["input_ids"]) for b in batches[:-1] or batches),
        "models": {},
    }
    for kind, path in paths.items():
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outputs = compare(sess, wrapper, batches)
        result["models"][kind] = {
            "path": str(path),
            "bytes": _file_size(path),
            "max_abs_diff": max(o["max_abs_diff"] for o in outputs.values()),
            "outputs": outputs,
        }
    fp = result["models"]["fp32"]
    result["fp32_within_atol"] = fp["max_abs_diff"] <= atol
    result["atol"] = atol

    # make the directory self-contained
    for name in (CONFIG_FILE, "labels.json", CALIBRATION_FILE):
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, out_dir / name)
    tokenizer.save_pretrained(str(out_dir / TOKENIZER_DIR))
    (out_dir / REPORT_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"torch params {n_params:,}")
    for kind, m in result["models"].items():
        agree = ", ".join(
            f"{name.removesuffix('_logits')} {o['argmax_agreement']:.3f}"
            for name, o in m["outputs"].items()
        )
        print(
            f"{kind}: {m['path']} {_mb(m['bytes'])}, max |onnx - torch| {m['max_abs_diff']:.2e}, "
            f"argmax agreement with torch: {agree}"
        )
    if not result["fp32_within_atol"]:
        raise RuntimeError(f"fp32 ONNX differs from torch by {fp['max_abs_diff']:.2e} > {atol}")
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from autogate_router.data import load_rows

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <checkpoint>/onnx")
    p.add_argument("--rows", type=Path, default=None, help="take verification examples from here")
    p.add_argument("--split", default="val")
    p.add_argument("--n-examples", type=int, default=64, help="verification rows from --rows")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--per-channel", action="store_true", help="per-channel int8 weights")
    a = p.parse_args(argv)
    rows = load_rows(a.rows, a.split, a.n_examples) if a.rows else None
    return export(
        a.checkpoint,
        a.out_dir or a.checkpoint / "onnx",
        rows,
        a.atol,
        a.opset,
        not a.no_quantize,
        a.per_channel,
    )


if __name__ == "__main__":
    main()
