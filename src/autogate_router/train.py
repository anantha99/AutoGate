r"""Train the AutoGate router (LoRA + three heads) with a plain PyTorch loop.

    python -m autogate_router.train --config configs/train_qwen3_0.6b.yaml \
        --rows data/generated/full/rows.jsonl --out-dir checkpoints/qwen3-0.6b

Every :class:`TrainConfig` field is a flag (``lora_r`` -> ``--lora-r``;
booleans also take ``--no-<name>``, e.g. ``--no-context`` for the
no-context ablation and ``--no-lora`` for full fine-tuning). ``--config``
loads defaults from YAML, and flags on the command line override it.

The loop: AdamW, linear warmup then linear decay, gradient accumulation,
gradient clipping, bf16 autocast on CUDA when supported (fp16 with a grad
scaler otherwise, fp32 on CPU). At the end of every epoch (and when
``--max-steps`` stops training early) it scores the val split: route
macro-F1, SWE and critical miss rate via ``autogate_eval.metrics``, intent
accuracy and span token-F1. The checkpoint with the lowest val SWE is saved
to ``<out-dir>/best``; ``<out-dir>/history.json`` has the step log and the
per-epoch metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import random
import sys
import time
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autogate_router.model import DEFAULT_BACKBONE, DEFAULT_LORA_TARGETS, POOLINGS

BEST_DIR = "best"
HISTORY_FILE = "history.json"


@dataclass
class TrainConfig:
    # data
    rows: str = "data/generated/full/rows.jsonl"
    train_split: str = "train"
    val_split: str = "val"
    limit_rows: int | None = None  # sample this many rows from each of train and val
    out_dir: str = "checkpoints/router"
    cost_matrix: str | None = None  # default: configs/cost_matrix.json
    # model
    backbone: str = DEFAULT_BACKBONE
    pooling: str = field(default="mean", metadata={"choices": POOLINGS})
    context: bool = True  # --no-context drops the context prefix (ablation)
    max_length: int = 96
    lora: bool = True  # --no-lora fine-tunes the whole backbone
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    head_dropout: float = 0.1
    class_weights: bool = True  # --no-class-weights: unweighted route loss
    # loss
    lambda_intent: float = 0.5
    lambda_span: float = 1.0
    # optimisation
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 2e-4
    head_lr: float | None = None  # default: same as lr
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    max_steps: int | None = None  # stop after this many optimizer steps
    precision: str = field(default="auto", metadata={"choices": ("auto", "bf16", "fp16", "fp32")})
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    # logging and eval
    log_every: int = 50
    eval_n_boot: int = 200


# --------------------------------------------------------------------------- #
# Config and CLI
# --------------------------------------------------------------------------- #


def _base_type(tp: Any) -> tuple[Any, bool]:
    """(inner type, optional?) for ``X | None`` annotations."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return args[0], True
    return tp, False


def _list_of_str(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _optional(conv):
    def parse(value: str):
        return None if value.lower() in ("none", "null", "") else conv(value)

    return parse


def build_parser(
    cls: type = TrainConfig, description: str | None = None
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description or __doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None, help="YAML file of defaults")
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        tp, optional = _base_type(hints[f.name])
        flag = "--" + f.name.replace("_", "-")
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if tp is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
            continue
        if typing.get_origin(tp) is list:
            conv = _list_of_str
            shown = ",".join(default) if default else default
            help_ = f"comma-separated (default: {shown})"
        else:
            conv = tp
            help_ = f"default: {default}"
        p.add_argument(
            flag,
            type=_optional(conv) if optional else conv,
            default=default,
            choices=f.metadata.get("choices"),
            help=help_,
        )
    return p


def load_yaml_defaults(path: Path, cls: type = TrainConfig) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in dataclasses.fields(cls)}
    out: dict[str, Any] = {}
    for k, v in data.items():
        key = k.replace("-", "_")
        if key not in known:
            raise ValueError(f"{path}: unknown config key {k!r}")
        if key == "lora_target_modules" and isinstance(v, str):
            v = _list_of_str(v)
        out[key] = v
    return out


def parse_config(argv: list[str] | None = None, cls: type = TrainConfig) -> Any:
    parser = build_parser(cls)
    pre, _ = parser.parse_known_args(argv)
    if pre.config is not None:
        parser.set_defaults(**load_yaml_defaults(pre.config, cls))
    ns = vars(parser.parse_args(argv))
    ns.pop("config")
    return cls(**ns)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> Any:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_precision(precision: str, device: Any) -> Any:
    """The autocast dtype, or None for fp32."""
    import torch

    if device.type != "cuda":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def linear_schedule(warmup: int, total: int):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    return fn


def span_token_f1(pred: list[list[int]], gold: list[list[int]], o_index: int = 0) -> dict:
    """Micro F1 over non-O subword tags (exact tag match); ``-100`` gold is skipped."""
    tp = fp = fn = 0
    for ps, gs in zip(pred, gold, strict=True):
        for p, g in zip(ps, gs, strict=True):
            if g < 0:
                continue
            if p != o_index and p == g:
                tp += 1
            else:
                fp += p != o_index
                fn += g != o_index
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def evaluate_split(
    model: Any, rows: list[dict], examples: list[dict], batch_size: int, device: Any, n_boot: int
) -> dict[str, Any]:
    """Val metrics for one pass of the model."""
    from autogate_eval.metrics import evaluate
    from autogate_router.predict import infer

    out = infer(model, examples, batch_size, device)
    labels = model.labels
    routes = [labels.routes[i] for i in out["route_logits"].argmax(-1)]
    intents = [labels.intents[i] for i in out["intent_logits"].argmax(-1)]
    m = evaluate(rows, routes, n_boot=n_boot)
    intent_acc = float(np.mean([p == r["intent"] for p, r in zip(intents, rows, strict=True)]))
    span = span_token_f1(out["span_tags"], [ex["span_labels"] for ex in examples])
    return {
        "n": m["n"],
        "swe": m["swe"],
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "mcc": m["mcc"],
        "critical_miss_rate": m["critical_miss_rate"],
        "critical_misses": m["critical_misses"],
        "intent_accuracy": intent_acc,
        "span_token_f1": span["f1"],
        "span_token_precision": span["precision"],
        "span_token_recall": span["recall"],
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(cfg: TrainConfig) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from autogate_router.data import (
        RouterDataset,
        class_weights_from_cost_matrix,
        collate,
        load_rows,
    )
    from autogate_router.model import AutoGateRouter, RouterConfig, load_tokenizer

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    amp_dtype = resolve_precision(cfg.precision, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2) + "\n", encoding="utf-8"
    )

    tokenizer = load_tokenizer(cfg.backbone)
    train_rows = load_rows(cfg.rows, cfg.train_split, cfg.limit_rows, cfg.seed)
    val_rows = load_rows(cfg.rows, cfg.val_split, cfg.limit_rows, cfg.seed)
    if not train_rows:
        raise SystemExit(f"no rows in split {cfg.train_split!r} of {cfg.rows}")
    ds_kw = dict(max_length=cfg.max_length, use_context=cfg.context)
    train_ds = RouterDataset(train_rows, tokenizer, **ds_kw)
    val_ds = RouterDataset(val_rows, tokenizer, **ds_kw)

    weights = class_weights_from_cost_matrix(cfg.cost_matrix) if cfg.class_weights else None
    rcfg = RouterConfig(
        backbone=cfg.backbone,
        pooling=cfg.pooling,
        use_context=cfg.context,
        max_length=cfg.max_length,
        use_lora=cfg.lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=list(cfg.lora_target_modules),
        head_dropout=cfg.head_dropout,
        lambda_intent=cfg.lambda_intent,
        lambda_span=cfg.lambda_span,
        route_class_weights=weights,
    )
    model = AutoGateRouter(rcfg).to(device)
    print(f"device {device}, autocast {amp_dtype}, {model.trainable_summary()}", flush=True)
    print(
        f"train rows {len(train_ds)}, val rows {len(val_ds)}, route class weights "
        + (
            ", ".join(f"{r}={w:.3f}" for r, w in zip(model.labels.routes, weights, strict=True))
            if weights
            else "none"
        ),
        flush=True,
    )

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=cfg.num_workers,
        collate_fn=functools.partial(collate, pad_token_id=pad_id),
    )

    head_ids = {id(p) for p in model.head_parameters()}
    decay, no_decay, heads = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            heads.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": cfg.lr},
        {"params": heads, "weight_decay": cfg.weight_decay, "lr": cfg.head_lr or cfg.lr},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]])
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    warmup = int(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_schedule(warmup, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    history: dict[str, Any] = {
        "config": dataclasses.asdict(cfg),
        "route_class_weights": weights,
        "total_steps": total_steps,
        "steps": [],
        "epochs": [],
        "best": None,
    }
    best_key: tuple[float, float] | None = None
    step = 0
    window: dict[str, list[float]] = {"loss": [], "route": [], "intent": [], "span": []}
    t0 = time.time()
    done = False

    def run_eval(epoch: int) -> None:
        nonlocal best_key
        model.eval()
        m = evaluate_split(
            model, val_rows, val_ds.examples, cfg.eval_batch_size, device, cfg.eval_n_boot
        )
        model.train()
        m.update(epoch=epoch, step=step)
        history["epochs"].append(m)
        crit = m["critical_miss_rate"]
        print(
            f"[eval] epoch {epoch} step {step}: val SWE {m['swe']:.4f} macro-F1 "
            f"{m['macro_f1']:.4f} acc {m['accuracy']:.4f} critical miss "
            f"{'-' if crit is None else f'{crit:.4f}'} intent acc {m['intent_accuracy']:.4f} "
            f"span token-F1 {m['span_token_f1']:.4f}",
            flush=True,
        )
        key = (m["swe"], -m["macro_f1"])
        if best_key is None or key < best_key:
            best_key = key
            model.save_pretrained(out_dir / BEST_DIR, tokenizer)
            history["best"] = {"epoch": epoch, "step": step, **m}
            print(f"[eval] new best (val SWE {m['swe']:.4f}) saved to {out_dir / BEST_DIR}")
        (out_dir / HISTORY_FILE).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                out = model(**batch)
            loss = out["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()
            window["loss"].append(out["loss"].item())
            window["route"].append(out["route_loss"].item())
            window["intent"].append(out["intent_loss"].item())
            window["span"].append(out["span_loss"].item())
            last_in_epoch = i + 1 == len(loader)
            if (i + 1) % cfg.grad_accum and not last_in_epoch:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % cfg.log_every == 0 or step == 1 or step == total_steps:
                means = {k: float(np.mean(v)) for k, v in window.items()}
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(time.time() - t0, 1),
                    **means,
                }
                history["steps"].append(rec)
                print(
                    f"step {step}/{total_steps} epoch {epoch} loss {means['loss']:.4f} "
                    f"(route {means['route']:.4f} intent {means['intent']:.4f} "
                    f"span {means['span']:.4f}) lr {rec['lr']:.2e} {rec['elapsed_s']}s",
                    flush=True,
                )
                window = {k: [] for k in window}
            if step >= total_steps:
                done = True
                break
        run_eval(epoch)
        if done:
            break

    history["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / HISTORY_FILE).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"done in {history['elapsed_s']}s; best checkpoint {out_dir / BEST_DIR}")
    return history


def main(argv: list[str] | None = None) -> dict[str, Any]:
    cfg = parse_config(argv)
    return train(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
