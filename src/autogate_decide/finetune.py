r"""Fine-tune the encoder end to end with the decision heads (Phase 4, run R6c).

    python -m autogate_decide.finetune --rows data/generated/full/rows.jsonl \
        --out checkpoints/R6c-lora16 --lora-rank 16 --epochs 3
    python -m autogate_decide.finetune ... --full          # full fine-tune, second

Only run when Gate 2 has not reached outcome A with frozen encoders (plan
section 6, Phase 4). fp32 throughout.

Model
    The encoder (EmbeddingGemma 2 text tower by default; any Hugging Face
    encoder path works, e.g. the tiny test backbone), mean pooling over the
    attention mask, L2, then one linear head per question (capability,
    distraction, actuation, consequence, sensitive) and the joint
    (capability, distraction, actuation) head. LoRA (rank ``r``, alpha 2r,
    dropout 0.05) on the attention projections (q, k, v, o), or ``--full``.
    AdamW: encoder lr 1e-4 (LoRA) or 2e-5 (full), heads in their own group
    at 1e-3, 6% linear warmup then linear decay. Pre-registered for R6c
    after an independent review (``docs/experiment-log.md``).

Loss
    Sum of cross-entropies weighted per question as R4 weighted its tags
    (actuation 1.0, distraction 1.0, capability 0.5, consequence 0.3) plus
    sensitive 0.5 and joint 0.5; tag classes weighted with
    ``autogate_router.data.tag_class_weights``. Unique train utterances,
    shuffled per epoch with a fixed seed.

Selection
    After every epoch the heads' probabilities on val (temperature 1, product
    of marginals) go through the conservative decoder and the rulebook, and
    the epoch with the lowest **val SWE** is kept (plan: "Select on val
    SWE"). The kept encoder is saved merged (``encoder/``, loadable with
    ``autogate_decide.encoders.load_encoder(<path>)``), so the usual
    ``embed`` -> ``run --method heads`` pipeline scores it with calibrated,
    freshly fitted heads on its own vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from autogate_decide.data import QUESTIONS, build_utterances, joint_answers, joint_vocab
from autogate_decide.encoders import ENCODERS

LOSS_WEIGHTS = {
    "capability": 0.5,
    "distraction": 1.0,
    "actuation": 1.0,
    "consequence": 0.3,
    "sensitive": 0.5,
    "joint": 0.5,
}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")  # attention only (R6c pre-registration)
HEAD_LR = 1e-3
PROBE_TEXTS = (
    "turn on the wipers",
    "how do I open the trunk",
    "open the trunk",
    "read my last message from Asha",
    "AC thoda kam karo",
)
WARMUP = 0.06


def _load_backbone(model_id: str, revision: str | None, text_only: bool):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if text_only:
        config.vision_config = None
        config.audio_config = None
    model = AutoModel.from_pretrained(
        model_id, config=config, revision=revision, dtype=torch.float32
    )
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def val_swe(rows: list[dict], utts: Any, probs: dict[str, np.ndarray]) -> dict[str, float]:
    """Conservative-decoder route metrics on val from per-utterance probabilities."""
    from autogate_decide import route as rt
    from autogate_eval.metrics import evaluate

    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    combo, _ = rt.conservative_combo(probs)
    ctx = np.array([rt.context_index(rows[i]) for i in idx])
    routes = rt.decode_conservative(combo[utts.row_utt[idx]], ctx, rt.route_tables())
    m = evaluate([rows[i] for i in idx], [rt.ROUTES[k] for k in routes], n_boot=20)
    return {k: m[k] for k in ("swe", "macro_f1", "critical_miss_rate")}


def finetune(
    rows_path: Path,
    out: Path,
    model_id: str = ENCODERS["embeddinggemma-2"].model_id,
    revision: str | None = ENCODERS["embeddinggemma-2"].revision,
    prefix: str = ENCODERS["embeddinggemma-2"].prefix,
    text_only: bool = True,
    lora_rank: int = 16,
    full: bool = False,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-4,
    max_length: int = 64,
    seed: int = 0,
    limit_train: int | None = None,
) -> dict[str, Any]:
    import torch

    from autogate_router.data import load_rows, tag_class_weights

    torch.manual_seed(seed)
    rows = load_rows(rows_path)
    utts = build_utterances(rows)
    train = np.flatnonzero(utts.mask("train"))
    if limit_train:
        train = np.random.default_rng(seed).permutation(train)[:limit_train]
    val = np.flatnonzero(utts.mask("val"))
    jv = joint_vocab(utts, utts.mask("train"))
    labels = {q: utts.answers[q] for q in QUESTIONS} | {"joint": joint_answers(utts, jv)}
    sizes = {q: len(v) for q, v in QUESTIONS.items()} | {"joint": len(jv)}
    tag_rows = [{t: QUESTIONS[t][utts.answers[t][i]] for t in QUESTIONS} for i in train]
    cw = tag_class_weights(tag_rows)

    backbone, tok = _load_backbone(model_id, revision, text_only)
    if not full:
        from peft import LoraConfig, get_peft_model

        present = {n.split(".")[-1] for n, _ in backbone.named_modules()}
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.05,
                target_modules=[t for t in LORA_TARGETS if t in present],
            ),
        )
    probe = tok(["x"], return_tensors="pt")
    with torch.no_grad():
        dim = backbone(**probe).last_hidden_state.shape[-1]
    heads = torch.nn.ModuleDict({q: torch.nn.Linear(dim, k) for q, k in sizes.items()})
    losses = {
        q: torch.nn.CrossEntropyLoss(
            weight=torch.tensor(cw[q], dtype=torch.float32) if q in cw else None,
            ignore_index=-1,
        )
        for q in sizes
    }
    enc_params = [p for p in backbone.parameters() if p.requires_grad]
    params = enc_params + list(heads.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr},
            {"params": list(heads.parameters()), "lr": HEAD_LR},
        ],
        weight_decay=0.01,
    )
    n_steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    n_warm = max(1, int(WARMUP * n_steps))

    def schedule(s: int) -> float:  # linear warmup, then linear decay to 0
        if s < n_warm:
            return (s + 1) / n_warm
        return max(0.0, (n_steps - s) / max(n_steps - n_warm, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    def pooled(texts: list[str]) -> Any:
        b = tok(
            [prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        h = backbone(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return torch.nn.functional.normalize(z, dim=-1) * 10.0

    def predict(index: np.ndarray) -> dict[str, np.ndarray]:
        backbone.eval()
        out: dict[str, list[np.ndarray]] = {q: [] for q in sizes}
        with torch.inference_mode():
            for i in range(0, len(index), 128):
                z = pooled([utts.texts[j] for j in index[i : i + 128]])
                for q, h in heads.items():
                    out[q].append(torch.softmax(h(z), -1).numpy())
        return {q: np.concatenate(v) for q, v in out.items()}

    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, Any] = {}
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "resume.pt"
    start = 1
    if ckpt.exists():  # resume after an interruption, at the next epoch boundary
        st = torch.load(ckpt, weights_only=False)
        named = dict(backbone.named_parameters())
        with torch.no_grad():
            for n, v in st["trainable"].items():
                named[n].copy_(v)
        heads.load_state_dict(st["heads"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        history, best, best_state, start = st["history"], st["best"], st["best_state"], st["next"]
        print(f"resumed at epoch {start} from {ckpt}")
    t0 = time.time()
    for epoch in range(start, epochs + 1):
        backbone.train()
        order = rng.permutation(train)
        total = 0.0
        for k, i in enumerate(range(0, len(order), batch_size)):
            idx = order[i : i + batch_size]
            z = pooled([utts.texts[j] for j in idx])
            loss = sum(
                LOSS_WEIGHTS[q] * losses[q](h(z), torch.tensor(labels[q][idx]).long())
                for q, h in heads.items()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
            if k % 50 == 0:
                print(f"  epoch {epoch} step {k}: loss {float(loss):.4f} ({time.time() - t0:.0f}s)")
        p_val = predict(val)
        full_p = {q: np.zeros((len(utts), sizes[q])) for q in sizes}
        for q in sizes:
            full_p[q][val] = p_val[q]
        m = val_swe(rows, utts, full_p)
        entry = {
            "epoch": epoch,
            "train_loss": total / max(k + 1, 1),
            **m,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        print(f"epoch {epoch}: {entry}")
        if best is None or m["swe"] < best["swe"]:
            best = entry
            best_state = {
                n: p.detach().clone() for n, p in backbone.named_parameters() if p.requires_grad
            }
        torch.save(
            {
                "trainable": {
                    n: p.detach().clone() for n, p in backbone.named_parameters() if p.requires_grad
                },
                "heads": heads.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "history": history,
                "best": best,
                "best_state": best_state,
                "next": epoch + 1,
            },
            ckpt.with_suffix(".tmp"),
        )
        os.replace(ckpt.with_suffix(".tmp"), ckpt)  # atomic: a restart never leaves half a file
    if best_state:
        with torch.no_grad():
            named = dict(backbone.named_parameters())
            for n, v in best_state.items():
                named[n].copy_(v)
    backbone.eval()
    with torch.inference_mode():  # what training saw, to check serving reproduces it
        probe = (pooled(list(PROBE_TEXTS)) / 10.0).numpy()
    np.save(out / "probe_vectors.npy", probe)
    enc = backbone if full else backbone.merge_and_unload()
    enc.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "encoder")
    report = {
        "model_id": model_id,
        "revision": revision,
        "prefix": prefix,
        "mode": "full" if full else f"lora{lora_rank}",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "head_lr": HEAD_LR,
        "warmup": WARMUP,
        "lora_targets": list(LORA_TARGETS) if not full else None,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "seed": seed,
        "n_train_utterances": int(len(train)),
        "history": history,
        "best": best,
    }
    (out / "finetune.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "encoder" / "autogate_encoder.json").write_text(
        json.dumps({"prefix": prefix, "dim": int(dim), "base": model_id, "revision": revision})
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--full", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--lr", type=float, default=None, help="encoder lr; default 1e-4 (LoRA), 2e-5 (full)"
    )
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    lr = a.lr if a.lr is not None else (2e-5 if a.full else 1e-4)
    return finetune(
        a.rows,
        a.out,
        lora_rank=a.lora_rank,
        full=a.full,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=lr,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
