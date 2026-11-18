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


def split_scores(
    rows: list[dict], utts: Any, probs: dict[str, np.ndarray], split: str = "val"
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Conservative-decoder scores on one split from per-utterance probabilities:
    route metrics, distraction dangerous-miss rate and capability accuracy, plus the
    row ids and routes (kept for A8's paired intent bootstrap)."""
    from autogate_decide import route as rt
    from autogate_eval.metrics import evaluate

    idx = np.array([i for i, r in enumerate(rows) if r["split"] == split])
    if len(idx) == 0:
        return {}, [], []
    combo, tags = rt.conservative_combo(probs)
    ctx = np.array([rt.context_index(rows[i]) for i in idx])
    routes = [
        rt.ROUTES[k]
        for k in rt.decode_conservative(combo[utts.row_utt[idx]], ctx, rt.route_tables())
    ]
    sub = [rows[i] for i in idx]
    m = evaluate(sub, routes, n_boot=20)
    pred = [tags[utts.row_utt[i]] for i in idx]
    high = [
        p["distraction"] != "high"
        for r, p in zip(sub, pred, strict=True)
        if r["distraction"] == "high"
    ]
    out = {
        k: m[k] for k in ("swe", "macro_f1", "critical_miss_rate", "critical_misses", "n_critical")
    }
    out["distraction_dangerous_miss"] = float(np.mean(high)) if high else None
    out["capability_accuracy"] = float(
        np.mean([p["capability"] == r["capability"] for r, p in zip(sub, pred, strict=True)])
    )
    out["_distraction"] = [p["distraction"] for p in pred]
    return out, [r["id"] for r in sub], routes


def val_swe(rows: list[dict], utts: Any, probs: dict[str, np.ndarray]) -> dict[str, float]:
    """Conservative-decoder route metrics on val (kept for callers of the A7 API)."""
    m, _, _ = split_scores(rows, utts, probs, "val")
    return {k: m[k] for k in ("swe", "macro_f1", "critical_miss_rate")}


def probe_pair_score(rows: list[dict], utts: Any, probs: dict[str, np.ndarray]) -> float | None:
    """Share of probe pairs (split ``probe``) with every routing tag right on both sides."""
    from autogate_decide import route as rt
    from autogate_decide.data import ROUTING_TAGS

    _, tags = rt.conservative_combo(probs)
    ok: dict[str, bool] = {}
    for i, r in enumerate(rows):
        if r["split"] == "probe" and r.get("pair_id"):
            t = tags[utts.row_utt[i]]
            ok[r["pair_id"]] = ok.get(r["pair_id"], True) and all(
                t[q] == r[q] for q in ROUTING_TAGS
            )
    return float(np.mean(list(ok.values()))) if ok else None


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
    holdout: Path | None = None,
    holdout_mode: str = "proxy",
    distill_weight: float = 0.0,
    frozen_cache: Path | None = None,
    freeze_encoder: bool = False,
    save_encoder: bool = True,
    keep: str = "best",
) -> dict[str, Any]:
    """A7 (R6c) and A8 (R7x) fine-tunes. A8 options:

    * ``holdout``/``holdout_mode``: :func:`autogate_decide.holdout.apply` on the rows
      (``proxy`` for selection runs, ``final`` for the final model); every epoch is
      then also scored on the ``proxy`` split and the probe pairs, and the proxy
      routes are saved per epoch (``epoch<k>_proxy.json``).
    * ``distill_weight`` (beta) with ``frozen_cache``: adds
      ``beta * mean(1 - cos(z, z_frozen))``, ``z_frozen`` from the frozen encoder's
      embedding cache.
    * ``freeze_encoder`` (R7-frz): no encoder update; with ``frozen_cache`` the pooled
      vectors come from the cache, so only the heads train, through the same path and
      scores as the fine-tuned runs.
    * ``save_encoder=False`` for selection runs (metrics only); ``keep="last"`` for the
      final model (the selected number of epochs, no further selection).
    """
    import torch

    from autogate_router.data import load_rows, tag_class_weights

    torch.manual_seed(seed)
    rows = load_rows(rows_path)
    holdout_report = None
    if holdout is not None:
        from autogate_decide.holdout import Holdout, apply, goal_texts

        rows, holdout_report = apply(
            rows, Holdout.load(holdout), goal_texts(), proxy=holdout_mode == "proxy"
        )
    utts = build_utterances(rows)
    train = np.flatnonzero(utts.mask("train"))
    if limit_train:
        train = np.random.default_rng(seed).permutation(train)[:limit_train]
    jv = joint_vocab(utts, utts.mask("train"))
    labels = {q: utts.answers[q] for q in QUESTIONS} | {"joint": joint_answers(utts, jv)}
    sizes = {q: len(v) for q, v in QUESTIONS.items()} | {"joint": len(jv)}
    tag_rows = [{t: QUESTIONS[t][utts.answers[t][i]] for t in QUESTIONS} for i in train]
    cw = tag_class_weights(tag_rows)

    teacher = None
    if frozen_cache is not None:
        from autogate_decide.embed import EmbeddingCache

        cache = EmbeddingCache.load(frozen_cache)
        teacher = torch.tensor(cache.lookup(utts.texts))  # [n_utterances, d], L2-normalised
    backbone, tok = _load_backbone(model_id, revision, text_only)
    if freeze_encoder:
        for p_ in backbone.parameters():
            p_.requires_grad_(False)
    elif not full:
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
    groups = [{"params": list(heads.parameters()), "lr": HEAD_LR}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    n_steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    n_warm = max(1, int(WARMUP * n_steps))

    def schedule(s: int) -> float:  # linear warmup, then linear decay to 0
        if s < n_warm:
            return (s + 1) / n_warm
        return max(0.0, (n_steps - s) / max(n_steps - n_warm, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    text_index = {t: i for i, t in enumerate(utts.texts)}

    def pooled(texts: list[str]) -> Any:
        if freeze_encoder and teacher is not None:
            return teacher[[text_index[t] for t in texts]] * 10.0
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
            if distill_weight and teacher is not None and not freeze_encoder:
                cos = torch.nn.functional.cosine_similarity(z / 10.0, teacher[idx], dim=-1)
                loss = loss + distill_weight * (1.0 - cos).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
            if k % 50 == 0:
                lv = float(loss.detach())
                print(f"  epoch {epoch} step {k}: loss {lv:.4f} ({time.time() - t0:.0f}s)")
        scored = np.flatnonzero(utts.mask("val", "proxy", "probe"))
        p_s = predict(scored)
        full_p = {q: np.zeros((len(utts), sizes[q])) for q in sizes}
        for q in sizes:
            full_p[q][scored] = p_s[q]
        m, _, _ = split_scores(rows, utts, full_p, "val")
        m.pop("_distraction", None)
        m = {k: m[k] for k in ("swe", "macro_f1", "critical_miss_rate")}
        entry = {
            "epoch": epoch,
            "train_loss": total / max(k + 1, 1),
            **m,
            "seconds": round(time.time() - t0, 1),
        }
        if holdout is not None:
            pm, ids, routes = split_scores(rows, utts, full_p, "proxy")
            if pm:
                dis = pm.pop("_distraction")
                entry["proxy"] = pm
                (out / f"epoch{epoch}_proxy.json").write_text(
                    json.dumps({"ids": ids, "routes": routes, "distraction": dis}),
                    encoding="utf-8",
                )
            entry["probe_pairs"] = probe_pair_score(rows, utts, full_p)
        history.append(entry)
        print(f"epoch {epoch}: {entry}")
        if best is None or m["swe"] < best["swe"] or keep == "last":
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
    if save_encoder and not freeze_encoder:
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
        "mode": "frozen" if freeze_encoder else ("full" if full else f"lora{lora_rank}"),
        "holdout": str(holdout) if holdout else None,
        "holdout_mode": holdout_mode if holdout else None,
        "holdout_report": holdout_report,
        "distill_weight": distill_weight,
        "frozen_cache": str(frozen_cache) if frozen_cache else None,
        "keep": keep,
        "rows_sha256": _sha(rows_path),
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
    if (out / "encoder").is_dir():
        (out / "encoder" / "autogate_encoder.json").write_text(
            json.dumps({"prefix": prefix, "dim": int(dim), "base": model_id, "revision": revision})
            + "\n",
            encoding="utf-8",
        )
    return report


def _sha(path: Path) -> str:
    from autogate_decide.embed import sha256_file

    return sha256_file(path)


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
    p.add_argument("--holdout", type=Path, default=None, help="A8: configs/decide/a8_holdout.yaml")
    p.add_argument("--holdout-mode", choices=("proxy", "final"), default="proxy")
    p.add_argument("--distill-weight", type=float, default=0.0)
    p.add_argument("--frozen-cache", type=Path, default=None, help="frozen encoder's embed cache")
    p.add_argument("--freeze-encoder", action="store_true", help="A8 R7-frz: heads only")
    p.add_argument("--no-save", action="store_true", help="selection runs: metrics only")
    p.add_argument("--keep", choices=("best", "last"), default="best")
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
        holdout=a.holdout,
        holdout_mode=a.holdout_mode,
        distill_weight=a.distill_weight,
        frozen_cache=a.frozen_cache,
        freeze_encoder=a.freeze_encoder,
        save_encoder=not a.no_save,
        keep=a.keep,
    )


if __name__ == "__main__":
    main()
