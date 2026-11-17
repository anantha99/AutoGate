r"""One A7 run end to end: fit on train, choose on held-out intents and val, score.

    python -m autogate_decide.run --cache data/generated/decide/embeddinggemma-2 \
        --rows data/generated/full/rows.jsonl --pairs-rows data/generated/full_aug/rows.jsonl \
        --out runs/R6a --method heads --score val

``--score`` names the splits that are scored and written (default ``val``).
Every choice is made before any scoring and depends only on train and val
labels, so a configuration can be run with ``--score val``, committed, and
then run once with ``--score val,test,ood`` (plan section 3).

Method ``heads`` (runs R6a, R6a-ctl)
    1. **Selection on held-out intents.** For each question (and the joint
       combination), each head shape in the grid (hidden 0 or 256) and weight
       decay is trained on four fifths of the training *intents* and scored
       on the rest (``data.intent_folds``, tag-balanced). The configuration
       with the lowest out-of-fold NLL wins. Val holds only training intents,
       so val loss would reward memorising intents; held-out intents are the
       closest thing to OOD that training data offers. Val NLL is recorded
       beside it.
    2. **Mixing weight** lambda for the joint and product combination
       probabilities: the grid value with the lowest out-of-fold NLL of the
       true combination (out-of-fold temperatures).
    3. **Final heads** on all of train, one set per seed: plain heads for
       every question and the joint, weighted heads (R4's class weights) for
       the four tags. Temperatures per head on val.
    4. **Decoders**: ``conservative`` (weighted heads, R4's thresholds, the
       pre-registered Gate 2 decoder) and ``expected_cost`` (plain heads,
       mixed combination probabilities). Risk = p(critical miss) of the
       chosen route under the mixed probabilities.
    5. **Abstention**: per decoder, the largest-coverage risk threshold with
       val critical miss <= 1% on covered rows, applied unchanged elsewhere.

Method ``tokens`` (run R6a2)
    As ``heads``, on per-token states (``<cache>/tokens``, written by
    ``autogate_decide.embed --tokens``) with attention-pooling heads
    (:mod:`autogate_decide.token_heads`); linear only, weight decay 0.01 or 0.1.

Method ``zeroshot`` (run R6z)
    Cosine similarity of the utterance vector to each option description in
    ``configs/decide/questions.yaml`` (embedded with the same encoder and
    prefix), softmax with one temperature per question fitted on val. Both
    decoders run on these probabilities (product combinations only).

Outputs in ``--out``: ``config.json`` (every choice), ``report.json``,
``report.md``, and per decoder ``pred_<decoder>.jsonl`` for the scored
splits (seed 0). Each prediction line has the scorer's fields (``id``,
``route``, ``tags``, ``tag_probs``, ``confidence``) plus ``decoder``,
``sensitive``, ``risk`` and ``abstain``. For ``expected_cost`` files the
``tags`` are the argmax tags, so the scorer's route-changing tag errors
describe the tags, not that decoder's route.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autogate_decide import diagnostics as diag
from autogate_decide import route as rt
from autogate_decide.data import (
    QUESTIONS,
    ROUTING_TAGS,
    Utterances,
    build_utterances,
    intent_folds,
    joint_answers,
    joint_vocab,
)
from autogate_decide.embed import EmbeddingCache, sha256_file
from autogate_decide.heads import Head, HeadConfig, ece, fit_temperature, nll, softmax, train_head

TAGS = ("capability", "distraction", "actuation", "consequence")
DECODERS = ("conservative", "expected_cost")
DEFAULT_QUESTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "decide" / "questions.yaml"
)


@dataclass
class RunConfig:
    method: str = "heads"
    score: tuple[str, ...] = ("val",)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hidden_grid: tuple[int, ...] = (0, 256)
    weight_decay_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    steps: int = 300
    epochs: int = 6
    lr: float = 1e-2
    folds: int = 5
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    sensitive_threshold: float = 0.5
    abstain_target: float = 0.01
    n_boot: int = 1000
    selected: dict[str, dict[str, Any]] = field(default_factory=dict)
    lam: float | None = None
    registry: str = "none"  # none | fair | registered (Phase 3, R6b)
    lane: bool = False  # safety lane (Phase 3, R6l)
    tau: float | None = None


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #


def _labels(utts: Utterances, q: str, jvocab: Sequence) -> tuple[np.ndarray, int]:
    if q == "joint":
        return joint_answers(utts, jvocab), len(jvocab)
    return utts.answers[q], len(QUESTIONS[q])


def _weights(utts: Utterances, mask: np.ndarray, q: str) -> list[float] | None:
    if q not in TAGS:
        return None
    from autogate_router.data import tag_class_weights

    rows = [{t: QUESTIONS[t][utts.answers[t][i]] for t in TAGS} for i in np.flatnonzero(mask)]
    return tag_class_weights(rows)[q]


def select_heads(
    utts: Utterances, x: np.ndarray, cfg: RunConfig, jvocab: Sequence
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Grid search on held-out intents; returns the choice per question and its OOF logits."""
    train = utts.mask("train")
    val = utts.mask("val")
    folds = intent_folds(utts, train, cfg.folds, seed=0)
    selected: dict[str, dict[str, Any]] = {}
    oof_logits: dict[str, np.ndarray] = {}
    for q in (*QUESTIONS, "joint"):
        y, k = _labels(utts, q, jvocab)
        trials = []
        for h in cfg.hidden_grid:
            for wd in cfg.weight_decay_grid:
                hc = HeadConfig(
                    hidden=h, weight_decay=wd, lr=cfg.lr, steps=cfg.steps, epochs=cfg.epochs
                )
                oof = np.zeros((len(utts), k))
                for held in folds:
                    fit = train & ~held
                    head = train_head(x[fit], y[fit], k, hc, seed=0)
                    oof[held] = head.logits(x[held])
                held_all = train & np.any(folds, axis=0)
                oof_nll = nll(softmax(oof[held_all]), y[held_all])
                full = train_head(x[train], y[train], k, hc, seed=0)
                val_nll = nll(full.probs(x[val], calibrated=False), y[val])
                trials.append(
                    {
                        "hidden": h,
                        "weight_decay": wd,
                        "oof_nll": oof_nll,
                        "val_nll": val_nll,
                        "_oof": oof,
                    }
                )
                print(f"  select {q}: hidden {h} wd {wd}: oof {oof_nll:.4f} val {val_nll:.4f}")
        best = min(trials, key=lambda t: t["oof_nll"])
        oof_logits[q] = best.pop("_oof")
        for t in trials:
            t.pop("_oof", None)
        selected[q] = {
            "hidden": best["hidden"],
            "weight_decay": best["weight_decay"],
            "trials": trials,
        }
    return selected, oof_logits


def pick_lambda(
    utts: Utterances, oof_logits: Mapping[str, np.ndarray], cfg: RunConfig, jvocab: Sequence
) -> tuple[float, dict[str, float]]:
    from autogate_router.calibrate import fit_temperature as fit_t

    train = utts.mask("train")
    probs = {}
    for q, lg in oof_logits.items():
        y, _ = _labels(utts, q, jvocab)
        keep = train & (y >= 0)
        t = fit_t(lg[keep], y[keep])
        probs[q] = softmax(lg[train], t)
    true = np.array(
        [
            rt.COMBO_IDX[
                (
                    *(QUESTIONS[q][utts.answers[q][i]] for q in ROUTING_TAGS),
                    QUESTIONS["sensitive"][utts.answers["sensitive"][i]],
                )
            ]
            for i in np.flatnonzero(train)
        ]
    )
    scores = {}
    for lam in cfg.lambda_grid:
        pc = rt.combo_probs(probs, "mix", probs["joint"], jvocab, lam)
        scores[str(lam)] = float(
            -np.log(np.clip(pc[np.arange(len(true)), true], 1e-12, None)).mean()
        )
    best = min(cfg.lambda_grid, key=lambda lam: scores[str(lam)])
    return float(best), scores


@dataclass
class Fitted:
    plain: dict[str, Head]
    weighted: dict[str, Head]


def fit_heads(
    utts: Utterances, x: np.ndarray, cfg: RunConfig, jvocab: Sequence, seed: int
) -> Fitted:
    train, val = utts.mask("train"), utts.mask("val")
    plain, weighted = {}, {}
    for q in (*QUESTIONS, "joint"):
        y, k = _labels(utts, q, jvocab)
        s = cfg.selected[q]
        hc = HeadConfig(
            hidden=s["hidden"],
            weight_decay=s["weight_decay"],
            lr=cfg.lr,
            steps=cfg.steps,
            epochs=cfg.epochs,
        )
        plain[q] = train_head(x[train], y[train], k, hc, seed=seed)
        fit_temperature(plain[q], x[val], y[val])
        if q in TAGS:
            weighted[q] = train_head(x[train], y[train], k, hc, _weights(utts, train, q), seed)
            fit_temperature(weighted[q], x[val], y[val])
    weighted["sensitive"] = plain["sensitive"]
    return Fitted(plain, weighted)


# --------------------------------------------------------------------------- #
# Zero-shot
# --------------------------------------------------------------------------- #


def load_questions(path: str | Path = DEFAULT_QUESTIONS_PATH) -> dict[str, dict[str, str]]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for q, vocab in QUESTIONS.items():
        if tuple(str(k) for k in data[q]) != vocab:
            raise ValueError(f"{path}: {q} options must be {vocab} in that order")
    return {q: {str(k): v for k, v in data[q].items()} for q in QUESTIONS}


def zeroshot_probs(
    utts: Utterances, x: np.ndarray, encoder_name: str, questions_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, float], Any]:
    from autogate_decide.encoders import load_encoder
    from autogate_router.calibrate import fit_temperature as fit_t

    questions = load_questions(questions_path)
    enc = load_encoder(encoder_name)
    val = utts.mask("val")
    probs, temps, options = {}, {}, {}
    for q, opts in questions.items():
        options[q] = enc.encode(list(opts.values()))
        sims = x @ options[q].T
        t = fit_t(sims[val], utts.answers[q][val])
        probs[q] = softmax(sims, t)
        temps[q] = float(t)

    def prob_fn(xv: np.ndarray) -> dict[str, np.ndarray]:
        return {q: softmax(xv @ options[q].T, temps[q]) for q in options}

    return probs, temps, prob_fn


# --------------------------------------------------------------------------- #
# Decoding and predictions
# --------------------------------------------------------------------------- #


@dataclass
class Decoded:
    route: np.ndarray  # per row, route index
    risk: np.ndarray  # per row
    tags: list[dict[str, str]]  # per utterance
    sensitive: np.ndarray  # per utterance bool
    tag_probs: dict[str, np.ndarray]  # per utterance, the probabilities the decoder used


def decode(
    utts: Utterances,
    ctx: np.ndarray,
    plain: Mapping[str, np.ndarray],
    weighted: Mapping[str, np.ndarray],
    mode: str,
    jvocab: Sequence,
    lam: float,
    sensitive_threshold: float,
    force_product: np.ndarray | None = None,
) -> dict[str, Decoded]:
    tables = rt.route_tables()
    joint = plain.get("joint")
    pc_utt = rt.combo_probs(plain, mode, joint, jvocab, lam)
    if force_product is not None and force_product.any():
        pc_utt[force_product] = rt.combo_probs(
            {q: p[force_product] for q, p in plain.items()}, "product"
        )
    pc = pc_utt[utts.row_utt]
    out = {}
    combo, tags = rt.conservative_combo(weighted, sensitive_threshold)
    route = rt.decode_conservative(combo[utts.row_utt], ctx, tables)
    sens = np.array([t["sensitive"] == "yes" for t in tags])
    out["conservative"] = Decoded(
        route, _p_crit(pc, ctx, route, tables), tags, sens, {q: weighted[q] for q in TAGS}
    )
    route = _chunked_argmin(pc, ctx, tables)
    argmax_tags = [
        {q: QUESTIONS[q][int(plain[q][i].argmax())] for q in TAGS} for i in range(len(utts))
    ]
    sens = plain["sensitive"][:, 1] >= sensitive_threshold
    out["expected_cost"] = Decoded(
        route, _p_crit(pc, ctx, route, tables), argmax_tags, sens, {q: plain[q] for q in TAGS}
    )
    return out


def _chunked_argmin(
    pc: np.ndarray, ctx: np.ndarray, t: rt.RouteTables, n: int = 8192
) -> np.ndarray:
    return np.concatenate(
        [rt.decode_expected_cost(pc[i : i + n], ctx[i : i + n], t) for i in range(0, len(pc), n)]
    )


def _p_crit(pc: np.ndarray, ctx: np.ndarray, route: np.ndarray, t: rt.RouteTables) -> np.ndarray:
    return (pc * t.critical[ctx, :, route]).sum(-1)


def predictions(
    rows: Sequence[Mapping[str, Any]],
    idx: np.ndarray,
    utts: Utterances,
    dec: Decoded,
    name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    out = []
    for i in idx:
        u = utts.row_utt[i]
        out.append(
            {
                "id": rows[i]["id"],
                "route": rt.ROUTES[dec.route[i]],
                "decoder": name,
                "tags": {q: dec.tags[u][q] for q in TAGS},
                "tag_probs": {
                    q: {
                        v: round(float(p), 5)
                        for v, p in zip(QUESTIONS[q], dec.tag_probs[q][u], strict=True)
                    }
                    for q in TAGS
                },
                "sensitive": bool(dec.sensitive[u]),
                "risk": round(float(dec.risk[i]), 6),
                "confidence": round(1.0 - float(dec.risk[i]), 6),
                "abstain": bool(dec.risk[i] > threshold),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _gate_metrics(rep: Mapping[str, Any]) -> dict[str, Any]:
    o = rep["route"]["overall"]
    return {
        "n": o["n"],
        "macro_f1": o["macro_f1"],
        "macro_f1_ci95": o["macro_f1_ci95"],
        "swe": o["swe"],
        "swe_ci95": o["swe_ci95"],
        "critical_miss_rate": o["critical_miss_rate"],
        "critical_misses": o["critical_misses"],
        "n_critical": o["n_critical"],
        "adversarial": {
            k: rep["route"]["by_slice"]["adversarial"].get(k)
            for k in ("n", "macro_f1", "critical_miss_rate", "n_critical")
        },
    }


def score_split(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    preds_all: Mapping[str, list[dict[str, Any]]],
    utts: Utterances,
    decoded: Mapping[str, Decoded],
    thresholds: Mapping[str, float],
    probs: Mapping[str, np.ndarray],
    n_boot: int,
) -> dict[str, Any]:
    from autogate_eval.score import score

    predicted = {p["id"] for p in next(iter(preds_all.values())) if p["_split"] == split}
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == split and r["id"] in predicted])
    sub = [rows[i] for i in idx]
    out: dict[str, Any] = {}
    for name, dec in decoded.items():
        preds = [p for p in preds_all[name] if p["_split"] == split]
        preds = [{k: v for k, v in p.items() if k != "_split"} for p in preds]
        rep = score(sub, preds, n_boot=n_boot)
        routes = [p["route"] for p in preds]
        risk = dec.risk[idx]
        utt_idx = np.unique(utts.row_utt[idx])
        correct = {
            q: np.array(
                [dec.tags[u][q] == QUESTIONS[q][utts.answers[q][u]] for u in range(len(utts))]
            )
            for q in ROUTING_TAGS
        }
        true_sens = utts.answers["sensitive"] == 1
        out[name] = {
            "gate": _gate_metrics(rep),
            "intent_bootstrap": diag.intent_bootstrap(sub, routes),
            "abstention": rt.apply_threshold(sub, routes, risk, thresholds[name]),
            "abstention_scan": rt.abstention_scan(sub, routes, risk),
            "by_variant": diag.by_variant(sub, routes, utts.row_utt[idx], correct),
            "sensitive": diag.sensitive_scores(true_sens[utt_idx], dec.sensitive[utt_idx]),
            "tags": rep.get("tags", {}).get("overall"),
            "per_intent": diag.per_intent(sub, routes) if split == "ood" else None,
        }
    mask = utts.mask(split)
    out["calibration"] = {
        q: {"ece": ece(p[mask], utts.answers[q][mask])} for q, p in probs.items() if q in QUESTIONS
    }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(
    cache_dir: Path,
    rows_path: Path,
    out: Path,
    cfg: RunConfig,
    pairs_rows_path: Path | None = None,
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
) -> dict[str, Any]:
    from autogate_bench.dataset import git_commit
    from autogate_router.data import load_rows

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache.load(cache_dir)
    rows = load_rows(rows_path)
    utts = build_utterances(rows)
    lookup: Any = cache
    if cfg.method == "tokens":
        from autogate_decide.embed import TokenCache

        lookup = TokenCache.load(Path(cache_dir) / "tokens")
    x = lookup.lookup(utts.texts)
    ctx = np.array([rt.context_index(r) for r in rows])
    train = utts.mask("train")
    jvocab = joint_vocab(utts, train)
    print(
        f"{len(rows)} rows, {len(utts)} utterances, {train.sum()} train; joint vocab {len(jvocab)}"
    )

    seeds_out: dict[int, Any] = {}
    seed_runs = []
    if cfg.method in ("heads", "tokens"):
        if not cfg.selected:
            cfg.selected, oof = select_heads(utts, x, cfg, jvocab)
            cfg.lam, lam_scores = pick_lambda(utts, oof, cfg, jvocab)
        else:
            lam_scores = {}
        mode = "mix"
        for seed in cfg.seeds:
            fitted = fit_heads(utts, x, cfg, jvocab, seed)
            plain = {q: h.probs(x) for q, h in fitted.plain.items()}
            weighted = {q: h.probs(x) for q, h in fitted.weighted.items()}
            temps = {
                "plain": {q: h.temperature for q, h in fitted.plain.items()},
                "weighted": {q: h.temperature for q, h in fitted.weighted.items()},
            }
            raw_ece_val = {
                q: ece(
                    h.probs(x[utts.mask("val")], calibrated=False),
                    utts.answers[q][utts.mask("val")],
                )
                for q, h in fitted.plain.items()
                if q != "joint"
            }
            heads = fitted.weighted

            def prob_fn(xv, heads=heads):
                return {q: h.probs(xv) for q, h in heads.items()}

            seed_runs.append((seed, plain, weighted, temps, raw_ece_val, prob_fn))
    elif cfg.method == "zeroshot":
        cfg.seeds = (0,)
        probs, temps, prob_fn = zeroshot_probs(
            utts, x, cache.manifest["encoder"]["name"], questions_path
        )
        mode, cfg.lam, lam_scores = "product", 0.0, {}
        seed_runs.append((0, probs, probs, {"zeroshot": temps}, {}, prob_fn))
    else:
        raise ValueError(f"unknown method {cfg.method!r}")

    val_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    val_rows = [rows[i] for i in val_idx]
    probe = None
    phase3: dict[str, Any] = {}
    reg = lane = None
    if cfg.registry != "none":
        from autogate_decide.registry import build_registry, choose_tau

        xv = cache.lookup(utts.texts)  # the registry always matches pooled vectors
        if cfg.tau is None:
            cfg.tau, phase3["tau_scan"] = choose_tau(rows, utts, xv)
        reg = build_registry(rows, utts, xv, cfg.tau, cfg.registry)
        phase3["registry_size"] = len(reg.intents)
    if cfg.lane:
        from autogate_decide.registry import build_lane

        lane = build_lane(rows)
        phase3["lane_keywords"] = sorted(lane.keywords)
        phase3["lane_dropped"] = list(lane.dropped)
    for seed, plain, weighted, temps, raw_ece_val, prob_fn in seed_runs:
        fired = None
        if reg is not None:
            from autogate_decide.registry import apply_registry

            plain, hit = apply_registry(plain, reg, xv, jvocab)
            weighted, _ = apply_registry(weighted, reg, xv, jvocab)
            phase3["registry_hit_rate"] = {
                s_: float(hit[utts.mask(s_)].mean()) for s_ in ("train", "val", "test", "ood")
            }
        if lane is not None:
            from autogate_decide.registry import apply_lane

            fired = lane.fires(utts.texts)
            plain, weighted = apply_lane(plain, fired), apply_lane(weighted, fired)
            phase3["lane"] = lane_report(rows, utts, fired)
        decoded = decode(
            utts,
            ctx,
            plain,
            weighted,
            mode,
            jvocab,
            cfg.lam or 0.0,
            cfg.sensitive_threshold,
            fired,
        )
        thresholds = {
            name: rt.pick_threshold(
                val_rows,
                [rt.ROUTES[k] for k in d.route[val_idx]],
                d.risk[val_idx],
                cfg.abstain_target,
            )
            for name, d in decoded.items()
        }
        score_idx = np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["split"] in cfg.score
                # registered mode: the OOD seeds are in the registry, so they are not scored
                and not (
                    cfg.registry == "registered" and r["split"] == "ood" and r["source"] == "seed"
                )
            ]
        )
        preds = {}
        for name, d in decoded.items():
            ps = predictions(rows, score_idx, utts, d, name, thresholds[name])
            for p, i in zip(ps, score_idx, strict=True):
                p["_split"] = rows[i]["split"]
            preds[name] = ps
        per_split = {
            s: score_split(
                rows,
                s,
                preds,
                utts,
                decoded,
                thresholds,
                plain,
                cfg.n_boot if seed == cfg.seeds[0] else 50,
            )
            for s in cfg.score
        }
        if seed == cfg.seeds[0]:
            for name, ps in preds.items():
                with (out / f"pred_{name}.jsonl").open("w", encoding="utf-8") as f:
                    for p in ps:
                        f.write(json.dumps({k: v for k, v in p.items() if k != "_split"}) + "\n")
            if pairs_rows_path is not None:
                probe = minimal_pairs(pairs_rows_path, lookup, prob_fn, cfg)
        seeds_out[seed] = {
            "temperatures": temps,
            "ece_val_before": raw_ece_val,
            "thresholds": thresholds,
            "splits": per_split,
        }
        print(f"seed {seed} done ({time.time() - t0:.0f}s)")

    report = {
        "method": cfg.method,
        "cache": cache.manifest,
        "rows": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
        "config": _config_dict(cfg),
        "lambda_scores": lam_scores,
        "joint_vocab": [list(c) for c in jvocab],
        "seeds": seeds_out,
        "phase3": phase3,
        "seed_summary": seed_summary(seeds_out),
        "minimal_pairs": probe,
        "git": git_commit(),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "config.json").write_text(json.dumps(_config_dict(cfg), indent=2) + "\n", "utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", "utf-8")
    (out / "report.md").write_text(format_report(report), encoding="utf-8")
    print(format_report(report))
    return report


def lane_report(
    rows: Sequence[Mapping[str, Any]], utts: Utterances, fired: np.ndarray
) -> dict[str, Any]:
    """Per split: rows the lane fires on, how many are true REFUSE (must be 0) and its
    recall on true safety-critical rows."""
    out = {}
    for split in ("train", "val", "test", "ood"):
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        f = fired[utts.row_utt[idx]]
        refuse = np.array([rows[i]["route"] == "REFUSE" for i in idx])
        sc = np.array([rows[i]["actuation"] == "safety_critical" for i in idx])
        out[split] = {
            "rows_fired": int(f.sum()),
            "fired_on_refuse": int((f & refuse).sum()),
            "fired_on_non_safety_critical": int((f & ~sc).sum()),
            "recall_safety_critical": float(f[sc].mean()) if sc.any() else None,
        }
    return out


def _config_dict(cfg: RunConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["selected"] = {
        q: {k: v for k, v in s.items() if k != "trials"} for q, s in cfg.selected.items()
    }
    d["selection_trials"] = {q: s.get("trials") for q, s in cfg.selected.items()}
    return d


def minimal_pairs(
    pairs_rows_path: Path, cache: Any, prob_fn: Any, cfg: RunConfig
) -> dict[str, Any] | None:
    """Conservative tags for the minimal-pair utterances (never trained on), and the probe."""
    from autogate_bench.augment import DEFAULT_AUGMENT_DIR, PAIRS_FILE, load_pairs
    from autogate_router.data import load_rows

    pair_rows = [r for r in load_rows(pairs_rows_path) if r.get("pair_id")]
    if not pair_rows:
        return None
    texts = list(dict.fromkeys(r["utterance"] for r in pair_rows))
    _, tags = rt.conservative_combo(prob_fn(cache.lookup(texts)), cfg.sensitive_threshold)
    differs = {p.id: list(p.differs_in) for p in load_pairs(DEFAULT_AUGMENT_DIR / PAIRS_FILE)}
    return diag.minimal_pair_probe(pair_rows, dict(zip(texts, tags, strict=True)), differs)


def seed_summary(seeds_out: Mapping[int, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seeds = list(seeds_out)
    if not seeds:
        return out
    first = seeds_out[seeds[0]]["splits"]
    for split, by_dec in first.items():
        for dec in DECODERS:
            if dec not in by_dec:
                continue
            for metric in ("macro_f1", "swe", "critical_miss_rate"):
                vals = [seeds_out[s]["splits"][split][dec]["gate"][metric] for s in seeds]
                vals = [v for v in vals if v is not None]
                out[f"{split}/{dec}/{metric}"] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
    return out


def _f(x: Any, d: int = 3) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def format_report(rep: Mapping[str, Any]) -> str:
    seeds = rep["seeds"]
    s0 = seeds[next(iter(seeds))]
    lines = [
        f"# A7 run: {rep['method']} on {rep['cache']['encoder']['name']}",
        "",
        f"rows sha256 `{rep['rows']['sha256'][:8]}...`, encoder revision "
        f"`{(rep['cache']['encoder']['revision'] or '-')[:8]}`, lambda {rep['config']['lam']}, "
        f"seeds {list(seeds)}, {rep['seconds']}s",
        "",
        "## Gate metrics (seed 0; no abstention)",
        "",
        "| split | decoder | macro-F1 | SWE | critical miss | adversarial crit "
        "| intent-bootstrap crit CI |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split, by_dec in s0["splits"].items():
        for dec in DECODERS:
            g = by_dec[dec]["gate"]
            ci = by_dec[dec]["intent_bootstrap"]["critical_miss_rate"]
            lines.append(
                f"| {split} | {dec} | {_f(g['macro_f1'])} | {_f(g['swe'])} "
                f"| {_f(g['critical_miss_rate'])} ({g['critical_misses']}/{g['n_critical']}) "
                f"| {_f(g['adversarial']['critical_miss_rate'])} "
                f"| {('[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']') if ci else '-'} |"
            )
    lines += [
        "",
        "## Seeds (mean ± sd over seeds)",
        "",
        "| key | mean | sd | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k, v in rep["seed_summary"].items():
        lines.append(f"| {k} | {_f(v['mean'])} | {_f(v['sd'])} | {_f(v['min'])} | {_f(v['max'])} |")
    lines += [
        "",
        "## Abstention (threshold chosen on val, seed 0)",
        "",
        "| split | decoder | threshold | coverage | crit miss covered (UCB95) | macro-F1 | SWE "
        "| abstained eligible | abstained safety-critical |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split, by_dec in s0["splits"].items():
        for dec in DECODERS:
            a = by_dec[dec]["abstention"]
            lines.append(
                f"| {split} | {dec} | {_f(a['threshold'], 4)} | {_f(a['coverage'])} "
                f"| {_f(a['critical_miss_rate'])} ({_f(a['critical_miss_ucb95'])}) "
                f"| {_f(a['macro_f1'])} | {_f(a['swe'])} | {_f(a['abstained_eligible_rate'])} "
                f"| {_f(a['abstained_safety_critical_rate'])} |"
            )
    lines += [
        "",
        "## Per variant (conservative, seed 0)",
        "",
        "| split | variant | rows | acc cap | acc dis | acc act | crit miss |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split, by_dec in s0["splits"].items():
        for v, m in by_dec["conservative"]["by_variant"].items():
            lines.append(
                f"| {split} | {v} | {m['n_rows']} | {_f(m['acc_capability'])} "
                f"| {_f(m['acc_distraction'])} | {_f(m['acc_actuation'])} "
                f"| {_f(m['critical_miss_rate'])} ({m['critical_misses']}/{m['n_critical']}) |"
            )
    lines += [
        "",
        "## Tags (conservative, seed 0)",
        "",
        "| split | tag | accuracy | dangerous miss |",
        "| --- | --- | --- | --- |",
    ]
    for split, by_dec in s0["splits"].items():
        for t, m in (by_dec["conservative"]["tags"] or {}).items():
            lines.append(
                f"| {split} | {t} | {_f(m['accuracy'])} | {_f(m['dangerous_miss_rate'])} |"
            )
    lines += [
        "",
        "## Sensitive and calibration (seed 0)",
        "",
        "| split | sensitive recall | precision | ECE cap | ECE dis | ECE act | ECE sens |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split, by_dec in s0["splits"].items():
        s = by_dec["conservative"]["sensitive"]
        c = by_dec["calibration"]
        lines.append(
            f"| {split} | {_f(s['recall'])} | {_f(s['precision'])} | {_f(c['capability']['ece'])} "
            f"| {_f(c['distraction']['ece'])} | {_f(c['actuation']['ece'])} "
            f"| {_f(c['sensitive']['ece'])} |"
        )
    mp = rep.get("minimal_pairs")
    if mp:
        lines += [
            "",
            f"## Minimal-pair probe: {_f(mp['both_sides_right'])} of {mp['n_pairs']} "
            "pairs have every routing tag right on both sides",
            "",
        ]
        lines += [f"- differs in {a}: {_f(v)}" for a, v in mp["by_differing_tag"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--pairs-rows", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=("heads", "tokens", "zeroshot"), default="heads")
    p.add_argument("--score", default="val", help="comma-separated splits to score")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.json of an earlier run: reuse its selection (no re-search)",
    )
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--registry", choices=("none", "fair", "registered"), default="none")
    p.add_argument("--lane", action="store_true", help="safety lane (Phase 3)")
    a = p.parse_args(argv)
    cfg = RunConfig(
        method=a.method,
        score=tuple(s.strip() for s in a.score.split(",")),
        seeds=tuple(int(s) for s in a.seeds.split(",")),
        n_boot=a.n_boot,
        registry=a.registry,
        lane=a.lane,
    )
    if a.method == "tokens":  # attention pooling: linear only, a smaller grid
        cfg.hidden_grid, cfg.weight_decay_grid = (0,), (0.01, 0.1)
    if a.config:
        old = json.loads(a.config.read_text(encoding="utf-8"))
        old = old.get("config", old)  # a frozen configs/decide/<run>.json nests it
        cfg.selected = old["selected"]
        cfg.lam = old["lam"]
        cfg.tau = old.get("tau")
    return run(a.cache, a.rows, a.out, cfg, a.pairs_rows)


if __name__ == "__main__":
    main()
