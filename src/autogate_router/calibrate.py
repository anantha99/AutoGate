r"""Temperature scaling and a confidence-threshold scan for the route head.

    python -m autogate_router.calibrate --checkpoint checkpoints/qwen3-0.6b/best \
        --rows data/generated/full/rows.jsonl

1. **Temperature.** Runs the checkpoint on the val split, fits one
   temperature T > 0 for the route logits by minimising the negative
   log-likelihood (LBFGS on log T, with a log-spaced grid as a fallback), and
   writes ``calibration.json`` beside the checkpoint:
   ``{temperature, ece_before, ece_after, nll_before, nll_after, n, split}``.
   ECE uses 15 equal-width bins over the max route probability.
   ``autogate_router.predict`` then reports ``confidence`` after T.

2. **Threshold scan.** For each threshold t in 0.50, 0.55, ..., 0.95, 0.96,
   ..., 0.99, the rows whose calibrated confidence is at least t are
   *covered*: the gate executes the router's route. Reported for them:
   coverage (fraction of rows covered), SWE, accuracy and the critical miss
   rate, via ``autogate_eval.metrics.evaluate``. The rest are *uncovered*:
   the gate neither executes nor refuses and asks the driver to clarify
   (CLARIFY). They are reported separately, since CLARIFY is not a route and
   has no cost-matrix entry: how many there are, how many the router would
   have got wrong, how many of those would have been critical misses, and
   how many are true safety-critical commands (for which asking again is
   itself a delay). The scan is written to ``threshold_scan.json`` and
   printed as a table. No threshold is chosen; the output names the
   smallest one whose critical miss rate on covered rows is below 1%.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autogate_router.predict import CALIBRATION_FILE, softmax

SCAN_FILE = "threshold_scan.json"
THRESHOLDS: tuple[float, ...] = tuple(
    round(x, 2) for x in [*np.arange(0.50, 0.951, 0.05), 0.96, 0.97, 0.98, 0.99]
)
CRITICAL_TARGET = 0.01


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the max probability, equal-width bins."""
    if len(y) == 0:
        return float("nan")
    conf = probs.max(-1)
    correct = probs.argmax(-1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """T minimising NLL: LBFGS on log T, checked against a log-spaced grid."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
    best_t = float(min(grid, key=lambda t: nll(logits, y, t)))
    try:
        import torch

        x = torch.tensor(logits, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.long)
        log_t = torch.tensor([np.log(best_t)], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(x / log_t.exp(), yt)
            loss.backward()
            return loss

        opt.step(closure)
        t = float(log_t.exp().item())
        if np.isfinite(t) and t > 0 and nll(logits, y, t) <= nll(logits, y, best_t):
            best_t = t
    except ImportError:
        pass
    return best_t


def calibration_summary(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    t = fit_temperature(logits, y)
    return {
        "temperature": t,
        "ece_before": ece(softmax(logits), y),
        "ece_after": ece(softmax(logits, t), y),
        "nll_before": nll(logits, y),
        "nll_after": nll(logits, y, t),
    }


def threshold_scan(
    rows: Sequence[dict],
    routes: Sequence[str],
    confidence: np.ndarray,
    thresholds: Sequence[float] = THRESHOLDS,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Coverage and covered-row metrics per threshold, plus what CLARIFY absorbs."""
    from autogate_eval.metrics import _L, _critical, _encode, evaluate

    y, p, sc = _encode(rows, routes)
    _, miss = _critical(y, p, sc)
    wrong = y != p
    n = len(rows)
    out = []
    for t in thresholds:
        cov = confidence >= t
        covered = [r for r, c in zip(rows, cov, strict=True) if c]
        m = evaluate(covered, [x for x, c in zip(routes, cov, strict=True) if c], n_boot=n_boot)
        unc = ~cov
        out.append(
            {
                "threshold": float(t),
                "coverage": float(cov.mean()) if n else 0.0,
                "n_covered": int(cov.sum()),
                "covered": {
                    "swe": m.get("swe"),
                    "accuracy": m.get("accuracy"),
                    "macro_f1": m.get("macro_f1"),
                    "critical_miss_rate": m.get("critical_miss_rate"),
                    "critical_misses": m.get("critical_misses", 0),
                    "n_critical": m.get("n_critical", 0),
                },
                "clarify": {
                    "n": int(unc.sum()),
                    "router_errors_avoided": int((wrong & unc).sum()),
                    "critical_misses_avoided": int((miss & unc).sum()),
                    "true_safety_critical_delayed": int((unc & sc & (y == _L)).sum()),
                    "true_routes": dict(
                        sorted(
                            Counter(r["route"] for r, u in zip(rows, unc, strict=True) if u).items()
                        )
                    ),
                },
            }
        )
    below = [
        s["threshold"]
        for s in out
        if s["covered"]["critical_miss_rate"] is not None
        and s["covered"]["critical_miss_rate"] < CRITICAL_TARGET
    ]
    return {
        "n": n,
        "thresholds": out,
        "smallest_threshold_critical_below_1pct": min(below) if below else None,
        "note": "No threshold is chosen automatically; uncovered rows go to CLARIFY.",
    }


def format_scan(scan: dict[str, Any]) -> str:
    def f(x: float | None) -> str:
        return "-" if x is None else f"{x:.3f}"

    lines = [
        "| t | coverage | SWE (covered) | acc (covered) | crit miss (covered) "
        "| clarify n | errors to clarify | crit to clarify | safety-crit delayed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in scan["thresholds"]:
        c, u = s["covered"], s["clarify"]
        crit = f"{f(c['critical_miss_rate'])} ({c['critical_misses']}/{c['n_critical']})"
        lines.append(
            f"| {s['threshold']:.2f} | {s['coverage']:.3f} | {f(c['swe'])} | {f(c['accuracy'])} "
            f"| {crit} | {u['n']} | {u['router_errors_avoided']} "
            f"| {u['critical_misses_avoided']} | {u['true_safety_critical_delayed']} |"
        )
    t = scan["smallest_threshold_critical_below_1pct"]
    lines.append("")
    lines.append(
        "smallest threshold with covered critical miss rate < 1%: "
        + (f"{t:.2f}" if t is not None else "none in the scan")
        + " (not applied; choose the operating point yourself)"
    )
    return "\n".join(lines)


def calibrate(
    checkpoint: str | Path,
    rows_path: str | Path,
    split: str = "val",
    batch_size: int = 64,
    device: str = "auto",
    limit_rows: int | None = None,
    out_dir: str | Path | None = None,
    n_boot: int = 200,
) -> dict[str, Any]:
    from autogate_router.data import build_examples, load_rows
    from autogate_router.model import AutoGateRouter, checkpoint_tokenizer
    from autogate_router.predict import infer
    from autogate_router.train import resolve_device

    dev = resolve_device(device)
    model = AutoGateRouter.from_pretrained(checkpoint).to(dev)
    tokenizer = checkpoint_tokenizer(checkpoint)
    rows = load_rows(rows_path, split, limit_rows)
    if not rows:
        raise SystemExit(f"no rows in split {split!r} of {rows_path}")
    cfg = model.config
    examples = build_examples(rows, tokenizer, cfg.max_length, cfg.use_context, model.labels)
    logits = infer(model, examples, batch_size, dev)["route_logits"]
    y = np.array([model.labels.route_index[r["route"]] for r in rows])

    cal = {**calibration_summary(logits, y), "n": len(rows), "split": split}
    probs = softmax(logits, cal["temperature"])
    routes = [model.labels.routes[i] for i in probs.argmax(-1)]
    scan = threshold_scan(rows, routes, probs.max(-1), n_boot=n_boot)
    scan.update(split=split, temperature=cal["temperature"])

    out = Path(out_dir or checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    (out / CALIBRATION_FILE).write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")
    (out / SCAN_FILE).write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    print(
        f"{split}: n={len(rows)} temperature {cal['temperature']:.4f}; "
        f"ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}; "
        f"NLL {cal['nll_before']:.4f} -> {cal['nll_after']:.4f}"
    )
    print(format_scan(scan))
    print(f"wrote {out / CALIBRATION_FILE} and {out / SCAN_FILE}")
    return {"calibration": cal, "scan": scan}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--split", default="val", help="split to fit T and scan on (default: val)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None, help="default: the checkpoint directory")
    p.add_argument("--n-boot", type=int, default=200)
    a = p.parse_args(argv)
    return calibrate(
        a.checkpoint, a.rows, a.split, a.batch_size, a.device, a.limit_rows, a.out_dir, a.n_boot
    )


if __name__ == "__main__":
    main()
