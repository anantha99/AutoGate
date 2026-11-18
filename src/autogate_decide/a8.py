r"""A8's pre-registered selection rules, applied to fine-tune outputs (plan section 6, Phase 1).

    python -m autogate_decide.a8 check-proxy --frz runs/a8/R7-frz --ctl runs/a8/R7-ctl \
        --rows data/generated/full_aug/rows.jsonl
    python -m autogate_decide.a8 select --ctl runs/a8/R7-ctl --frz runs/a8/R7-frz \
        --candidates runs/a8/R7a runs/a8/R7b --rows data/generated/full_aug/rows.jsonl
    python -m autogate_decide.a8 outcome --ctl runs/a8/R7-ctl runs/a8/R7-ctl-s1 \
        --winner runs/a8/R7a runs/a8/R7a-s1 --frz runs/a8/R7-frz --rows ...

Each run directory is a ``autogate_decide.finetune`` output with ``finetune.json`` (per
epoch: val scores and ``proxy`` scores) and ``epoch<k>_proxy.json`` (row ids, routes and
predicted distraction on the proxy split).

Rules (``docs/plan-a8-finetune-a6.md``):

* **Reference:** R7-ctl's epoch with the lowest val SWE.
* **Val filter:** val SWE <= reference + 0.03 and val critical miss <= reference + 0.01.
* **Epoch of a run** (R7-ctl included): among epochs passing the filter, the lowest proxy
  critical miss; within 1 point, the lower proxy SWE. **Winner** between candidates: the
  same rule over all their epochs. R7-frz (no encoder update) uses its last epoch.
* **Paired intent bootstrap:** resample the proxy intents with replacement (all rows of
  a drawn intent together); statistic = (a - b) on critical miss rate, SWE or
  distraction dangerous-miss rate; 95% percentile interval.
* **Proxy validity:** R7-frz better than R7-ctl on proxy critical miss and on distraction
  dangerous miss, each interval of (frz - ctl) entirely below 0.
* **Outcome A:** on every seed pair, the winner beats R7-ctl on proxy critical miss and
  SWE (interval of winner - ctl below 0) and is no worse than R7-frz on either (point
  estimate of winner - frz <= 0). **B:** a winner exists but A fails. **C:** no candidate
  epoch passes the val filter.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SWE_TOL = 0.03
CRIT_TOL = 0.01
TIE = 0.01
N_BOOT = 2000


def load_run(d: str | Path) -> dict[str, Any]:
    d = Path(d)
    ft = json.loads((d / "finetune.json").read_text(encoding="utf-8"))
    ft["dir"] = str(d)
    return ft


def reference(ctl: Mapping[str, Any]) -> dict[str, float]:
    e = min(ctl["history"], key=lambda h: h["swe"])
    return {"epoch": e["epoch"], "swe": e["swe"], "critical_miss_rate": e["critical_miss_rate"]}


def passes(h: Mapping[str, Any], ref: Mapping[str, float]) -> bool:
    return (
        h["swe"] <= ref["swe"] + SWE_TOL
        and h["critical_miss_rate"] <= ref["critical_miss_rate"] + CRIT_TOL
    )


def pick(candidates: Sequence[tuple[str, Mapping[str, Any]]], ref: Mapping[str, float]):
    """(run name, epoch entry) by the rule, or None when nothing passes the filter."""
    ok = [(n, h) for n, run in candidates for h in run["history"] if passes(h, ref)]
    if not ok:
        return None
    best_crit = min(h["proxy"]["critical_miss_rate"] for _, h in ok)
    close = [(n, h) for n, h in ok if h["proxy"]["critical_miss_rate"] <= best_crit + TIE]
    return min(close, key=lambda nh: (nh[1]["proxy"]["swe"], nh[1]["proxy"]["critical_miss_rate"]))


def _proxy_preds(run: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    return json.loads((Path(run["dir"]) / f"epoch{epoch}_proxy.json").read_text("utf-8"))


def _row_stats(rows_by_id: Mapping[str, Mapping[str, Any]], preds: Mapping[str, Any]):
    from autogate_eval.cost_matrix import default_cost_matrix
    from autogate_eval.metrics import _critical, _encode, _row_costs

    rows = [rows_by_id[i] for i in preds["ids"]]
    y, p, sc = _encode(rows, preds["routes"])
    eligible, miss = _critical(y, p, sc)
    cost = _row_costs(y, p, sc, default_cost_matrix())
    high = np.array([r["distraction"] == "high" for r in rows])
    dmiss = high & (np.array(preds["distraction"]) != "high")
    intents = np.array([r["intent"] for r in rows])
    return {
        "eligible": eligible,
        "miss": miss,
        "cost": cost,
        "high": high,
        "dmiss": dmiss,
        "intent": intents,
    }


def paired_bootstrap(
    a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], seed: int = 0
) -> dict[str, dict[str, float]]:
    """95% intervals of (a - b) for critical miss, SWE and distraction dangerous miss."""
    if list(a["intent"]) != list(b["intent"]) or len(a["miss"]) != len(b["miss"]):
        raise ValueError("paired bootstrap needs the same proxy rows in the same order")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(a["intent"]):
        groups[name].append(i)
    members = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)

    def stats(s, idx):
        el, hi = s["eligible"][idx].sum(), s["high"][idx].sum()
        return (
            s["miss"][idx].sum() / el if el else np.nan,
            s["cost"][idx].mean(),
            s["dmiss"][idx].sum() / hi if hi else np.nan,
        )

    full = np.arange(len(a["intent"]))
    point = np.array(stats(a, full)) - np.array(stats(b, full))
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        boots.append(np.array(stats(a, idx)) - np.array(stats(b, idx)))
    boots = np.array(boots)
    out = {}
    for k, name in enumerate(("critical_miss_rate", "swe", "distraction_dangerous_miss")):
        col = boots[:, k][~np.isnan(boots[:, k])]
        out[name] = {
            "diff": float(point[k]),
            "lo": float(np.percentile(col, 2.5)) if len(col) else float("nan"),
            "hi": float(np.percentile(col, 97.5)) if len(col) else float("nan"),
        }
    return out


def _rows(path: Path) -> dict[str, Mapping[str, Any]]:
    from autogate_router.data import load_rows

    return {r["id"]: r for r in load_rows(path)}


def check_proxy(frz_dir, ctl_dir, rows_path) -> dict[str, Any]:
    rows = _rows(rows_path)
    frz, ctl = load_run(frz_dir), load_run(ctl_dir)
    ref = reference(ctl)
    ctl_pick = pick([("R7-ctl", ctl)], ref)
    if ctl_pick is None:
        return {"valid": False, "why": "R7-ctl has no epoch passing its own filter"}
    f_epoch = frz["history"][-1]["epoch"]
    f = _row_stats(rows, _proxy_preds(frz, f_epoch))
    c = _row_stats(rows, _proxy_preds(ctl, ctl_pick[1]["epoch"]))
    bs = paired_bootstrap(f, c)
    valid = bs["critical_miss_rate"]["hi"] < 0 and bs["distraction_dangerous_miss"]["hi"] < 0
    return {
        "valid": bool(valid),
        "frz_epoch": f_epoch,
        "ctl_epoch": ctl_pick[1]["epoch"],
        "frz_minus_ctl": bs,
        "reference": ref,
    }


def select(ctl_dir, frz_dir, cand_dirs, rows_path) -> dict[str, Any]:
    ctl = load_run(ctl_dir)
    ref = reference(ctl)
    cands = [(Path(d).name, load_run(d)) for d in cand_dirs]
    w = pick(cands, ref)
    if w is None:
        return {"outcome": "C", "reference": ref, "why": "no candidate epoch passes the val filter"}
    ctl_pick = pick([("R7-ctl", ctl)], ref)
    return {
        "winner": w[0],
        "epoch": w[1]["epoch"],
        "entry": w[1],
        "reference": ref,
        "ctl_epoch": ctl_pick[1]["epoch"] if ctl_pick else None,
    }


def outcome(ctl_dirs, win_dirs, frz_dir, rows_path) -> dict[str, Any]:
    """Outcome A or B over seed pairs (ctl seed k, winner seed k); epochs re-picked per seed."""
    rows = _rows(rows_path)
    frz = load_run(frz_dir)
    f = _row_stats(rows, _proxy_preds(frz, frz["history"][-1]["epoch"]))
    per_seed, ok = [], True
    for cd, wd in zip(ctl_dirs, win_dirs, strict=True):
        ctl, win = load_run(cd), load_run(wd)
        ref = reference(ctl)
        cp, wp = pick([("ctl", ctl)], ref), pick([("win", win)], ref)
        if cp is None or wp is None:
            per_seed.append({"ctl": cd, "winner": wd, "a": False, "why": "val filter"})
            ok = False
            continue
        c = _row_stats(rows, _proxy_preds(ctl, cp[1]["epoch"]))
        w = _row_stats(rows, _proxy_preds(win, wp[1]["epoch"]))
        vs_ctl, vs_frz = paired_bootstrap(w, c), paired_bootstrap(w, f)
        a = (
            vs_ctl["critical_miss_rate"]["hi"] < 0
            and vs_ctl["swe"]["hi"] < 0
            and vs_frz["critical_miss_rate"]["diff"] <= 0
            and vs_frz["swe"]["diff"] <= 0
        )
        ok &= a
        per_seed.append(
            {
                "ctl": cd,
                "winner": wd,
                "ctl_epoch": cp[1]["epoch"],
                "winner_epoch": wp[1]["epoch"],
                "winner_minus_ctl": vs_ctl,
                "winner_minus_frz": vs_frz,
                "a": bool(a),
            }
        )
    return {"outcome": "A" if ok else "B", "per_seed": per_seed}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-proxy")
    c.add_argument("--frz", required=True)
    c.add_argument("--ctl", required=True)
    s = sub.add_parser("select")
    s.add_argument("--ctl", required=True)
    s.add_argument("--frz", required=True)
    s.add_argument("--candidates", nargs="+", required=True)
    o = sub.add_parser("outcome")
    o.add_argument("--ctl", nargs="+", required=True)
    o.add_argument("--winner", nargs="+", required=True)
    o.add_argument("--frz", required=True)
    for sp in (c, s, o):
        sp.add_argument("--rows", type=Path, required=True)
        sp.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.cmd == "check-proxy":
        res = check_proxy(a.frz, a.ctl, a.rows)
    elif a.cmd == "select":
        res = select(a.ctl, a.frz, a.candidates, a.rows)
    else:
        res = outcome(a.ctl, a.winner, a.frz, a.rows)
    text = json.dumps(res, indent=2, default=float)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
