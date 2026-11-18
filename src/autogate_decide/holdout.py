r"""A8's held-out selection data: the proxy holdout and the probe pairs.

    python -m autogate_decide.holdout pick --rows data/generated/full_aug/rows.jsonl \
        --out configs/decide/a8_holdout.yaml
    python -m autogate_decide.holdout report --rows data/generated/full_aug/rows.jsonl

See ``docs/plan-a8-finetune-a6.md`` section 4.

Proxy holdout
    For each real OOD intent, one training intent with the same capability,
    distraction and actuation, drawn with a fixed seed (:func:`pick_proxy`).
    During selection runs its train and val rows become split ``proxy``
    (scored as unseen intents, never trained on); its test rows are left
    alone, so the test split is only ever touched by the final model.

Leak guards (as ``autogate_bench.augment`` applies them to the real OOD set)
    A synthetic goal is dropped from training when any of its rows has token
    Jaccard > 0.5 (rules-baseline content tokens) with any row of a proxy
    intent, or when its name or description contains a proxy intent's key
    words (hand-written in the YAML, like ``augment.OOD_KEYWORDS``).
    Sensitive slot fills (names, numbers) are blanked before comparing, since
    A6 compares templates. A
    minimal pair that uses a proxy intent is dropped.

Probe pairs
    About 20% of the minimal pairs, whole intent pairs only, none touching a
    proxy intent, stratified by ``differs_in`` (:func:`pick_probe_pairs`).
    Their rows become split ``probe`` in every phase: never trained on,
    report-only.

:func:`apply` does all of this to a list of rows at run time, so the
derived lists follow the data (e.g. after the owner's audit of A6); its
report records what was moved or dropped.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT_PATH = REPO / "configs" / "decide" / "a8_holdout.yaml"
JACCARD_LIMIT = 0.5
PROBE_FRACTION = 0.2
ROUTING = ("capability", "distraction", "actuation")


@dataclass(frozen=True)
class Holdout:
    proxy: dict[str, str]  # real OOD intent -> proxy training intent
    keywords: dict[str, tuple[tuple[str, ...], ...]]  # proxy intent -> key-word entries
    probe_pairs: tuple[str, ...]
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def proxy_intents(self) -> frozenset[str]:
        return frozenset(self.proxy.values())

    @classmethod
    def load(cls, path: str | Path = DEFAULT_HOLDOUT_PATH) -> Holdout:
        import yaml

        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            proxy=dict(d["proxy"]),
            keywords={k: tuple(tuple(e) for e in v) for k, v in (d.get("keywords") or {}).items()},
            probe_pairs=tuple(d["probe_pairs"]),
            seed=int(d.get("seed", 0)),
            meta=dict(d.get("meta") or {}),
        )


def _intent_tags() -> dict[str, tuple[str, str, str]]:
    from autogate_bench.intents import INTENTS

    return {i.name: tuple(str(getattr(i, t)) for t in ROUTING) for i in INTENTS}


def _splits(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if not r.get("is_synthetic") and not r.get("pair_id"):
            out[r["intent"]].add(r["split"])
    return out


def pick_proxy(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> dict[str, str]:
    tags = _intent_tags()
    splits = _splits(rows)
    ood = sorted(i for i, s in splits.items() if s == {"ood"})
    train = sorted(i for i, s in splits.items() if "train" in s)
    rng = np.random.default_rng(seed)
    out = {}
    for o in ood:
        cands = [t for t in train if tags[t] == tags[o] and t not in out.values()]
        if not cands:
            raise ValueError(f"no training intent shares the tags of {o}: {tags[o]}")
        out[o] = cands[int(rng.integers(len(cands)))]
    return out


def _pair_intents(rows: Sequence[Mapping[str, Any]]) -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("pair_id"):
            out[r["pair_id"]].add(r["intent"])
    return {k: frozenset(v) for k, v in out.items()}


def pick_probe_pairs(
    rows: Sequence[Mapping[str, Any]],
    proxy_intents: frozenset[str],
    differs_in: Mapping[str, Sequence[str]],
    fraction: float = PROBE_FRACTION,
    seed: int = 0,
) -> tuple[str, ...]:
    """Whole intent pairs, none touching a proxy intent, about ``fraction`` of all pairs,
    with each ``differs_in`` stratum filled towards its share."""
    pairs = _pair_intents(rows)
    groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for pid, intents in pairs.items():
        groups[intents].append(pid)
    eligible = [g for g in groups if not g & proxy_intents]
    rng = np.random.default_rng(seed)
    order = [eligible[i] for i in rng.permutation(len(eligible))]
    key = {pid: tuple(sorted(differs_in.get(pid, ()))) for pid in pairs}
    total = Counter(key[p] for p in pairs)
    target = {k: fraction * n for k, n in total.items()}
    n_target = round(fraction * len(pairs))
    chosen: list[str] = []
    got: Counter = Counter()
    for g in order:
        if len(chosen) >= n_target:
            break
        add = Counter(key[p] for p in groups[g])
        if all(got[k] + add[k] <= target[k] + 2 for k in add):
            chosen += sorted(groups[g])
            got += add
    return tuple(sorted(chosen))


def _keyword_hits(text: str, keywords: Mapping[str, Sequence[Sequence[str]]]) -> list[str]:
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    return [
        f"{intent}: {' '.join(entry)}"
        for intent, entries in keywords.items()
        for entry in entries
        if all(any(w.startswith(k) for w in words) for k in entry)
    ]


def _unfilled(r: Mapping[str, Any]) -> str:
    """The utterance with its sensitive slot fills blanked: A6 compares templates, and a
    shared invented name or number is not a shared phrasing."""
    text = r["utterance"]
    for sp in sorted(r.get("spans") or [], key=lambda x: -x["start"]):
        text = text[: sp["start"]] + " " + text[sp["end"] :]
    return text


def leaking_goals(
    rows: Sequence[Mapping[str, Any]],
    holdout: Holdout,
    goal_text: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Synthetic goal -> why it leaks into the proxy holdout."""
    from autogate_bench.augment import _content_tokens, token_jaccard

    tok = _content_tokens()
    proxy = holdout.proxy_intents
    ref = [
        tok(_unfilled(r))
        for r in rows
        if r["intent"] in proxy and not r.get("is_synthetic") and not r.get("pair_id")
    ]
    ref = list({t for t in ref if t})
    out: dict[str, str] = {}
    for name, text in (goal_text or {}).items():
        hits = _keyword_hits(text, holdout.keywords)
        if hits:
            out[name] = f"key words: {', '.join(hits)}"
    for r in rows:
        if not r.get("is_synthetic") or r["intent"] in out:
            continue
        t = tok(_unfilled(r))
        best = max((token_jaccard(t, x) for x in ref), default=0.0)
        if best > JACCARD_LIMIT:
            out[r["intent"]] = f"token Jaccard {best:.2f} > {JACCARD_LIMIT}: {r['utterance']!r}"
    return out


def apply(
    rows: Sequence[Mapping[str, Any]],
    holdout: Holdout,
    goal_text: Mapping[str, str] | None = None,
    proxy: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows for an A8 run. ``proxy=True`` (selection runs): proxy-intent train and val rows
    become split ``proxy`` and leaking goals and proxy pairs are dropped. ``proxy=False``
    (the final model): only the probe pairs are held out. Probe-pair rows always become
    split ``probe``."""
    pairs = _pair_intents(rows)
    probe = set(holdout.probe_pairs)
    proxy_intents = holdout.proxy_intents
    drop_pairs = {p for p, i in pairs.items() if i & proxy_intents} if proxy else set()
    drop_goals = leaking_goals(rows, holdout, goal_text) if proxy else {}
    out, moved = [], Counter()
    for r in rows:
        pid = r.get("pair_id")
        if pid in probe:
            out.append({**r, "split": "probe"})
            moved["probe"] += 1
        elif pid in drop_pairs or (r.get("is_synthetic") and r["intent"] in drop_goals):
            moved["dropped"] += 1
        elif proxy and r["intent"] in proxy_intents and r["split"] in ("train", "val"):
            out.append({**r, "split": "proxy"})
            moved["proxy"] += 1
        else:
            out.append(dict(r))
    report = {
        "proxy_mode": proxy,
        "proxy_intents": sorted(proxy_intents),
        "rows_moved": dict(moved),
        "dropped_pairs": sorted(drop_pairs),
        "dropped_goals": drop_goals,
        "probe_pairs": len(probe & set(pairs)),
        "probe_pairs_missing": sorted(probe - set(pairs)),
    }
    return out, report


def goal_texts(augment_dir: str | Path | None = None) -> dict[str, str]:
    from autogate_bench.augment import DEFAULT_AUGMENT_DIR, GOALS_FILE, load_goals

    d = Path(augment_dir or DEFAULT_AUGMENT_DIR)
    return {
        g.name: f"{g.name.replace('_', ' ')} {g.description}" for g in load_goals(d / GOALS_FILE)
    }


def pair_differs(augment_dir: str | Path | None = None) -> dict[str, list[str]]:
    from autogate_bench.augment import DEFAULT_AUGMENT_DIR, PAIRS_FILE, load_pairs

    d = Path(augment_dir or DEFAULT_AUGMENT_DIR)
    return {p.id: list(p.differs_in) for p in load_pairs(d / PAIRS_FILE)}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from autogate_router.data import load_rows

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pick", help="print the proxy pick and probe pairs as YAML")
    pk.add_argument("--rows", type=Path, required=True, help="the augmented rows.jsonl")
    pk.add_argument("--seed", type=int, default=0)
    rp = sub.add_parser("report", help="what apply() moves and drops")
    rp.add_argument("--rows", type=Path, required=True)
    rp.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT_PATH)
    a = p.parse_args(argv)
    rows = load_rows(a.rows)
    if a.cmd == "pick":
        proxy = pick_proxy(rows, a.seed)
        probe = pick_probe_pairs(rows, frozenset(proxy.values()), pair_differs(), seed=a.seed)
        out = {"seed": a.seed, "proxy": proxy, "probe_pairs": list(probe)}
        print(json.dumps(out, indent=2))
        return out
    _, report = apply(rows, Holdout.load(a.holdout), goal_texts())
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
