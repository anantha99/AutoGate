"""Benchmark rows: fill seeds, pair them with contexts, label, split, write.

One row is one (utterance, context) pair with its rulebook label. The
pipeline is deterministic: every seed draws from its own rng, derived from
the generation seed and the seed id, so adding or editing one seed never
changes another seed's rows, and the same arguments always give a
byte-identical ``rows.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from autogate_bench.contexts import flip_profile, sample_contexts
from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.rulebook import route
from autogate_bench.schema import (
    ActuationClass,
    Connectivity,
    Context,
    LocalModelTier,
    PrivacyMode,
    Route,
    SpeedBucket,
    Workload,
)
from autogate_bench.seeds import Seed, SeedStyle, load_seeds
from autogate_bench.spans import Span, bio_tags, fills_for_seed, tokens
from autogate_bench.splits import OOD_INTENTS, SPLITS, assign_splits, seed_id

SLICES: tuple[str, ...] = ("context_invariant", "context_flipped", "sensitive", "adversarial")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Row:
    """One labeled (utterance, context) pair. ``to_dict`` gives the flat record."""

    id: str
    seed_id: str
    intent: str
    group: str
    style: str
    utterance: str
    tokens: tuple[str, ...]
    spans: tuple[Span, ...]
    bio: tuple[str, ...]
    has_sensitive_spans: bool
    # context
    speed_bucket: str
    connectivity: str
    driver_workload: str
    privacy_mode: str
    passenger_present: bool
    local_model_tier: str
    context_prefix: str
    # intent tags
    capability: str
    distraction: str
    actuation: str
    consequence: str
    # label
    route: str
    reason: str
    demand: str
    # slices
    context_invariant: bool
    context_flipped: bool
    sensitive: bool
    adversarial: bool
    split: str

    def context(self) -> Context:
        return Context(
            SpeedBucket(self.speed_bucket),
            Connectivity(self.connectivity),
            Workload(self.driver_workload),
            PrivacyMode(self.privacy_mode),
            self.passenger_present,
            LocalModelTier(self.local_model_tier),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tokens"] = list(self.tokens)
        d["spans"] = [s.to_dict() for s in self.spans]
        d["bio"] = list(self.bio)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Row:
        kw = {f.name: d[f.name] for f in fields(cls)}
        kw["tokens"] = tuple(d["tokens"])
        kw["spans"] = tuple(Span.from_dict(s) for s in d["spans"])
        kw["bio"] = tuple(d["bio"])
        return cls(**kw)


def is_adversarial(style: str, actuation: str, route_: str) -> bool:
    """A polite or roundabout request that the label must still refuse.

    A question asking for a restricted actuation ("can you unlock the back
    door") is adversarial in every context; a question or indirect request
    is adversarial wherever its label is REFUSE.
    """
    if style == SeedStyle.QUESTION and actuation == ActuationClass.RESTRICTED:
        return True
    return style in (SeedStyle.QUESTION, SeedStyle.INDIRECT) and route_ == Route.REFUSE


def build_rows(
    seeds: Mapping[str, Sequence[Seed]] | None = None,
    rng_seed: int = 0,
    contexts_per_utterance: int = 3,
    n_sensitive: int = 2,
    n_generic: int = 1,
    policy: Policy = DEFAULT_POLICY,
    ood_intents: Iterable[str] = OOD_INTENTS,
) -> list[Row]:
    """Generate every row of the seed-only dataset, in seed-file order."""
    seeds = load_seeds() if seeds is None else seeds
    ids = [seed_id(intent, i) for intent, items in seeds.items() for i in range(len(items))]
    splits = assign_splits(ids, random.Random(f"{rng_seed}/splits"), ood_intents)

    rows: list[Row] = []
    for intent_name, items in seeds.items():
        intent = INTENTS_BY_NAME[intent_name]
        for index, seed in enumerate(items):
            sid = seed_id(intent_name, index)
            rng = random.Random(f"{rng_seed}/{sid}")
            for fi, filled in enumerate(fills_for_seed(seed, rng, n_sensitive, n_generic)):
                sens = filled.has_sensitive_spans
                invariant = len(flip_profile(intent, sens, policy)) == 1
                toks = tuple(tokens(filled.utterance))
                bio = tuple(bio_tags(filled.utterance, filled.spans))
                contexts = sample_contexts(intent, sens, rng, contexts_per_utterance, policy)
                for ci, ctx in enumerate(contexts):
                    decision = route(intent, ctx, sens, policy)
                    rows.append(
                        Row(
                            id=f"{sid}-{fi}-{ci}",
                            seed_id=sid,
                            intent=intent_name,
                            group=str(intent.group),
                            style=str(seed.style),
                            utterance=filled.utterance,
                            tokens=toks,
                            spans=filled.spans,
                            bio=bio,
                            has_sensitive_spans=sens,
                            speed_bucket=str(ctx.speed_bucket),
                            connectivity=str(ctx.connectivity),
                            driver_workload=str(ctx.driver_workload),
                            privacy_mode=str(ctx.privacy_mode),
                            passenger_present=ctx.passenger_present,
                            local_model_tier=str(ctx.local_model_tier),
                            context_prefix=ctx.to_prefix(),
                            capability=str(intent.capability),
                            distraction=str(intent.distraction),
                            actuation=str(intent.actuation),
                            consequence=str(intent.consequence),
                            route=str(decision.route),
                            reason=str(decision.reason),
                            demand=str(decision.demand),
                            context_invariant=invariant,
                            context_flipped=not invariant,
                            sensitive=sens,
                            adversarial=is_adversarial(
                                seed.style, intent.actuation, decision.route
                            ),
                            split=splits[sid],
                        )
                    )
    return rows


# --------------------------------------------------------------------------- #
# Writers and readers
# --------------------------------------------------------------------------- #


def write_jsonl(rows: Iterable[Row], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[Row]:
    with Path(path).open(encoding="utf-8") as f:
        return [Row.from_dict(json.loads(line)) for line in f if line.strip()]


def _arrow_schema():
    import pyarrow as pa

    span = pa.struct(
        [("start", pa.int64()), ("end", pa.int64()), ("label", pa.string()), ("text", pa.string())]
    )
    types = {
        "tokens": pa.list_(pa.string()),
        "spans": pa.list_(span),
        "bio": pa.list_(pa.string()),
    }
    out = []
    for f in fields(Row):
        if f.name in types:
            out.append((f.name, types[f.name]))
        elif f.type == "bool":
            out.append((f.name, pa.bool_()))
        else:
            out.append((f.name, pa.string()))
    return pa.schema(out)


def write_parquet(rows: Iterable[Row], path: str | Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([r.to_dict() for r in rows], schema=_arrow_schema())
    pq.write_table(table, str(path))


def read_parquet(path: str | Path) -> list[Row]:
    import pyarrow.parquet as pq

    return [Row.from_dict(d) for d in pq.read_table(str(path)).to_pylist()]


# --------------------------------------------------------------------------- #
# Summary and manifest
# --------------------------------------------------------------------------- #


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize(rows: Sequence[Row]) -> dict[str, Any]:
    """Counts used by the manifest and ``docs/pilot.md``."""
    utterances: dict[tuple[str, str], Row] = {}
    for r in rows:
        utterances.setdefault((r.seed_id, r.id.rsplit("-", 2)[1]), r)
    profiles = Counter(
        "+".join(sorted(flip_profile(u.intent, u.has_sensitive_spans))) for u in utterances.values()
    )
    n = len(rows)
    return {
        "n_rows": n,
        "n_seeds": len({r.seed_id for r in rows}),
        "n_utterances": len(utterances),
        "sensitive_rate": round(sum(r.has_sensitive_spans for r in rows) / n, 4) if n else 0.0,
        "sensitive_utterance_rate": round(
            sum(u.has_sensitive_spans for u in utterances.values()) / len(utterances), 4
        )
        if utterances
        else 0.0,
        "split": {s: sum(r.split == s for r in rows) for s in SPLITS},
        "seeds_per_split": {s: len({r.seed_id for r in rows if r.split == s}) for s in SPLITS},
        "route": {str(x): sum(r.route == x for r in rows) for x in Route},
        "reason": _count(r.reason for r in rows),
        "group": _count(r.group for r in rows),
        "style": _count(r.style for r in rows),
        "slice": {s: sum(getattr(r, s) for r in rows) for s in SLICES},
        "split_route": {
            s: {str(x): sum(r.split == s and r.route == x for r in rows) for x in Route}
            for s in SPLITS
        },
        "utterances": {
            "invariant": sum(u.context_invariant for u in utterances.values()),
            "flipped": sum(u.context_flipped for u in utterances.values()),
            "flip_profiles": dict(sorted(profiles.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
    }


def git_commit(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return p.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else None}


def write_manifest(
    rows: Sequence[Row], path: str | Path, args: Mapping[str, Any], rows_path: Path
) -> dict[str, Any]:
    manifest = {
        "generator": "autogate_bench.generate",
        "args": dict(args),
        "git": git_commit(),
        "rows_jsonl_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        **summarize(rows),
    }
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
