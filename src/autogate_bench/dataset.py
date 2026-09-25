"""Benchmark rows: fill seeds, pair them with contexts, label, split, write.

One row is one (utterance, context) pair with its rulebook label. An
utterance is a seed, one of its paraphrases (``data/paraphrases``), or an
ASR-noised copy of one of its plain paraphrases; all three go through the
same filling, context sampling and labeling, and inherit the seed's split.

Utterance ids: a seed is ``{intent}/{i}``, its paraphrase ``k`` (position in
the seed's list in the file) is ``{intent}/{i}/p{k}``, and the ``n``-th ASR
copy of that paraphrase is ``{intent}/{i}/p{k}/a{n}``. A row id appends
``-{fill}-{context}``.

The pipeline is deterministic: every utterance draws from its own rng,
derived from the generation seed and the utterance id, so adding or editing
one seed or paraphrase never changes another utterance's rows, and the same
arguments always give a byte-identical ``rows.jsonl``. A seed-only run
writes no ``source`` / ``variant`` columns to ``rows.jsonl`` (they would be
constant), so its bytes are those of the original seed-only pipeline.
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

from autogate_bench.asr_noise import asr_noise
from autogate_bench.contexts import flip_profile, sample_contexts
from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.paraphrases import Paraphrase, Variant, normalize, validate
from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.rulebook import route
from autogate_bench.schema import (
    ActuationClass,
    Connectivity,
    Context,
    Intent,
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
SOURCES: tuple[str, ...] = ("seed", "paraphrase", "asr")
VARIANTS: tuple[str, ...] = tuple(str(v) for v in Variant)
SOURCE_FIELDS: tuple[str, ...] = ("source", "variant")
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
    # provenance: "seed" | "paraphrase" | "asr", and the Variant ("plain" for seeds)
    source: str = "seed"
    variant: str = "plain"

    def context(self) -> Context:
        return Context(
            SpeedBucket(self.speed_bucket),
            Connectivity(self.connectivity),
            Workload(self.driver_workload),
            PrivacyMode(self.privacy_mode),
            self.passenger_present,
            LocalModelTier(self.local_model_tier),
        )

    def to_dict(self, include_source: bool = True) -> dict[str, Any]:
        """The flat record. ``include_source=False`` leaves out ``source`` and ``variant``."""
        d = asdict(self)
        d["tokens"] = list(self.tokens)
        d["spans"] = [s.to_dict() for s in self.spans]
        d["bio"] = list(self.bio)
        if not include_source:
            for k in SOURCE_FIELDS:
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Row:
        """Inverse of ``to_dict``; a record without ``source``/``variant`` is a seed row."""
        kw = {f.name: d[f.name] for f in fields(cls) if f.name not in SOURCE_FIELDS}
        kw["source"] = d.get("source", "seed")
        kw["variant"] = d.get("variant", "plain")
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


def _utterance_rows(
    intent: Intent,
    seed: Seed,
    sid: str,
    uid: str,
    rng: random.Random,
    split: str,
    source: str,
    variant: str,
    contexts_per_utterance: int,
    n_sensitive: int,
    n_generic: int,
    policy: Policy,
) -> list[Row]:
    """Fill one utterance (a seed or a derived text in ``seed``), sample contexts, label."""
    rows: list[Row] = []
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
                    id=f"{uid}-{fi}-{ci}",
                    seed_id=sid,
                    intent=intent.name,
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
                    adversarial=is_adversarial(seed.style, intent.actuation, decision.route),
                    split=split,
                    source=source,
                    variant=variant,
                )
            )
    return rows


def asr_texts(
    plain: Sequence[tuple[int, Paraphrase]],
    rng: random.Random,
    n: int,
    avoid: Iterable[str] = (),
) -> list[tuple[int, int, str]]:
    """``n`` ASR copies of a seed's plain paraphrases, spread across them.

    ``plain`` is ``(paraphrase index, paraphrase)`` in file order. The order
    is shuffled once and walked round-robin, so ``n`` up to the number of
    plain paraphrases uses each at most once. A paraphrase with no
    applicable edit, or an output equal to a text already present (``avoid``
    or an earlier copy), is skipped. Returns ``(paraphrase index, copy
    number, text)``; the copy number counts copies of that paraphrase.
    """
    order = list(plain)
    rng.shuffle(order)
    seen = {normalize(t) for t in avoid}
    per_source: dict[int, int] = {}
    out: list[tuple[int, int, str]] = []
    attempts = 0
    while order and len(out) < n and attempts < n * len(order) * 3:
        k, p = order[attempts % len(order)]
        attempts += 1
        text = asr_noise(p.text, rng)
        if text is None or normalize(text) in seen:
            continue
        seen.add(normalize(text))
        a = per_source.get(k, 0)
        per_source[k] = a + 1
        out.append((k, a, text))
    return out


def build_rows(
    seeds: Mapping[str, Sequence[Seed]] | None = None,
    rng_seed: int = 0,
    contexts_per_utterance: int = 3,
    n_sensitive: int = 2,
    n_generic: int = 1,
    policy: Policy = DEFAULT_POLICY,
    ood_intents: Iterable[str] = OOD_INTENTS,
    paraphrases: Mapping[str, Mapping[int, Sequence[Paraphrase]]] | None = None,
    asr_per_seed: int = 4,
) -> list[Row]:
    """Generate every row, in seed-file order.

    Each seed's rows come first, then its paraphrases' in file order, then its
    ``asr_per_seed`` ASR copies. ``paraphrases`` (from
    ``autogate_bench.paraphrases.load_paraphrases``) is optional; without it
    the dataset is seed-only and ``asr_per_seed`` has no effect. Paraphrases
    must pass ``validate`` without errors (quota not enforced).
    """
    seeds = load_seeds() if seeds is None else seeds
    paraphrases = paraphrases or {}
    if paraphrases:
        report = validate(paraphrases, seeds, check_warnings=False)
        if report.errors:
            shown = "\n".join(e.format() for e in report.errors[:20])
            raise ValueError(f"{len(report.errors)} paraphrase errors:\n{shown}")
    ids = [seed_id(intent, i) for intent, items in seeds.items() for i in range(len(items))]
    splits = assign_splits(ids, random.Random(f"{rng_seed}/splits"), ood_intents)
    common = dict(
        contexts_per_utterance=contexts_per_utterance,
        n_sensitive=n_sensitive,
        n_generic=n_generic,
        policy=policy,
    )

    rows: list[Row] = []
    for intent_name, items in seeds.items():
        intent = INTENTS_BY_NAME[intent_name]
        by_seed = paraphrases.get(intent_name, {})
        for index, seed in enumerate(items):
            sid = seed_id(intent_name, index)
            split = splits[sid]
            rng = random.Random(f"{rng_seed}/{sid}")
            rows += _utterance_rows(
                intent, seed, sid, sid, rng, split, "seed", str(Variant.PLAIN), **common
            )
            para = list(enumerate(by_seed.get(index, ())))
            for k, p in para:
                uid = f"{sid}/p{k}"
                derived = Seed(intent_name, p.text, seed.style)
                rows += _utterance_rows(
                    intent,
                    derived,
                    sid,
                    uid,
                    random.Random(f"{rng_seed}/{uid}"),
                    split,
                    "paraphrase",
                    p.variant,
                    **common,
                )
            plain = [(k, p) for k, p in para if p.variant == Variant.PLAIN]
            if not plain or asr_per_seed <= 0:
                continue
            avoid = [seed.text, *(p.text for _, p in para)]
            asr_rng = random.Random(f"{rng_seed}/{sid}/asr")
            for k, a, text in asr_texts(plain, asr_rng, asr_per_seed, avoid):
                uid = f"{sid}/p{k}/a{a}"
                derived = Seed(intent_name, text, seed.style)
                rows += _utterance_rows(
                    intent,
                    derived,
                    sid,
                    uid,
                    random.Random(f"{rng_seed}/{uid}"),
                    split,
                    "asr",
                    str(Variant.ASR),
                    **common,
                )
    return rows


# --------------------------------------------------------------------------- #
# Writers and readers
# --------------------------------------------------------------------------- #


def write_jsonl(rows: Iterable[Row], path: str | Path) -> None:
    """One JSON record per line. ``source``/``variant`` are written only when
    some row is not a seed, so a seed-only file keeps its original bytes."""
    rows = list(rows)
    include = any(r.source != "seed" for r in rows)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(include_source=include), ensure_ascii=True) + "\n")


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


def utterance_key(row: Row) -> str:
    """The id of the filled utterance a row came from: its row id without the context index."""
    return row.id.rsplit("-", 1)[0]


def summarize(rows: Sequence[Row]) -> dict[str, Any]:
    """Counts used by the manifest and ``docs/pilot.md``."""
    utterances: dict[str, Row] = {}
    for r in rows:
        utterances.setdefault(utterance_key(r), r)
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
        "source": {s: sum(r.source == s for r in rows) for s in SOURCES},
        "variant": {v: sum(r.variant == v for r in rows) for v in VARIANTS},
        "utterances_by_source": {
            s: sum(u.source == s for u in utterances.values()) for s in SOURCES
        },
        "utterances_by_variant": {
            v: sum(u.variant == v for u in utterances.values()) for v in VARIANTS
        },
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
