"""Rows -> router training examples.

Input text
    ``context_prefix + " " + utterance``, or the bare utterance with
    ``use_context=False`` (the no-context ablation). The prefix is the
    bracketed context from ``autogate_bench.schema.Context.to_prefix``.

Tokens
    The text goes through a *fast* tokenizer with ``return_offsets_mapping``.
    A token belongs to the utterance if its character range, with leading
    and trailing whitespace trimmed, overlaps the utterance. Its offsets are
    then stored relative to the utterance (``offsets``; ``(-1, -1)`` for
    prefix, special and padding tokens), and ``utterance_mask`` is 1 on it.

Span labels (subword BIO)
    Built from the row's character-offset ``spans``, not from its word-level
    ``bio``. An utterance token that overlaps a span is ``B-<label>`` if it
    is the first token of that span and ``I-<label>`` otherwise, so the
    continuation subwords of a word inside a span are ``I-``. Utterance
    tokens outside every span are ``O``. Prefix, special and padding tokens
    get ``-100`` and are ignored by the loss. :func:`spans_from_tags` inverts
    this through the same offsets.

Class weights
    :func:`class_weights_from_cost_matrix` turns the cost matrix into route
    loss weights; the formula is in its docstring and ``docs/training.md``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from autogate_eval.cost_matrix import (
    DEFAULT_COST_MATRIX_PATH,
    LOCAL_OTHER,
    LOCAL_SAFETY_CRITICAL,
    CostMatrix,
)
from autogate_router.labels import DEFAULT_LABELS, IGNORE_INDEX, Labels

Row = Mapping[str, Any]
NO_OFFSET = (-1, -1)


# --------------------------------------------------------------------------- #
# Reading rows
# --------------------------------------------------------------------------- #


def _parse_splits(split: str | Iterable[str] | None) -> set[str] | None:
    if split is None:
        return None
    if isinstance(split, str):
        parts = [s.strip() for s in split.split(",") if s.strip()]
        return set(parts) if parts and parts != ["all"] else None
    return set(split)


def load_rows(
    path: str | Path,
    split: str | Iterable[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Rows as dicts from ``rows.jsonl`` or ``rows.parquet``.

    ``split`` keeps only those splits (a name, a comma list, or an iterable;
    ``None`` or ``"all"`` keeps everything). ``limit`` keeps a random sample
    of that many rows (drawn with ``seed``, file order preserved) so smoke
    runs still see many intents.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(str(path)).to_pylist()
    else:
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    keep = _parse_splits(split)
    if keep is not None:
        rows = [r for r in rows if r["split"] in keep]
    if limit is not None and 0 <= limit < len(rows):
        chosen = sorted(random.Random(seed).sample(range(len(rows)), limit))
        rows = [rows[i] for i in chosen]
    return rows


# --------------------------------------------------------------------------- #
# Tokenization and subword BIO
# --------------------------------------------------------------------------- #


def input_text(row: Row, use_context: bool = True) -> tuple[str, int]:
    """The model input and the character offset where the utterance starts."""
    if not use_context:
        return row["utterance"], 0
    prefix = row["context_prefix"]
    return f"{prefix} {row['utterance']}", len(prefix) + 1


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def utterance_offsets(
    text: str, offset_mapping: Sequence[Sequence[int]], utt_start: int
) -> list[tuple[int, int]]:
    """Per token: its whitespace-trimmed range relative to the utterance, or ``(-1, -1)``.

    Special tokens (empty range) and tokens entirely inside the prefix map
    to ``(-1, -1)``. A token straddling the prefix boundary is clipped.
    """
    out: list[tuple[int, int]] = []
    for a, b in offset_mapping:
        if b <= a:
            out.append(NO_OFFSET)
            continue
        ta, tb = _trim(text, a, b)
        if tb <= ta:  # whitespace-only token
            ta = tb = max(a, utt_start)
            if b <= utt_start:
                out.append(NO_OFFSET)
                continue
            out.append((ta - utt_start, tb - utt_start))
            continue
        if tb <= utt_start:
            out.append(NO_OFFSET)
            continue
        out.append((max(ta, utt_start) - utt_start, tb - utt_start))
    return out


def _span_of(a: int, b: int, spans: Sequence[Row]) -> Row | None:
    for s in spans:
        if a < s["end"] and b > s["start"]:
            return s
        if a == b and s["start"] < a < s["end"]:  # whitespace token inside a span
            return s
    return None


def subword_bio(
    offsets: Sequence[tuple[int, int]], spans: Sequence[Row], labels: Labels = DEFAULT_LABELS
) -> list[int]:
    """Tag ids per token from char spans; ``-100`` where the offset is ``(-1, -1)``."""
    out: list[int] = []
    open_span: Row | None = None
    for a, b in offsets:
        if (a, b) == NO_OFFSET:
            out.append(IGNORE_INDEX)
            continue
        hit = _span_of(a, b, spans)
        if hit is None:
            out.append(labels.bio_index["O"])
            open_span = None
        elif hit is open_span:
            out.append(labels.bio_index[f"I-{hit['label']}"])
        else:
            out.append(labels.bio_index[f"B-{hit['label']}"])
            open_span = hit
    return out


def spans_from_tags(
    utterance: str,
    offsets: Sequence[tuple[int, int]],
    tags: Sequence[str],
) -> list[dict[str, Any]]:
    """Merge subword B/I runs back into character spans of the utterance.

    ``tags`` are tag names aligned with ``offsets``; tokens at ``(-1, -1)``
    are skipped. An ``I-x`` continues an open span of label ``x`` (a
    whitespace-only token in between does not break it); any other ``I-x``
    starts a new span, so a model that forgets the ``B-`` still yields spans.
    """
    out: list[dict[str, Any]] = []
    cur: list[int] | None = None
    label = ""

    def close() -> None:
        if cur is not None:
            a, b = _trim(utterance, cur[0], cur[1])
            if b > a:
                out.append({"start": a, "end": b, "label": label, "text": utterance[a:b]})

    for (a, b), tag in zip(offsets, tags, strict=True):
        if (a, b) == NO_OFFSET:
            continue
        if tag.startswith("I-") and cur is not None and tag[2:] == label:
            cur[1] = max(cur[1], b)
            continue
        if tag == "O" and a == b and cur is not None:
            continue  # empty token never splits a span
        close()
        cur = None
        if tag.startswith(("B-", "I-")):
            cur, label = [a, b], tag[2:]
    close()
    return out


def encode_row(
    row: Row,
    tokenizer: Any,
    max_length: int = 96,
    use_context: bool = True,
    labels: Labels = DEFAULT_LABELS,
) -> dict[str, Any]:
    """One example as plain lists (no torch)."""
    text, utt_start = input_text(row, use_context)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = utterance_offsets(text, enc["offset_mapping"], utt_start)
    ex: dict[str, Any] = {
        "id": row.get("id"),
        "input_ids": list(enc["input_ids"]),
        "attention_mask": [1] * len(enc["input_ids"]),
        "utterance_mask": [0 if o == NO_OFFSET else 1 for o in offsets],
        "offsets": offsets,
        "span_labels": subword_bio(offsets, row.get("spans") or (), labels),
    }
    if row.get("route") is not None:
        ex["route_label"] = labels.route_index[row["route"]]
    if row.get("intent") is not None:
        ex["intent_label"] = labels.intent_index.get(row["intent"], IGNORE_INDEX)
    return ex


def build_examples(
    rows: Iterable[Row],
    tokenizer: Any,
    max_length: int = 96,
    use_context: bool = True,
    labels: Labels = DEFAULT_LABELS,
) -> list[dict[str, Any]]:
    """Encode every row; see :func:`encode_row`."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer is required (offset mapping)")
    return [encode_row(r, tokenizer, max_length, use_context, labels) for r in rows]


class RouterDataset:
    """A map-style dataset over encoded rows (usable with ``torch.utils.data.DataLoader``)."""

    def __init__(
        self,
        rows: Sequence[Row],
        tokenizer: Any,
        max_length: int = 96,
        use_context: bool = True,
        labels: Labels = DEFAULT_LABELS,
    ):
        self.rows = list(rows)
        self.examples = build_examples(self.rows, tokenizer, max_length, use_context, labels)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: Any,
        split: str | Iterable[str] | None = None,
        limit: int | None = None,
        seed: int = 0,
        **kw: Any,
    ) -> RouterDataset:
        return cls(load_rows(path, split, limit, seed), tokenizer, **kw)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.examples[i]


def collate(batch: Sequence[Mapping[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """Right-pad a list of examples into tensors (``span_labels`` padded with -100)."""
    import torch

    n = max(len(ex["input_ids"]) for ex in batch)

    def pad(key: str, value: int) -> torch.Tensor:
        return torch.tensor(
            [list(ex[key]) + [value] * (n - len(ex[key])) for ex in batch], dtype=torch.long
        )

    out: dict[str, Any] = {
        "input_ids": pad("input_ids", pad_token_id),
        "attention_mask": pad("attention_mask", 0),
        "utterance_mask": pad("utterance_mask", 0),
        "span_labels": pad("span_labels", IGNORE_INDEX),
    }
    if all("route_label" in ex for ex in batch):
        out["route_labels"] = torch.tensor([ex["route_label"] for ex in batch], dtype=torch.long)
    if all("intent_label" in ex for ex in batch):
        out["intent_labels"] = torch.tensor([ex["intent_label"] for ex in batch], dtype=torch.long)
    return out


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #


def class_weights_from_cost_matrix(
    path: str | Path | None = None, labels: Labels = DEFAULT_LABELS
) -> list[float]:
    """Route loss weights from the cost matrix, in ``labels.routes`` order.

    For a true route ``c`` the raw weight is the mean cost of getting it
    wrong, ``w_c = mean_{p != c} cost(c, p)`` over the four wrong
    predictions. The matrix splits the true-LOCAL row by whether the intent
    is safety-critical, which a per-class weight cannot express, so LOCAL
    takes the mean of its two rows' means. The weights are then divided by
    their mean so they average 1 and the loss scale matches plain
    cross-entropy.
    """
    cm = CostMatrix.load(path or DEFAULT_COST_MATRIX_PATH)

    def row_mean(row: str, correct: str) -> float:
        vals = [v for p, v in cm.costs[row].items() if str(p) != correct]
        return sum(vals) / len(vals)

    raw = []
    for r in labels.routes:
        if r == "LOCAL":
            raw.append((row_mean(LOCAL_SAFETY_CRITICAL, r) + row_mean(LOCAL_OTHER, r)) / 2)
        else:
            raw.append(row_mean(r, r))
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw]
