"""Rules baseline: keyword intent matcher + regex PII detector + the rulebook.

The route is ``rulebook.route(matched intent, row context, detected
sensitivity)``, so every error comes from the two front-end guesses, never
from the policy.

Intent matcher
    Each seed becomes a set of content tokens: lowercase, punctuation
    stripped, ``{slot}`` placeholders and a small stopword list dropped,
    anything with a digit normalised to ``<num>``. An utterance gets the
    intent of the seed with the highest ``|A & B| / |B|`` (A the utterance
    tokens, B the seed tokens), ties broken by ``|A & B|`` and then by seed
    order. On the seed-only pilot the matcher has seen every seed, including
    val, test and OOD ones, so its intent accuracy there is a lexical upper
    bound; ``--train-only`` builds it from train seeds only.

PII detector
    Regexes for a phone-like run of ten digits or a ``+91`` prefix, a
    standalone 4 to 6 digit number (OTP, card digits, PIN code), "card
    ending" or "ending <digits>", and any capitalised word that does not
    start a sentence. Utterances are lowercase ASR-style text, so the name
    rule never fires and contact names, relation words aside, go
    undetected; street addresses without a PIN code are missed too. That is
    the known weakness of rules that a learned span head should beat.

``--emit predictions.jsonl`` writes the baseline's predictions in the schema
``autogate_eval.score`` reads: the route, the matched intent and the regex
spans (:func:`detect_spans`), with ``route_probs``, ``intent_probs_top3``
and ``confidence`` null because the rules have no probabilities.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.rulebook import route
from autogate_bench.schema import (
    Connectivity,
    Context,
    LocalModelTier,
    PrivacyMode,
    Route,
    SpeedBucket,
    Workload,
)
from autogate_bench.seeds import Seed, load_seeds
from autogate_bench.splits import seed_id
from autogate_eval.metrics import format_report, report

STOPWORDS: frozenset[str] = frozenset(
    """a an the to me my i you your can could please for of on in at it its is are am be
    and or so this that from with just do does will would there here we us our im ill ive
    id dont some""".split()
)

_SLOT = re.compile(r"\{[^{}]*\}")
_NON_WORD = re.compile(r"[^a-z0-9<>\s]")

_PHONE = re.compile(r"\+91|(?<!\d)(?:\d[\s-]?){9}\d(?!\d)")
_SHORT_NUMBER = re.compile(r"(?<![\d.:])\d{4,6}(?![\d.:])")
_CARD = re.compile(r"\bcard ending\b|\bending\s+\d+", re.IGNORECASE)


def content_tokens(text: str) -> frozenset[str]:
    """Lowercased content tokens with slots and stopwords removed, digits as ``<num>``."""
    text = _SLOT.sub(" ", text.lower()).replace("'", "")
    text = _NON_WORD.sub(" ", text)
    out = set()
    for tok in text.split():
        if any(ch.isdigit() for ch in tok):
            out.add("<num>")
        elif tok not in STOPWORDS:
            out.add(tok)
    return frozenset(out)


class IntentMatcher:
    """Best token-overlap seed, over seeds in a fixed order."""

    def __init__(self, seeds: Iterable[Seed]):
        self.entries = [(s.intent, content_tokens(s.text)) for s in seeds]
        self.entries = [(i, b) for i, b in self.entries if b]
        if not self.entries:
            raise ValueError("no seed has any content tokens")

    def match(self, utterance: str) -> str:
        a = content_tokens(utterance)
        best_key, best = None, self.entries[0][0]
        for order, (intent, b) in enumerate(self.entries):
            inter = len(a & b)
            key = (inter / len(b), inter, -order)
            if best_key is None or key > best_key:
                best_key, best = key, intent
        return best


def detect_sensitive(utterance: str) -> bool:
    """Regex guess at ``has_sensitive_spans`` (misses lowercase names by design)."""
    if _PHONE.search(utterance) or _SHORT_NUMBER.search(utterance) or _CARD.search(utterance):
        return True
    words = utterance.split()
    for prev, word in zip(words, words[1:], strict=False):
        if word[:1].isupper() and word != "I" and not prev.endswith((".", "!", "?")):
            return True
    return False


_CARD_DIGITS = re.compile(r"\bending\s+(\d+)", re.IGNORECASE)
_CAPITALISED = re.compile(r"\S+")


def detect_spans(utterance: str) -> list[dict[str, Any]]:
    """Character spans for what the regexes in :func:`detect_sensitive` found.

    Labels are a guess from the pattern: a phone-like run (merged with an
    adjacent ``+91``) is ``phone``, the digits after "ending" are ``card``,
    any other standalone 4 to 6 digit number is ``otp``, and a capitalised
    word that does not start a sentence is ``contact``. "card ending" with
    no digits after it makes :func:`detect_sensitive` fire but has nothing
    to mask, so it gives no span. Overlapping matches keep the earlier
    pattern in that order.
    """
    found: list[tuple[int, int, str]] = []

    def add(a: int, b: int, label: str) -> None:
        if all(b <= x or a >= y for x, y, _ in found):
            found.append((a, b, label))

    phones = [[m.start(), m.end()] for m in _PHONE.finditer(utterance)]
    merged: list[list[int]] = []
    for a, b in phones:
        if merged and not utterance[merged[-1][1] : a].strip():
            merged[-1][1] = b
        else:
            merged.append([a, b])
    for a, b in merged:
        add(a, b, "phone")
    for m in _CARD_DIGITS.finditer(utterance):
        add(m.start(1), m.end(1), "card")
    for m in _SHORT_NUMBER.finditer(utterance):
        add(m.start(), m.end(), "otp")
    words = list(_CAPITALISED.finditer(utterance))
    for prev, word in zip(words, words[1:], strict=False):
        w = word.group()
        if w[:1].isupper() and w != "I" and not prev.group().endswith((".", "!", "?")):
            add(word.start(), word.end(), "contact")
    found.sort()
    return [{"start": a, "end": b, "label": lab, "text": utterance[a:b]} for a, b, lab in found]


def _get(row: Any, name: str) -> Any:
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def row_context(row: Any) -> Context:
    return Context(
        SpeedBucket(_get(row, "speed_bucket")),
        Connectivity(_get(row, "connectivity")),
        Workload(_get(row, "driver_workload")),
        PrivacyMode(_get(row, "privacy_mode")),
        bool(_get(row, "passenger_present")),
        LocalModelTier(_get(row, "local_model_tier")),
    )


class RulesBaseline:
    def __init__(self, seeds: Iterable[Seed] | None = None):
        if seeds is None:
            seeds = [s for items in load_seeds().values() for s in items]
        self.matcher = IntentMatcher(seeds)

    def predict_intent(self, row: Any) -> str:
        return self.matcher.match(_get(row, "utterance"))

    def predict(self, row: Any) -> Route:
        intent = INTENTS_BY_NAME[self.predict_intent(row)]
        sensitive = detect_sensitive(_get(row, "utterance"))
        return route(intent, row_context(row), sensitive).route


_DEFAULT: RulesBaseline | None = None


def predict(row: Any) -> Route:
    """Predict with a baseline built from every seed in ``data/seeds.yaml``."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RulesBaseline()
    return _DEFAULT.predict(row)


def _rate(hits: Sequence[bool]) -> float:
    return sum(hits) / len(hits) if hits else float("nan")


def evaluate_baseline(rows: Sequence[Any], baseline: RulesBaseline, n_boot: int = 1000) -> dict:
    intents = [baseline.predict_intent(r) for r in rows]
    preds = [baseline.predict(r) for r in rows]
    detected = [detect_sensitive(_get(r, "utterance")) for r in rows]
    truth = [bool(_get(r, "has_sensitive_spans")) for r in rows]
    splits = sorted({_get(r, "split") for r in rows}, key=["train", "val", "test", "ood"].index)
    correct = [i == _get(r, "intent") for i, r in zip(intents, rows, strict=True)]
    return {
        "intent_accuracy": _rate(correct),
        "intent_accuracy_by_split": {
            s: _rate([c for c, r in zip(correct, rows, strict=True) if _get(r, "split") == s])
            for s in splits
        },
        "pii_detection": {
            "recall": _rate([d for d, t in zip(detected, truth, strict=True) if t]),
            "precision": _rate([t for d, t in zip(detected, truth, strict=True) if d]),
        },
        "report": report(rows, preds, n_boot=n_boot),
    }


def format_baseline(result: Mapping[str, Any]) -> str:
    by_split = ", ".join(f"{s} {v:.3f}" for s, v in result["intent_accuracy_by_split"].items())
    pii = result["pii_detection"]
    return "\n".join(
        [
            f"intent accuracy: {result['intent_accuracy']:.3f} ({by_split})",
            f"PII detection: recall {pii['recall']:.3f}, precision {pii['precision']:.3f}",
            "",
            format_report(result["report"]),
        ]
    )


def emit_predictions(rows: Sequence[Any], baseline: RulesBaseline, path: Path) -> Path:
    """Write predictions in the ``autogate_eval.score`` schema, one line per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            rec = {
                "id": _get(r, "id"),
                "route": str(baseline.predict(r)),
                "route_probs": None,
                "intent": baseline.predict_intent(r),
                "intent_probs_top3": None,
                "spans": detect_spans(_get(r, "utterance")),
                "confidence": None,
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Score the rules baseline on generated rows.")
    p.add_argument("--rows", type=Path, required=True, help="rows.jsonl")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="build the intent matcher from train seeds only (default: every seed)",
    )
    p.add_argument("--json", type=Path, default=None, help="also write the full result here")
    p.add_argument(
        "--emit",
        type=Path,
        default=None,
        help="also write predictions.jsonl for autogate_eval.score",
    )
    a = p.parse_args(argv)

    with a.rows.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    seeds = load_seeds()
    if a.train_only:
        train = {r["seed_id"] for r in rows if r["split"] == "train"}
        chosen = [
            s for n, items in seeds.items() for i, s in enumerate(items) if seed_id(n, i) in train
        ]
    else:
        chosen = [s for items in seeds.values() for s in items]
    baseline = RulesBaseline(chosen)
    result = evaluate_baseline(rows, baseline)
    print(format_baseline(result))
    if a.emit:
        emit_predictions(rows, baseline, a.emit)
        print(f"wrote {len(rows)} predictions to {a.emit}")
    if a.json:
        a.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
