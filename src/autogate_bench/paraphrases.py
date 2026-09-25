"""Paraphrases of the seeds: file format, loader, validator and CLI.

Each seed in ``data/seeds.yaml`` is expanded into 40 paraphrases written by
hand (by generation agents following ``docs/paraphrase-spec.md``) or by
``autogate_bench.paraphrase_api``. They live in one file per intent::

    data/paraphrases/<intent>.yaml

    # header comment (free text)
    "0":                                   # seed index in data/seeds.yaml, as a string
      - {text: "switch the headlights on", variant: plain}
      - {text: "headlight chalu karo", variant: hinglish}
    "1":
      - ...

A paraphrase inherits everything from its seed: intent, style, split and the
exact set of ``{slot}`` placeholders, which the span injector fills after
paraphrasing. ``variant`` is one of ``plain``, ``hinglish``,
``kannada_english`` or ``disfluent``. The fifth variant, ``asr``, is made by
``autogate_bench.asr_noise`` at generation time and is never written in a
file.

CLI::

    python -m autogate_bench.paraphrases validate [--dir data/paraphrases] [--strict]
    python -m autogate_bench.paraphrases status [--dir data/paraphrases]

``validate`` prints counts per intent and variant and every error and warning
as ``file:seed:line``; it exits non-zero on any error. ``status`` shows which
intents have files and how far each is from its quota.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from autogate_bench.intents import INTENTS
from autogate_bench.seeds import _PLACEHOLDER, DEFAULT_SEEDS_PATH, Seed, SeedStyle, load_seeds

DEFAULT_PARAPHRASES_DIR = Path(__file__).resolve().parents[2] / "data" / "paraphrases"


class Variant(StrEnum):
    """What kind of rewording a paraphrase is."""

    PLAIN = "plain"  # natural Indian-English driver speech
    HINGLISH = "hinglish"  # Hindi-English code-mixing, Latin script
    KANNADA_ENGLISH = "kannada_english"  # Kannada-English code-mixing, Latin script
    DISFLUENT = "disfluent"  # fillers, restarts, self-corrections
    ASR = "asr"  # produced by autogate_bench.asr_noise, never written by hand


WRITTEN_VARIANTS: tuple[Variant, ...] = (
    Variant.PLAIN,
    Variant.HINGLISH,
    Variant.KANNADA_ENGLISH,
    Variant.DISFLUENT,
)

QUOTA: dict[Variant, int] = {
    Variant.PLAIN: 24,
    Variant.HINGLISH: 6,
    Variant.KANNADA_ENGLISH: 4,
    Variant.DISFLUENT: 6,
}
"""Paraphrases per seed, per variant (40 in total)."""

PER_SEED = sum(QUOTA.values())
MAX_WORDS = 24
NEAR_DUPLICATE = 0.6  # word-set Jaccard above which two lines are near-duplicates
MIN_FIRST_WORDS = 4  # distinct first words wanted across a seed's plain paraphrases
DRIFT_MIN_SHARED = 2  # content tokens shared with another intent's seed before drift is flagged
FRAGMENT_MAX_WORDS = 6  # a fragment-style paraphrase longer than this is flagged
FRAGMENT_MAX_WORDS_DISFLUENT = 9

# The synthetic first names of autogate_bench.spans.NAMES. A name is always
# written as {contact}; one of these in a paraphrase is a leaked literal name.
POOL_FIRST_NAMES: frozenset[str] = frozenset(
    """anil kavya ravi deepa suresh meghana karthik divya arun lakshmi senthil priya vignesh
    rahul neha amit pooja vikas sunita rohit nikhil anjali vishnu sreeja joseph arjun reshma
    sourav moumita arindam tanushree debashis riya harpreet gurpreet manjeet simran jaspreet
    navjot farah""".split()
)

# English openers of a spoken request-as-question, after any leading fillers
_QUESTION_OPENERS = re.compile(
    r"^(?:(?:uh|um|er|so|hey|okay|ok)\s+)*"
    r"(?:(?:can|could|would|will|shall)\s+(?:you|we|i)|is\s+there|are\s+there|do\s+you)\b"
)
_ALLOWED_CHARS = re.compile(r"[a-z0-9' ]*")
_DIGIT_RUN = re.compile(r"\d{4,}")
_ANY_DIGIT = re.compile(r"\d")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Paraphrase:
    """One paraphrase of one seed. ``variant`` is kept as written so the
    validator can report an unknown one; ``line`` is the 1-based line in the
    file (0 when not loaded from a file) and takes no part in equality."""

    intent: str
    seed_index: int
    text: str
    variant: str
    line: int = field(default=0, compare=False, repr=False)

    @property
    def slots(self) -> tuple[str, ...]:
        """Slot names in the text, in order of appearance (repeats kept)."""
        return tuple(m.group(1) for m in _PLACEHOLDER.finditer(self.text))


Paraphrases = dict[str, dict[int, tuple[Paraphrase, ...]]]


class _LineLoader(yaml.SafeLoader):
    """A safe loader that records each mapping's line and rejects duplicate keys."""

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ValueError(f"line {key_node.start_mark.line + 1}: duplicate key {key!r}")
            seen.add(key)
        mapping = super().construct_mapping(node, deep=deep)
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


def load_paraphrase_file(path: str | Path) -> dict[int, tuple[Paraphrase, ...]]:
    """Parse one ``<intent>.yaml`` file into ``{seed index: paraphrases}``.

    Raises ``ValueError`` on malformed structure only (not a mapping of seed
    index to a list of ``{text, variant}`` mappings). Content problems, such
    as an unknown variant or a wrong slot, are left to :func:`validate`.
    """
    path = Path(path)
    intent = path.stem
    with path.open(encoding="utf-8") as f:
        try:
            raw = yaml.load(f, Loader=_LineLoader)  # a SafeLoader subclass
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of seed index to a list of paraphrases")
    raw.pop("__line__", None)
    out: dict[int, tuple[Paraphrase, ...]] = {}
    for key, items in raw.items():
        key_s = str(key)
        if isinstance(key, bool) or not key_s.isdigit():
            raise ValueError(f'{path}: seed key {key!r} is not a seed index like "0"')
        index = int(key_s)
        if index in out:
            raise ValueError(f"{path}: seed {index} appears twice")
        if not isinstance(items, list):
            raise ValueError(f"{path}: seed {index}: expected a list of paraphrases")
        parsed = []
        for i, item in enumerate(items):
            where = f"{path}: seed {index} item {i}"
            if not isinstance(item, dict):
                raise ValueError(f"{where}: expected {{text: ..., variant: ...}}")
            line = item.pop("__line__", 0)
            if set(item) != {"text", "variant"}:
                raise ValueError(f"{where} (line {line}): expected exactly 'text' and 'variant'")
            text, variant = item["text"], item["variant"]
            if not isinstance(text, str) or not isinstance(variant, str):
                raise ValueError(f"{where} (line {line}): 'text' and 'variant' must be strings")
            parsed.append(Paraphrase(intent, index, text, variant, line))
        out[index] = tuple(parsed)
    return dict(sorted(out.items()))


def load_paraphrases(directory: str | Path = DEFAULT_PARAPHRASES_DIR) -> Paraphrases:
    """Load every ``<intent>.yaml`` in ``directory``: ``{intent: {seed index: paraphrases}}``.

    A missing directory gives an empty mapping. Files are read in name order.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {p.stem: load_paraphrase_file(p) for p in sorted(directory.glob("*.yaml"))}


def format_paraphrase_file(
    intent: str,
    by_seed: Mapping[int, Sequence[Paraphrase]],
    seeds: Sequence[Seed] | None = None,
) -> str:
    """Render the canonical file text: header, one quoted key per seed, one flow item per line."""
    lines = [
        f"# Paraphrases of the {intent} seeds in data/seeds.yaml.",
        "# Format and rules: docs/paraphrase-spec.md. One key per seed index (as a string),",
        "# each a list of {text, variant}; variant is plain, hinglish, kannada_english or",
        "# disfluent. Check with: python -m autogate_bench.paraphrases validate --strict",
    ]
    for index in sorted(by_seed):
        lines.append("")
        if seeds is not None and index < len(seeds):
            s = seeds[index]
            lines.append(f"# seed {index} ({s.style}): {s.text}")
        lines.append(f'"{index}":')
        for p in by_seed[index]:
            lines.append(f"  - {{text: {json.dumps(p.text)}, variant: {p.variant}}}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def normalize(text: str) -> str:
    """Comparison key: lowercase, apostrophes and a trailing ``?`` dropped, whitespace collapsed."""
    t = text.lower().replace("’", "'").replace("‘", "'").replace("'", "")
    t = " ".join(t.split())
    return t.removesuffix("?").rstrip()


def word_set(text: str) -> frozenset[str]:
    """The words of the normalised text, slots kept as ``{slot}`` tokens."""
    return frozenset(normalize(text).split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _asks(text: str) -> bool:
    """Whether the line opens like a request asked as a question ("can you ...")."""
    return bool(_QUESTION_OPENERS.match(text))


def _strip_slots(text: str) -> str:
    return _PLACEHOLDER.sub(" ", text)


def _format_errors(p: Paraphrase, seed: Seed) -> list[str]:
    """Hard per-line rule violations, independent of every other line."""
    errs: list[str] = []
    text = p.text
    if not text.strip():
        return ["empty text"]
    bare = _strip_slots(text)
    if "{" in bare or "}" in bare:
        errs.append("unbalanced brace")
    if set(p.slots) != set(seed.slots):
        want = ", ".join(sorted(set(seed.slots))) or "none"
        got = ", ".join(sorted(set(p.slots))) or "none"
        errs.append(f"slots {{{got}}} differ from the seed's {{{want}}}")
    n_words = len(text.split())
    if n_words > MAX_WORDS:
        errs.append(f"{n_words} words (max {MAX_WORDS})")
    if not text.isascii():
        errs.append("non-ASCII character")
    if text != text.lower():
        errs.append("uppercase letter")
    if text != text.strip() or "  " in text:
        errs.append("leading, trailing or double space")
    body = bare.removesuffix("?")
    if not _ALLOWED_CHARS.fullmatch(body.lower()) and text.isascii():
        bad = sorted({c for c in body.lower() if not _ALLOWED_CHARS.fullmatch(c)})
        errs.append(f"disallowed punctuation {''.join(bad)!r}")
    if text.endswith(" ?"):
        errs.append("space before the question mark")
    if _DIGIT_RUN.search(bare):
        errs.append("a literal number of 4+ digits (use a slot)")
    names = sorted(set(normalize(bare).split()) & POOL_FIRST_NAMES)
    if names:
        errs.append(f"literal name {', '.join(names)} (write {{contact}})")
    return errs


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Issue:
    """One error or warning, located by intent, seed index and file line."""

    level: str  # "error" or "warning"
    intent: str
    seed_index: int | None
    line: int
    message: str

    def format(self, directory: str | Path | None = None) -> str:
        file = (
            f"{self.intent}.yaml" if directory is None else f"{Path(directory)}/{self.intent}.yaml"
        )
        seed = "-" if self.seed_index is None else str(self.seed_index)
        return f"{file}:{seed}:{self.line}: {self.level}: {self.message}"


@dataclass
class ValidationReport:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    counts: dict[str, Counter] = field(default_factory=dict)  # intent -> variant counts

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, intent: str, seed: int | None, line: int, msg: str) -> None:
        self.errors.append(Issue("error", intent, seed, line, msg))

    def warn(self, intent: str, seed: int | None, line: int, msg: str) -> None:
        self.warnings.append(Issue("warning", intent, seed, line, msg))

    def only(self, intents: Iterable[str]) -> ValidationReport:
        keep = set(intents)
        return ValidationReport(
            [i for i in self.errors if i.intent in keep],
            [i for i in self.warnings if i.intent in keep],
            {k: v for k, v in self.counts.items() if k in keep},
        )


def quota_shortfall(items: Iterable[Paraphrase]) -> list[str]:
    """``["plain 22/24", ...]`` for each variant whose count differs from :data:`QUOTA`."""
    got = Counter(p.variant for p in items)
    return [f"{v} {got.get(v, 0)}/{q}" for v, q in QUOTA.items() if got.get(v, 0) != q]


def _content_tokens() -> Callable[[str], frozenset[str]]:
    # Imported lazily: autogate_eval depends on autogate_bench, not the reverse.
    from autogate_eval.baselines.rules import content_tokens

    return content_tokens


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """The rules baseline's match score: the share of the seed's content tokens present."""
    return len(a & b) / len(b) if b else 0.0


def validate(
    paraphrases: Mapping[str, Mapping[int, Sequence[Paraphrase]]],
    seeds: Mapping[str, Sequence[Seed]],
    strict: bool = False,
    check_warnings: bool = True,
) -> ValidationReport:
    """Check paraphrases against their seeds.

    Errors (the data is wrong): unknown intent or seed index; unknown variant
    or ``asr``; slot set differs from the seed's; more than 24 words;
    non-ASCII; uppercase; punctuation other than apostrophes and one trailing
    ``?``; a duplicate within the intent (after :func:`normalize`); identical
    to any seed of any intent or to a paraphrase of another intent; a digit
    run of 4 or more, or a synthetic-pool first name (a literal value where a
    slot belongs); with ``strict``, a seed whose per-variant counts differ
    from :data:`QUOTA`.

    Warnings (worth a human look): possible intent drift, meaning the line
    shares at least two content tokens with a seed of another intent and
    matches that seed better than any seed of its own intent, using the
    rules baseline's tokenizer and score (``|A & B| / |B|``, B the seed's
    content tokens); word-set Jaccard above 0.6 with the seed or
    with another paraphrase of the same seed (near-duplicate); fewer than 4
    distinct first words across a seed's plain paraphrases (low shape
    diversity); a style slip (a ``?`` or a question opener such as "can
    you" on a seed that is not a question, a long fragment, a literal
    digit).
    """
    rep = ValidationReport()
    seed_text_owner: dict[str, str] = {}
    for intent, items in seeds.items():
        for s in items:
            seed_text_owner.setdefault(normalize(s.text), intent)

    # which intents use each normalised paraphrase text
    text_intents: dict[str, set[str]] = {}
    for intent, by_seed in paraphrases.items():
        for items in by_seed.values():
            for p in items:
                text_intents.setdefault(normalize(p.text), set()).add(intent)

    ct = _content_tokens() if check_warnings else None
    seed_tokens: dict[str, list[frozenset[str]]] = {}
    if ct is not None:
        seed_tokens = {i: [ct(s.text) for s in items] for i, items in seeds.items()}

    for intent, by_seed in paraphrases.items():
        counts: Counter = Counter()
        rep.counts[intent] = counts
        if intent not in seeds:
            rep.error(intent, None, 0, f"unknown intent {intent!r} (no seeds for it)")
            continue
        intent_seeds = seeds[intent]
        seen: dict[str, int] = {}  # normalised text -> first line
        for index, items in by_seed.items():
            if not 0 <= index < len(intent_seeds):
                rep.error(
                    intent, index, 0, f"unknown seed index {index} ({len(intent_seeds)} seeds)"
                )
                continue
            seed = intent_seeds[index]
            for p in items:
                counts[p.variant] += 1
                if p.variant == Variant.ASR:
                    rep.error(intent, index, p.line, "variant asr is made by code, not written")
                elif p.variant not in WRITTEN_VARIANTS:
                    rep.error(intent, index, p.line, f"unknown variant {p.variant!r}")
                for msg in _format_errors(p, seed):
                    rep.error(intent, index, p.line, f"{msg}: {p.text!r}")
                key = normalize(p.text)
                if key in seen:
                    rep.error(intent, index, p.line, f"duplicate of line {seen[key]}: {p.text!r}")
                else:
                    seen[key] = p.line
                if key in seed_text_owner:
                    rep.error(
                        intent,
                        index,
                        p.line,
                        f"identical to a seed of {seed_text_owner[key]}: {p.text!r}",
                    )
                others = text_intents.get(key, set()) - {intent}
                if others:
                    rep.error(
                        intent,
                        index,
                        p.line,
                        f"identical to a paraphrase of {', '.join(sorted(others))}: {p.text!r}",
                    )
            if check_warnings:
                _warn_seed(rep, intent, index, seed, items, seeds, seed_tokens, ct)
        if strict:
            for index in range(len(intent_seeds)):
                short = quota_shortfall(by_seed.get(index, ()))
                if short:
                    line = by_seed[index][0].line if by_seed.get(index) else 0
                    rep.error(intent, index, line, f"quota not met: {', '.join(short)}")
    return rep


def _warn_seed(
    rep: ValidationReport,
    intent: str,
    index: int,
    seed: Seed,
    items: Sequence[Paraphrase],
    seeds: Mapping[str, Sequence[Seed]],
    seed_tokens: Mapping[str, list[frozenset[str]]],
    ct: Callable[[str], frozenset[str]] | None,
) -> None:
    seed_words = word_set(seed.text)
    words = [word_set(p.text) for p in items]
    for i, p in enumerate(items):
        j = jaccard(words[i], seed_words)
        if j > NEAR_DUPLICATE:
            rep.warn(intent, index, p.line, f"near-duplicate of the seed ({j:.2f}): {p.text!r}")
        for k in range(i):
            j = jaccard(words[i], words[k])
            if j > NEAR_DUPLICATE and normalize(p.text) != normalize(items[k].text):
                rep.warn(
                    intent,
                    index,
                    p.line,
                    f"near-duplicate of line {items[k].line} ({j:.2f}): {p.text!r}",
                )
        # style slips
        if seed.style is not SeedStyle.QUESTION:
            if p.text.endswith("?"):
                rep.warn(intent, index, p.line, f"'?' on a {seed.style} seed: {p.text!r}")
            elif _asks(p.text):
                rep.warn(
                    intent, index, p.line, f"reads as a question on a {seed.style} seed: {p.text!r}"
                )
        if seed.style is SeedStyle.FRAGMENT:
            limit = (
                FRAGMENT_MAX_WORDS_DISFLUENT
                if p.variant == Variant.DISFLUENT
                else FRAGMENT_MAX_WORDS
            )
            if len(p.text.split()) > limit:
                rep.warn(
                    intent, index, p.line, f"long for a fragment (> {limit} words): {p.text!r}"
                )
        if _ANY_DIGIT.search(_strip_slots(p.text)) and not _DIGIT_RUN.search(p.text):
            rep.warn(intent, index, p.line, f"literal digit (spell numbers out): {p.text!r}")
        # intent drift under the rules baseline's tokenizer
        if ct is not None:
            a = ct(p.text)
            if a:
                own = max((_overlap(a, b) for b in seed_tokens.get(intent, [])), default=0.0)
                best, best_intent, best_text = own, None, ""
                for other, toks in seed_tokens.items():
                    if other == intent:
                        continue
                    for s, b in zip(seeds[other], toks, strict=True):
                        score = _overlap(a, b)
                        if score > best and len(a & b) >= DRIFT_MIN_SHARED:
                            best, best_intent, best_text = score, other, s.text
                if best_intent is not None:
                    rep.warn(
                        intent,
                        index,
                        p.line,
                        f"possible intent drift: closer to {best_intent} seed {best_text!r} "
                        f"({best:.2f}) than to any {intent} seed ({own:.2f}): {p.text!r}",
                    )
    plain = [p for p in items if p.variant == Variant.PLAIN]
    if len(plain) >= MIN_FIRST_WORDS:
        firsts = {p.text.split()[0] for p in plain if p.text.split()}
        if len(firsts) < MIN_FIRST_WORDS:
            rep.warn(
                intent,
                index,
                plain[0].line,
                f"low shape diversity: plain paraphrases start with only "
                f"{len(firsts)} distinct words ({', '.join(sorted(firsts))})",
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _counts_table(rep: ValidationReport) -> str:
    cols = [str(v) for v in WRITTEN_VARIANTS]
    width = max([len("intent"), *(len(i) for i in rep.counts)])
    head = f"{'intent':<{width}}  " + "  ".join(f"{c:>15}" for c in cols) + f"  {'total':>6}"
    lines = [head]
    tot: Counter = Counter()
    for intent, c in rep.counts.items():
        tot.update(c)
        row = "  ".join(f"{c.get(v, 0):>15}" for v in cols)
        extra = sum(n for v, n in c.items() if v not in cols)
        lines.append(
            f"{intent:<{width}}  {row}  {sum(c.values()):>6}"
            + (f"  (+{extra} bad)" if extra else "")
        )
    row = "  ".join(f"{tot.get(v, 0):>15}" for v in cols)
    lines.append(f"{'all':<{width}}  {row}  {sum(tot.values()):>6}")
    return "\n".join(lines)


def _cmd_validate(a: argparse.Namespace) -> int:
    seeds = load_seeds(a.seeds)
    try:
        paraphrases = load_paraphrases(a.dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rep = validate(paraphrases, seeds, strict=a.strict, check_warnings=not a.no_warnings)
    if a.intents:
        wanted = [s for s in a.intents.split(",") if s]
        missing = [i for i in wanted if i not in paraphrases]
        rep = rep.only(wanted)
        for i in missing:
            rep.error(i, None, 0, f"no file {i}.yaml in {a.dir}")
    if not rep.counts and not rep.errors:
        print(f"no paraphrase files in {a.dir}")
        return 0
    print(_counts_table(rep))
    print()
    for issue in rep.errors + rep.warnings:
        print(issue.format(a.dir))
    print(
        f"\n{len(rep.errors)} errors, {len(rep.warnings)} warnings{' (strict)' if a.strict else ''}"
    )
    return 1 if rep.errors else 0


def status_lines(
    paraphrases: Mapping[str, Mapping[int, Sequence[Paraphrase]]],
    seeds: Mapping[str, Sequence[Seed]],
) -> list[str]:
    """One line per intent in taxonomy order: file present, counts against quota, done or not."""
    width = max(len(i.name) for i in INTENTS)
    lines = []
    done = 0
    have_total = need_total = 0
    for intent in (i.name for i in INTENTS):
        n = len(seeds.get(intent, ()))
        need = {v: n * q for v, q in QUOTA.items()}
        by_seed = paraphrases.get(intent)
        if by_seed is None:
            lines.append(f"{intent:<{width}}  {n} seeds  no file (needs {n * PER_SEED})")
            need_total += n * PER_SEED
            continue
        per_seed_ok = all(not quota_shortfall(by_seed.get(k, ())) for k in range(n))
        got = Counter(p.variant for items in by_seed.values() for p in items)
        cells = "  ".join(f"{v} {got.get(v, 0)}/{need[v]}" for v in WRITTEN_VARIANTS)
        missing = sum(max(0, need[v] - got.get(v, 0)) for v in WRITTEN_VARIANTS)
        done += per_seed_ok
        have_total += sum(min(got.get(v, 0), need[v]) for v in WRITTEN_VARIANTS)
        need_total += n * PER_SEED
        state = "complete" if per_seed_ok else f"{missing} to go"
        lines.append(f"{intent:<{width}}  {n} seeds  {cells}  {state}")
    lines.append("")
    lines.append(
        f"{done}/{len(INTENTS)} intents complete, {have_total}/{need_total} paraphrases "
        f"({len(paraphrases)} files)"
    )
    return lines


def _cmd_status(a: argparse.Namespace) -> int:
    try:
        paraphrases = load_paraphrases(a.dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n".join(status_lines(paraphrases, load_seeds(a.seeds))))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m autogate_bench.paraphrases",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check paraphrase files against the seeds")
    v.add_argument("--dir", type=Path, default=Path("data/paraphrases"))
    v.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS_PATH)
    v.add_argument("--strict", action="store_true", help="also require the per-seed quota")
    v.add_argument("--intents", default="", help="comma-separated intents to report on")
    v.add_argument("--no-warnings", action="store_true", help="skip the warning checks")
    v.set_defaults(func=_cmd_validate)
    s = sub.add_parser("status", help="show which intents have files and what is left")
    s.add_argument("--dir", type=Path, default=Path("data/paraphrases"))
    s.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS_PATH)
    s.set_defaults(func=_cmd_status)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
