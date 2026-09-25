"""Regenerate paraphrases with the Claude API (for third parties with their own key).

The released paraphrases were written by generation agents following
``docs/paraphrase-spec.md``. This script produces the same kind of file from
the short-form prompt in ``prompts/paraphrase.md``, one API call per seed::

    pip install 'autogate[api]'        # or: uv sync --extra api
    export ANTHROPIC_API_KEY=...       # or an `ant auth login` profile
    python -m autogate_bench.paraphrase_api --intents call_contact,wipers_on \\
        --out data/paraphrases --resume

For every seed of every requested intent it renders the prompt, calls the
model with a JSON-schema structured output, and checks the answer with
``autogate_bench.paraphrases.validate`` plus the per-seed quota. If anything
fails, it asks once more with the errors appended to the prompt; a seed that
still fails is left out of the file and reported, and the exit code is 1.
The intent file is rewritten after every seed, so an interrupted run loses
at most one call. ``--resume`` keeps every seed of an existing file that
already meets its quota and generates only the rest; without it an existing
file is refused unless ``--overwrite`` is given. ``--dry-run`` prints the
first rendered prompt and makes no call.

Requests use ``claude-opus-5`` by default (``--model`` to change), stream
the response (``get_final_message``) so a long answer cannot hit an HTTP
timeout, and opt into server-side refusal fallbacks (``fallbacks:
"default"``; ``--no-fallbacks`` to turn off, for example on a model that
does not support it).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.paraphrases import (
    QUOTA,
    WRITTEN_VARIANTS,
    Paraphrase,
    format_paraphrase_file,
    load_paraphrase_file,
    load_paraphrases,
    quota_shortfall,
    validate,
)
from autogate_bench.seeds import DEFAULT_SEEDS_PATH, Seed, load_seeds

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "paraphrase.md"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "paraphrases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "variant": {"type": "string", "enum": [str(v) for v in WRITTEN_VARIANTS]},
                },
                "required": ["text", "variant"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["paraphrases"],
    "additionalProperties": False,
}

PLACEHOLDERS: tuple[str, ...] = (
    "intent",
    "description",
    "group",
    "seed_text",
    "style",
    "slots",
    "quota_plain",
    "quota_hinglish",
    "quota_kannada_english",
    "quota_disfluent",
)

_FIELD = re.compile(r"\{(\w+)\}")


class GenerationError(RuntimeError):
    """The model's answer could not be used (refusal, truncation, bad JSON)."""


# --------------------------------------------------------------------------- #
# Prompt and response
# --------------------------------------------------------------------------- #


def prompt_values(intent: str, seed: Seed) -> dict[str, str]:
    info = INTENTS_BY_NAME[intent]
    slots = sorted(set(seed.slots))
    return {
        "intent": intent,
        "description": info.description,
        "group": str(info.group),
        "seed_text": seed.text,
        "style": str(seed.style),
        "slots": ", ".join(f"`{{{s}}}`" for s in slots) if slots else "none",
        **{f"quota_{v}": str(q) for v, q in QUOTA.items()},
    }


def render_prompt(template: str, intent: str, seed: Seed) -> str:
    """Fill the template's placeholders in one pass.

    Only the names in :data:`PLACEHOLDERS` are replaced, so literal slot
    examples such as ``{contact}`` in the template, and the braces of the
    seed text once inserted, are left alone.
    """
    values = prompt_values(intent, seed)
    return _FIELD.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def parse_response(text: str, intent: str, seed_index: int) -> list[Paraphrase]:
    """Turn the model's JSON answer into paraphrases (structure only; see :func:`check_seed`)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise GenerationError(f"response is not JSON: {e}") from None
    items = data.get("paraphrases") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise GenerationError("response has no 'paraphrases' list")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise GenerationError(f"paraphrase {i} is not {{text, variant}}")
        out.append(
            Paraphrase(intent, seed_index, item["text"].strip(), str(item.get("variant")), i + 1)
        )
    return out


def check_seed(
    intent: str,
    seed_index: int,
    items: Sequence[Paraphrase],
    seeds: Mapping[str, Sequence[Seed]],
    others: Mapping[str, Mapping[int, Sequence[Paraphrase]]],
) -> list[str]:
    """Every validation error for one seed's paraphrases, plus its quota.

    ``others`` holds the paraphrases already accepted (this intent's other
    seeds and other intents), so duplicates across them are caught too.
    Messages cite the item's position in the answer (1-based) as its line.
    """
    merged = {k: dict(v) for k, v in others.items()}
    merged.setdefault(intent, {})[seed_index] = tuple(items)
    report = validate(merged, seeds, check_warnings=False)
    errors = [
        f"item {e.line}: {e.message}"
        for e in report.errors
        if e.intent == intent and e.seed_index == seed_index
    ]
    short = quota_shortfall(items)
    if short:
        errors.append(f"wrong counts: {', '.join(short)} (want exactly these)")
    return errors


def retry_prompt(prompt: str, answer: str, errors: Sequence[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return (
        f"{prompt}\n\n## Your previous answer failed validation\n\n{listed}\n\n"
        f"Previous answer:\n\n```json\n{answer}\n```\n\n"
        "Return the complete corrected set: every variant at its exact count, "
        "each line obeying every hard rule."
    )


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def make_client():
    """An ``anthropic.Anthropic`` client, or ``SystemExit`` with a clear message."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "the anthropic SDK is not installed: pip install 'autogate[api]' "
            "(or uv sync --extra api)"
        ) from None
    message = (
        "no Anthropic credentials found: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), "
        "or run `ant auth login`. This script never runs without credentials."
    )
    try:
        client = anthropic.Anthropic()
    except anthropic.CredentialsError as e:  # a profile was selected but cannot be read
        raise SystemExit(f"{message}\n({e})") from None
    if not (client.api_key or client.auth_token or client.credentials):
        raise SystemExit(message)
    return client


def call_model(client: Any, prompt: str, model: str, fallbacks: bool = True) -> str:
    """One streamed request with a JSON-schema structured output; returns the JSON text."""
    kwargs: dict[str, Any] = {}
    if fallbacks:
        kwargs = {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    with client.beta.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        category = getattr(message.stop_details, "category", None)
        raise GenerationError(f"the model declined the request (category {category})")
    if message.stop_reason == "max_tokens":
        raise GenerationError(f"the answer was cut off at {MAX_TOKENS} tokens")
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise GenerationError(f"no text in the answer (stop_reason {message.stop_reason})")
    return text


def generate_seed(
    call: Any,
    template: str,
    intent: str,
    seed_index: int,
    seeds: Mapping[str, Sequence[Seed]],
    others: Mapping[str, Mapping[int, Sequence[Paraphrase]]],
) -> tuple[list[Paraphrase], list[str]]:
    """Ask for one seed's paraphrases, retrying once with the errors appended.

    ``call(prompt) -> str`` sends one request. Returns the paraphrases
    (in written-variant order) and the errors of the last attempt (empty on
    success).
    """
    prompt = render_prompt(template, intent, seeds[intent][seed_index])
    errors: list[str] = []
    answer = ""
    for attempt in range(2):
        answer = call(prompt if attempt == 0 else retry_prompt(prompt, answer, errors))
        try:
            items = parse_response(answer, intent, seed_index)
        except GenerationError as e:
            items, errors = [], [str(e)]
            continue
        errors = check_seed(intent, seed_index, items, seeds, others)
        if not errors:
            order = {str(v): i for i, v in enumerate(WRITTEN_VARIANTS)}
            items.sort(key=lambda p: order[p.variant])  # stable within a variant
            return [Paraphrase(intent, seed_index, p.text, p.variant) for p in items], []
    return [], errors


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m autogate_bench.paraphrase_api",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--intents", required=True, help="comma-separated intents, or 'all'")
    p.add_argument("--out", type=Path, default=Path("data/paraphrases"))
    p.add_argument("--resume", action="store_true", help="keep seeds already at quota")
    p.add_argument("--overwrite", action="store_true", help="replace existing files")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-fallbacks", action="store_true", help="no server-side refusal fallback")
    p.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS_PATH)
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    p.add_argument("--dry-run", action="store_true", help="print the first prompt, call nothing")
    a = p.parse_args(argv)

    seeds = load_seeds(a.seeds)
    intents = list(seeds) if a.intents == "all" else [s for s in a.intents.split(",") if s]
    unknown = [i for i in intents if i not in seeds]
    if unknown:
        p.error(f"unknown intents: {', '.join(unknown)}")
    template = a.prompt.read_text(encoding="utf-8")

    if a.dry_run:
        print(render_prompt(template, intents[0], seeds[intents[0]][0]))
        return 0

    for intent in intents:
        path = a.out / f"{intent}.yaml"
        if path.exists() and not (a.resume or a.overwrite):
            p.error(f"{path} exists; pass --resume to fill it in or --overwrite to replace it")

    client = make_client()
    import anthropic

    def call(prompt: str) -> str:
        return call_model(client, prompt, a.model, fallbacks=not a.no_fallbacks)

    a.out.mkdir(parents=True, exist_ok=True)
    others: dict[str, dict[int, tuple[Paraphrase, ...]]] = {
        k: dict(v) for k, v in load_paraphrases(a.out).items()
    }
    failed: list[str] = []
    for intent in intents:
        path = a.out / f"{intent}.yaml"
        kept: dict[int, tuple[Paraphrase, ...]] = {}
        if path.exists() and a.resume:
            existing = load_paraphrase_file(path)
            kept = {k: v for k, v in existing.items() if not quota_shortfall(v)}
        others[intent] = dict(kept)
        for index in range(len(seeds[intent])):
            if index in kept:
                continue
            print(f"{intent}/{index}: generating", file=sys.stderr)
            try:
                items, errors = generate_seed(call, template, intent, index, seeds, others)
            except anthropic.AuthenticationError:
                raise SystemExit("the API rejected the credentials (401); check your key") from None
            except (anthropic.APIStatusError, anthropic.APIConnectionError, GenerationError) as e:
                items, errors = [], [f"{type(e).__name__}: {e}"]
            if errors:
                failed.append(f"{intent}/{index}")
                for e in errors:
                    print(f"{intent}/{index}: {e}", file=sys.stderr)
                continue
            others[intent][index] = tuple(items)
            path.write_text(
                format_paraphrase_file(intent, others[intent], seeds[intent]), encoding="utf-8"
            )
    if failed:
        print(f"failed after one retry: {', '.join(failed)} (rerun with --resume)", file=sys.stderr)
        return 1
    print(f"done: {len(intents)} intents in {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
