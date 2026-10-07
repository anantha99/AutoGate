"""Zero-shot LLM baselines: an LLM prompted with the policy routes each row.

    python -m autogate_eval.baselines.llm --rows data/generated/full/rows.jsonl \\
        --split test,ood --provider anthropic --model claude-opus-5-5 --out-dir runs/opus-5-5

The system prompt holds the whole labeling policy (the precedence tree,
Tables 1-4 rendered from the ``Policy`` in use, the switches) and the intent
taxonomy with each intent's tags. Each row is one request: the vehicle
context and the utterance. The model answers with a JSON object, constrained
by a JSON schema where the backend supports it::

    {"intent": "<one of the 65>", "sensitive_spans": [{"text", "label"}], "route": "<ROUTE>"}

One answer gives two predictions files, the two decision modes of the paper:

``predictions_e2e.jsonl``
    end to end: the route the model chose.
``predictions_rulebook.jsonl``
    perception + rulebook: ``rulebook.route(answered intent, row context,
    any sensitive span answered, policy)``. Errors here are perception
    errors only.

Both carry the answered intent, and the answered spans mapped to character
offsets (exact substring, then case-insensitive; a span not found in the
utterance is dropped and counted), so ``autogate_eval.score`` reports intent
accuracy, span scores and the leak rate. A row whose answer is missing or
invalid (no JSON, an unknown intent or route, a refusal, a cut-off answer,
an API error) gets ``--on-invalid`` as its route (default ``REFUSE``: a gate
that cannot decide fails closed), intent ``__invalid__`` and no spans, and
is counted in ``summary.json``.

Providers

``anthropic``
    Claude through the Anthropic SDK (``pip install 'autogate[baselines]'``),
    with the system prompt cached, ``output_config.effort`` (default
    ``low``) and server-side refusal fallbacks on (``--no-fallbacks`` to
    turn them off). Each answer records the model that served it, and the
    summary counts rows served by a fallback model; rerun with
    ``--no-fallbacks`` if any were, so every row is the named model's.
``openai``
    Any OpenAI-compatible chat-completions endpoint: OpenAI itself, Gemini's
    OpenAI-compatible endpoint, or a local server such as vLLM for the
    on-device tiers (``vllm serve Qwen/Qwen3-4B-Instruct-2507``, then
    ``--base-url http://localhost:8000/v1``). ``--extra-body`` passes
    provider-specific fields, for example
    ``'{"chat_template_kwargs": {"enable_thinking": false}}'`` for Qwen3-0.6B.
``oracle``
    No model: answers with each row's gold intent, gold spans and gold
    route. Its rulebook file is the oracle-perception ceiling (perfect by
    construction, reported to make the circularity explicit); its e2e file
    is a pipeline check.

Every answer is appended to ``responses.jsonl`` as it arrives, so an
interrupted run resumes where it stopped (``--resume``; rows that failed
are retried). ``run.json`` records the configuration and the prompt's
sha256, and a resume with a different prompt is refused. ``summary.json``
has the counts, token usage, latency and an estimated cost (built-in prices
for Claude models; ``--price-in`` and friends for others, in USD per
million tokens). ``--dry-run`` prints the system prompt and the first
request and calls nothing; ``--sample N`` scores a seeded random subset.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from autogate_bench.intents import INTENTS, INTENTS_BY_NAME
from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.rulebook import route as rulebook_route
from autogate_bench.schema import (
    ActuationClass,
    Connectivity,
    DistractionLevel,
    DrivingDemand,
    Route,
    SpeedBucket,
    Workload,
)
from autogate_bench.seeds import SENSITIVE_SLOTS
from autogate_eval.baselines.rules import row_context
from autogate_eval.score import format_score, load_rows, score

ROUTES: tuple[str, ...] = tuple(str(r) for r in Route)
SPAN_LABELS: tuple[str, ...] = tuple(sorted(SENSITIVE_SLOTS))
INVALID_INTENT = "__invalid__"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [i.name for i in INTENTS]},
        "sensitive_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string", "enum": list(SPAN_LABELS)},
                },
                "required": ["text", "label"],
                "additionalProperties": False,
            },
        },
        "route": {"type": "string", "enum": list(ROUTES)},
    },
    "required": ["intent", "sensitive_spans", "route"],
    "additionalProperties": False,
}

# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
CLAUDE_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25, 12.5),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25),
}


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_TREE = """\
Apply these rules in order; the first that fires decides the route.

1. The intent's actuation is safety_critical -> LOCAL (always, whatever the context).
2. The intent's actuation is comfort or restricted and Table 3 does not allow it at the
   RAW driving demand -> REFUSE. (A passenger never relaxes this.)
3. Table 2 does not allow the intent's distraction level at the DISTRACTION demand
   -> REFUSE. (The distraction demand is the raw demand relaxed by the passenger rule.)
4. The intent is locally capable -> LOCAL. Locally capable means its Table 4
   capability is local_ok, or it is needs_small_local and the onboard model is small.
5. Connectivity is not good enough for the cloud (see switches): capability
   cloud_preferred -> LOCAL (onboard fallback); otherwise -> DEFER.
6. The utterance has sensitive spans: privacy mode strict -> the strict-privacy
   route (see switches); otherwise -> CLOUD_MASKED.
7. Otherwise -> CLOUD."""

_ROUTES_TEXT = """\
- LOCAL: answer on the head unit, nothing leaves the car.
- CLOUD: send the request to the cloud model as is.
- CLOUD_MASKED: send it to the cloud with the sensitive spans masked.
- DEFER: queue it until connectivity returns.
- REFUSE: do not run it now (unsafe or too distracting in this driving state)."""

_SPANS_TEXT = """\
A sensitive span is an exact substring of the utterance that names a specific person
(label contact), or gives a phone number (phone), a street address (address), a one-time
code (otp) or card digits (card). A relation or role word on its own ("my boss", "amma",
"mom") is not a span. Copy each span exactly as it appears in the utterance. Use an empty
list when there are none."""


def _grid(rows: Sequence[Any], cols: Sequence[Any], cell: Any, corner: str) -> str:
    head = f"| {corner} | " + " | ".join(str(c) for c in cols) + " |"
    sep = "| --- " * (len(cols) + 1) + "|"
    body = [f"| {r} | " + " | ".join(cell(r, c) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _allow(x: bool) -> str:
    return "allow" if x else "refuse"


def system_prompt(policy: Policy = DEFAULT_POLICY) -> str:
    """The full instructions: task, routes, rules, Tables 1-4, switches, intents, output."""
    t1 = _grid(SpeedBucket, Workload, lambda s, w: str(policy.demand(s, w)), "speed \\ workload")
    t2 = _grid(
        DistractionLevel,
        DrivingDemand,
        lambda lv, d: _allow(policy.distraction_allowed(lv, d)),
        "distraction \\ demand",
    )
    gated = [c for c in ActuationClass if c is not ActuationClass.NONE]
    t3 = _grid(
        gated, DrivingDemand, lambda c, d: _allow(policy.actuation_allowed(c, d)), "class \\ demand"
    )
    cloud = ", ".join(str(c) for c in Connectivity if c in policy.cloud_connectivity)
    intents = "\n".join(
        f"| {i.name} | {i.group} | {i.description} | {i.distraction} | {i.actuation} | "
        f"{policy.capability_of(i)} |"
        for i in INTENTS
    )
    return f"""\
You are the routing gate of an in-car voice assistant. For each request you get the
vehicle context and the driver's utterance (already transcribed; it may be code-mixed
Hindi-English or Kannada-English, disfluent, or contain speech-recognition errors).
Decide three things: which intent the utterance expresses, which sensitive spans it
contains, and where the request should run.

## Routes

{_ROUTES_TEXT}

## Rules

{_TREE}

## Table 1: raw driving demand from speed and driver workload

{t1}

## Table 2: distraction gate (is a response of this distraction level allowed?)

{t2}

## Table 3: actuation gate (may the car perform this class of command?)

{t3}

## Switches

- Passenger rule: if a passenger is present, the distraction demand is the raw demand
  relaxed by {policy.passenger_relax_steps} step(s) in the order parked < low < medium <
  high, never below low and never for a parked car. It never changes Table 3.
- Connectivity good enough for the cloud: {cloud}.
- Strict-privacy route: {policy.strict_privacy_route}.

## Table 4: capability meanings

- local_ok: any onboard model can answer it, including a tiny one.
- needs_small_local: the onboard model can answer it only if it is small (<= 8B); a tiny
  (<= 2B) model cannot.
- needs_cloud: needs live data or a frontier model.
- cloud_preferred: better in the cloud, but onboard data can answer it.

## Intents

| name | group | description | distraction | actuation | capability |
| --- | --- | --- | --- | --- | --- |
{intents}

## Sensitive spans

{_SPANS_TEXT}

## Answer

Reply with only a JSON object: {{"intent": <intent name>, "sensitive_spans": [{{"text":
<exact substring>, "label": <{" | ".join(SPAN_LABELS)}>}}], "route": <{" | ".join(ROUTES)}>}}.
"""


def user_message(row: Mapping[str, Any]) -> str:
    ctx = row_context(row)
    return (
        f"Vehicle context: speed {ctx.speed_bucket}, connectivity {ctx.connectivity}, "
        f"driver workload {ctx.driver_workload}, privacy mode {ctx.privacy_mode}, "
        f"passenger {'present' if ctx.passenger_present else 'not present'}, "
        f"onboard model {ctx.local_model_tier}.\n"
        f"Utterance: {json.dumps(row['utterance'], ensure_ascii=False)}"
    )


def prompt_sha256(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Answer:
    intent: str
    route: str
    spans: tuple[tuple[str, str], ...]  # (text, label)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_answer(text: str | None) -> tuple[Answer | None, str | None]:
    """The model's answer, or ``(None, reason)`` when it cannot be used."""
    if not text:
        return None, "empty"
    s = _FENCE.sub("", text.strip())
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < a:
        return None, "no_json"
    try:
        data = json.loads(s[a : b + 1])
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(data, dict):
        return None, "bad_json"
    intent, route = data.get("intent"), data.get("route")
    if intent not in INTENTS_BY_NAME:
        return None, "unknown_intent"
    if route not in ROUTES:
        return None, "unknown_route"
    spans = []
    for item in data.get("sensitive_spans") or []:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            return None, "bad_spans"
        if item.get("label") not in SPAN_LABELS:
            return None, "bad_spans"
        if item["text"].strip():
            spans.append((item["text"].strip(), item["label"]))
    return Answer(intent, route, tuple(spans)), None


def locate_spans(utterance: str, spans: Sequence[tuple[str, str]]) -> tuple[list[dict], int]:
    """Character spans for the answered substrings, and how many could not be found.

    Each substring takes its first occurrence (exact, then case-insensitive) that does
    not overlap a span already placed.
    """
    placed: list[tuple[int, int, str]] = []
    missing = 0
    lower = utterance.lower()
    for text, label in spans:
        hit = None
        for hay, needle in ((utterance, text), (lower, text.lower())):
            start = hay.find(needle)
            while start >= 0:
                end = start + len(needle)
                if all(end <= x or start >= y for x, y, _ in placed):
                    hit = (start, end)
                    break
                start = hay.find(needle, start + 1)
            if hit:
                break
        if hit is None:
            missing += 1
        else:
            placed.append((hit[0], hit[1], label))
    placed.sort()
    spans_out = [
        {"start": a, "end": b, "label": lab, "text": utterance[a:b]} for a, b, lab in placed
    ]
    return spans_out, missing


def predictions(
    rows: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
    policy: Policy = DEFAULT_POLICY,
    on_invalid: str = "REFUSE",
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """The e2e and rulebook predictions for every row, plus answer counts."""
    e2e, rb = [], []
    invalid: Counter[str] = Counter()
    unplaced = 0
    for row in rows:
        resp = responses.get(row["id"])
        answer, reason = (None, "missing") if resp is None else parse_answer(resp.get("text"))
        if resp is not None and resp.get("error"):
            answer, reason = None, "error"
        if answer is None:
            invalid[reason or "invalid"] += 1
            base = {"intent": INVALID_INTENT, "spans": [], "valid": False}
            e2e_route = rb_route = on_invalid
        else:
            spans, missing = locate_spans(row["utterance"], answer.spans)
            unplaced += missing
            base = {"intent": answer.intent, "spans": spans, "valid": True}
            e2e_route = answer.route
            rb_route = str(
                rulebook_route(
                    INTENTS_BY_NAME[answer.intent], row_context(row), bool(answer.spans), policy
                ).route
            )
        common = {"id": row["id"], "route_probs": None, "intent_probs_top3": None}
        e2e.append({**common, "route": e2e_route, **base, "confidence": None})
        rb.append({**common, "route": rb_route, **base, "confidence": None})
    counts = {
        "rows": len(rows),
        "valid": len(rows) - sum(invalid.values()),
        "invalid": dict(sorted(invalid.items())),
        "spans_not_found_in_utterance": unplaced,
    }
    return e2e, rb, counts


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


@dataclass
class Completion:
    text: str | None
    served_by: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class Backend(Protocol):
    model: str

    async def complete(self, system: str, user: str, row: Mapping[str, Any]) -> Completion: ...


class OracleBackend:
    """Answers with the row's gold intent, spans and route; calls nothing."""

    model = "oracle"

    async def complete(self, system: str, user: str, row: Mapping[str, Any]) -> Completion:
        spans = [{"text": s["text"], "label": s["label"]} for s in row["spans"]]
        answer = {"intent": row["intent"], "sensitive_spans": spans, "route": row["route"]}
        return Completion(json.dumps(answer), served_by="oracle")


class AnthropicBackend:
    def __init__(
        self,
        model: str,
        effort: str | None = "low",
        fallbacks: bool = True,
        max_tokens: int = 4096,
    ):
        try:
            import anthropic
        except ImportError:
            raise SystemExit(
                "the anthropic SDK is not installed: pip install 'autogate[baselines]' "
                "(or uv sync --extra baselines)"
            ) from None
        message = (
            "no Anthropic credentials found: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), "
            "or run `ant auth login`"
        )
        try:
            self.client = anthropic.AsyncAnthropic(max_retries=6)
        except anthropic.CredentialsError as e:
            raise SystemExit(f"{message}\n({e})") from None
        c = self.client
        if not (c.api_key or c.auth_token or getattr(c, "credentials", None)):
            raise SystemExit(message)
        self._anthropic = anthropic
        self.model, self.effort, self.fallbacks, self.max_tokens = (
            model,
            effort,
            fallbacks,
            max_tokens,
        )

    async def complete(self, system: str, user: str, row: Mapping[str, Any]) -> Completion:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = {}
        if self.fallbacks:
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            msg = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
                **kwargs,
            )
        except self._anthropic.AuthenticationError:
            raise SystemExit("the API rejected the credentials (401); check your key") from None
        except self._anthropic.BadRequestError as e:
            raise SystemExit(f"the API rejected the request (400): {e.message}") from None
        except self._anthropic.APIError as e:
            return Completion(None, error=f"{type(e).__name__}: {e}")
        u = msg.usage
        usage = {
            "input": u.input_tokens or 0,
            "output": u.output_tokens or 0,
            "cache_read": u.cache_read_input_tokens or 0,
            "cache_write": u.cache_creation_input_tokens or 0,
        }
        text = next((b.text for b in msg.content or [] if b.type == "text"), None)
        error = None
        if msg.stop_reason == "refusal":
            error = f"refusal ({getattr(msg.stop_details, 'category', None)})"
        elif msg.stop_reason == "max_tokens":
            error = f"cut off at {self.max_tokens} tokens"
        return Completion(text, msg.model, error, msg.stop_reason, usage)


class OpenAICompatBackend:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        response_format: str = "json_schema",
        extra_body: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise SystemExit(
                "the openai SDK is not installed: pip install 'autogate[baselines]' "
                "(or uv sync --extra baselines)"
            ) from None
        key = os.environ.get(api_key_env)
        if not key:
            if base_url and re.match(r"https?://(localhost|127\.0\.0\.1)", base_url):
                key = "EMPTY"  # local servers ignore the key
            else:
                raise SystemExit(f"no API key: set {api_key_env} (or pass --api-key-env)")
        self.client = openai.AsyncOpenAI(api_key=key, base_url=base_url, max_retries=6)
        self._openai = openai
        self.model = model
        self.kwargs: dict[str, Any] = {}
        if response_format == "json_schema":
            self.kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "route_decision", "schema": OUTPUT_SCHEMA, "strict": True},
            }
        elif response_format == "json_object":
            self.kwargs["response_format"] = {"type": "json_object"}
        if extra_body:
            self.kwargs["extra_body"] = dict(extra_body)
        if temperature is not None:
            self.kwargs["temperature"] = temperature
        if max_tokens is not None:
            self.kwargs["max_completion_tokens"] = max_tokens

    async def complete(self, system: str, user: str, row: Mapping[str, Any]) -> Completion:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **self.kwargs,
            )
        except self._openai.AuthenticationError:
            raise SystemExit("the API rejected the credentials (401); check your key") from None
        except self._openai.BadRequestError as e:
            raise SystemExit(f"the API rejected the request (400): {e}") from None
        except self._openai.APIError as e:
            return Completion(None, error=f"{type(e).__name__}: {e}")
        if not resp.choices:
            return Completion(None, resp.model, "no choices in the response")
        choice = resp.choices[0]
        usage: dict[str, int] = {}
        if resp.usage is not None:
            details = getattr(resp.usage, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
            usage = {
                "input": (resp.usage.prompt_tokens or 0) - cached,
                "output": resp.usage.completion_tokens or 0,
                "cache_read": cached,
                "cache_write": 0,
            }
        error = None
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            error = f"refusal ({refusal[:80]})"
        elif choice.finish_reason == "length":
            error = "cut off by the token limit"
        return Completion(choice.message.content, resp.model, error, choice.finish_reason, usage)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def read_responses(path: Path) -> dict[str, dict[str, Any]]:
    """The last record per id in ``responses.jsonl`` (later lines win)."""
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["id"]] = rec
    return out


async def run(
    rows: Sequence[Mapping[str, Any]],
    backend: Backend,
    system: str,
    responses_path: Path,
    concurrency: int = 8,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Ask the backend about every row without a usable response; append each answer."""
    done = read_responses(responses_path)
    todo = [r for r in rows if r["id"] not in done or done[r["id"]].get("error")]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    finished = 0

    with responses_path.open("a", encoding="utf-8", newline="\n") as f:

        async def one(row: Mapping[str, Any]) -> None:
            nonlocal finished
            async with sem:
                t0 = time.perf_counter()
                c = await backend.complete(system, user_message(row), row)
                rec = {
                    "id": row["id"],
                    "model": backend.model,
                    "served_by": c.served_by,
                    "text": c.text,
                    "error": c.error,
                    "stop_reason": c.stop_reason,
                    "usage": c.usage,
                    "latency_s": round(time.perf_counter() - t0, 4),
                }
            async with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[row["id"]] = rec
                finished += 1
                if progress and (finished % 100 == 0 or finished == len(todo)):
                    print(f"{finished}/{len(todo)} answered", file=sys.stderr)

        await asyncio.gather(*(one(r) for r in todo))
    return done


def _served_by_other(served: str | None, model: str) -> bool:
    """A response from a model other than the one asked for (a dated id of it counts as it)."""
    return bool(served) and not served.startswith(model)


def _percentile(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def usage_summary(
    responses: Sequence[Mapping[str, Any]],
    model: str,
    prices: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    tokens: Counter[str] = Counter()
    for r in responses:
        tokens.update(r.get("usage") or {})
    served = Counter(r.get("served_by") or "none" for r in responses)
    latencies = [r["latency_s"] for r in responses if r.get("latency_s") is not None]
    cost = None
    if prices is not None:
        p_in, p_out, p_read, p_write = prices
        cost = (
            tokens["input"] * p_in
            + tokens["output"] * p_out
            + tokens["cache_read"] * p_read
            + tokens["cache_write"] * p_write
        ) / 1e6
    n = len(responses)
    return {
        "model": model,
        "served_by": dict(served),
        "served_by_another_model": sum(
            _served_by_other(r.get("served_by"), model) for r in responses
        ),
        "tokens": dict(tokens),
        "prices_usd_per_mtok": None
        if prices is None
        else dict(zip(("input", "output", "cache_read", "cache_write"), prices, strict=True)),
        "estimated_cost_usd": None if cost is None else round(cost, 4),
        "estimated_cost_usd_per_1k_requests": None if cost is None or not n else cost / n * 1000,
        "latency_s": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
    }


def write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    return path


def select_rows(
    rows: Sequence[Mapping[str, Any]], splits: str | None, sample: int | None, seed: int
) -> list[Mapping[str, Any]]:
    if splits:
        keep = {s.strip() for s in splits.split(",") if s.strip()}
        rows = [r for r in rows if r["split"] in keep]
    rows = list(rows)
    if sample is not None and sample < len(rows):
        picked = set(random.Random(seed).sample(range(len(rows)), sample))
        rows = [r for i, r in enumerate(rows) if i in picked]
    return rows


def make_backend(a: argparse.Namespace) -> Backend:
    if a.provider == "oracle":
        return OracleBackend()
    if not a.model:
        raise SystemExit("--model is required for this provider")
    if a.provider == "anthropic":
        effort = None if a.effort == "none" else a.effort
        return AnthropicBackend(a.model, effort, not a.no_fallbacks, a.max_tokens or 4096)
    return OpenAICompatBackend(
        a.model,
        a.base_url,
        a.api_key_env,
        a.response_format,
        json.loads(a.extra_body) if a.extra_body else None,
        a.temperature,
        a.max_tokens,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(
        prog="python -m autogate_eval.baselines.llm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rows", type=Path, required=True, help="rows.jsonl or rows.parquet")
    p.add_argument("--split", default="test,ood", help="comma-separated splits ('' for all)")
    p.add_argument("--sample", type=int, default=None, help="score a random subset of N rows")
    p.add_argument("--seed", type=int, default=0, help="seed for --sample")
    p.add_argument("--provider", choices=("anthropic", "openai", "oracle"), required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--policy", type=Path, default=None, help="policy JSON (default: built-in)")
    p.add_argument("--resume", action="store_true", help="continue an existing run")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--on-invalid", choices=ROUTES, default="REFUSE")
    p.add_argument("--dry-run", action="store_true", help="print the prompt, call nothing")
    p.add_argument("--n-boot", type=int, default=1000)
    g = p.add_argument_group("anthropic")
    g.add_argument("--effort", default="low", help="low|medium|high|xhigh|max, or none to omit")
    g.add_argument("--no-fallbacks", action="store_true", help="no server-side refusal fallback")
    g = p.add_argument_group("openai-compatible")
    g.add_argument("--base-url", default=None, help="e.g. http://localhost:8000/v1 for vLLM")
    g.add_argument("--api-key-env", default="OPENAI_API_KEY")
    g.add_argument(
        "--response-format", choices=("json_schema", "json_object", "none"), default="json_schema"
    )
    g.add_argument("--extra-body", default=None, help="JSON object merged into each request")
    g.add_argument("--temperature", type=float, default=None)
    g = p.add_argument_group("both")
    g.add_argument("--max-tokens", type=int, default=None)
    g = p.add_argument_group("cost (USD per million tokens; built in for Claude models)")
    g.add_argument("--price-in", type=float, default=None)
    g.add_argument("--price-out", type=float, default=None)
    g.add_argument("--price-cache-read", type=float, default=None)
    g.add_argument("--price-cache-write", type=float, default=None)
    a = p.parse_args(argv)

    policy = Policy.load(a.policy) if a.policy else DEFAULT_POLICY
    system = system_prompt(policy)
    rows = select_rows(load_rows(a.rows), a.split, a.sample, a.seed)
    if not rows:
        p.error("no rows selected")
    if a.dry_run:
        print(system)
        print("---")
        print(user_message(rows[0]))
        print(f"--- {len(rows)} rows; system prompt ~{len(system) // 4} tokens", file=sys.stderr)
        return {}

    run_path = a.out_dir / "run.json"
    responses_path = a.out_dir / "responses.jsonl"
    config = {
        "provider": a.provider,
        "model": a.model or a.provider,
        "rows": str(a.rows),
        "split": a.split,
        "sample": a.sample,
        "seed": a.seed,
        "policy": str(a.policy) if a.policy else "default",
        "prompt_sha256": prompt_sha256(system),
        "on_invalid": a.on_invalid,
        "effort": a.effort if a.provider == "anthropic" else None,
        "fallbacks": (not a.no_fallbacks) if a.provider == "anthropic" else None,
        "base_url": a.base_url,
        "extra_body": a.extra_body,
        "response_format": a.response_format if a.provider == "openai" else None,
    }
    if responses_path.exists():
        if not a.resume:
            p.error(f"{responses_path} exists; pass --resume to continue it")
        old = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
        if old.get("prompt_sha256") != config["prompt_sha256"]:
            p.error("the existing run used a different prompt or policy; use a new --out-dir")
        if old.get("model") != config["model"]:
            p.error(f"the existing run used model {old.get('model')!r}; use a new --out-dir")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    backend = make_backend(a)
    responses = asyncio.run(run(rows, backend, system, responses_path, a.concurrency))

    e2e, rb, counts = predictions(rows, responses, policy, a.on_invalid)
    write_jsonl(e2e, a.out_dir / "predictions_e2e.jsonl")
    write_jsonl(rb, a.out_dir / "predictions_rulebook.jsonl")

    prices = CLAUDE_PRICES.get(config["model"]) if a.provider == "anthropic" else None
    if a.price_in is not None and a.price_out is not None:
        prices = (
            a.price_in,
            a.price_out,
            a.price_cache_read if a.price_cache_read is not None else a.price_in,
            a.price_cache_write if a.price_cache_write is not None else a.price_in,
        )
    used = [responses[r["id"]] for r in rows if r["id"] in responses]
    summary = {
        "config": config,
        "answers": counts,
        "usage": usage_summary(used, config["model"], prices),
    }
    results = {}
    for name, preds in (("e2e", e2e), ("rulebook", rb)):
        results[name] = score(rows, preds, n_boot=a.n_boot)
        (a.out_dir / f"score_{name}.json").write_text(
            json.dumps(results[name], indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n===== {config['model']}: {name} =====\n{format_score(results[name])}")
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nanswers: {json.dumps(counts)}")
    print(f"usage: {json.dumps(summary['usage'])}")
    if summary["usage"]["served_by_another_model"]:
        print(
            f"note: {summary['usage']['served_by_another_model']} rows were served by a fallback "
            "model; rerun with --no-fallbacks for a pure baseline",
            file=sys.stderr,
        )
    return summary


if __name__ == "__main__":
    main()
