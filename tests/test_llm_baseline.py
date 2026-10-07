"""autogate_eval.baselines.llm: prompt, answer parsing, both decision modes, runs, backends.

Nothing here calls a real API. The backend tests point the real SDKs at a local HTTP
server that imitates the Anthropic and OpenAI-compatible endpoints.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from autogate_bench.intents import INTENTS
from autogate_bench.policy import DEFAULT_CAPABILITY, Policy
from autogate_bench.schema import Capability, Route
from autogate_eval.baselines import llm
from autogate_eval.baselines.llm import (
    INVALID_INTENT,
    OUTPUT_SCHEMA,
    Completion,
    OracleBackend,
    locate_spans,
    parse_answer,
    predictions,
    read_responses,
    run,
    system_prompt,
    user_message,
)


def _row(**kw):
    base = {
        "id": "r1",
        "utterance": "call ravi gowda now",
        "intent": "call_contact",
        "route": "LOCAL",
        "spans": [{"start": 5, "end": 15, "label": "contact", "text": "ravi gowda"}],
        "split": "test",
        "speed_bucket": "high",
        "connectivity": "good",
        "driver_workload": "low",
        "privacy_mode": "standard",
        "passenger_present": False,
        "local_model_tier": "tiny",
    }
    return {**base, **kw}


def _answer(intent="call_contact", route="LOCAL", spans=()):
    return json.dumps(
        {
            "intent": intent,
            "sensitive_spans": [{"text": t, "label": lab} for t, lab in spans],
            "route": route,
        }
    )


# ---- Prompt ------------------------------------------------------------------ #


def test_system_prompt_lists_every_intent_route_and_table():
    text = system_prompt()
    for intent in INTENTS:
        assert f"| {intent.name} |" in text
    for r in Route:
        assert str(r) in text
    for heading in ("Table 1", "Table 2", "Table 3", "Table 4"):
        assert heading in text


def test_system_prompt_follows_the_policy():
    """An OEM variant changes what the model is told (RQ4 relabels need this)."""
    strong = Policy(capability={**DEFAULT_CAPABILITY, "read_messages": Capability.LOCAL_OK})
    line = next(ln for ln in system_prompt(strong).splitlines() if "| read_messages |" in ln)
    assert line.endswith("| local_ok |")
    line = next(ln for ln in system_prompt().splitlines() if "| read_messages |" in ln)
    assert line.endswith("| needs_small_local |")


def test_user_message_carries_context_and_utterance():
    msg = user_message(_row(passenger_present=True, local_model_tier="small"))
    assert "speed high" in msg and "passenger present" in msg and "onboard model small" in msg
    assert msg.endswith('Utterance: "call ravi gowda now"')


def test_output_schema_enumerates_intents_and_routes():
    props = OUTPUT_SCHEMA["properties"]
    assert len(props["intent"]["enum"]) == len(INTENTS)
    assert props["route"]["enum"] == [str(r) for r in Route]


# ---- Answers ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (None, "empty"),
        ("i think LOCAL", "no_json"),
        ('{"intent": "call_contact", "route": "LOCAL"', "no_json"),
        ('{"intent": "teleport", "sensitive_spans": [], "route": "LOCAL"}', "unknown_intent"),
        ('{"intent": "call_contact", "sensitive_spans": [], "route": "MAYBE"}', "unknown_route"),
        (
            '{"intent": "call_contact", "sensitive_spans": [{"text": "x", "label": "ssn"}], '
            '"route": "LOCAL"}',
            "bad_spans",
        ),
    ],
)
def test_parse_answer_rejects(text, reason):
    assert parse_answer(text) == (None, reason)


def test_parse_answer_accepts_fenced_json_and_drops_blank_spans():
    text = "```json\n" + _answer(spans=[("ravi", "contact"), ("  ", "phone")]) + "\n```"
    answer, reason = parse_answer(text)
    assert reason is None
    assert answer.intent == "call_contact" and answer.route == "LOCAL"
    assert answer.spans == (("ravi", "contact"),)


def test_locate_spans_exact_then_case_insensitive_without_overlap():
    spans, missing = locate_spans(
        "Ravi said ravi", [("ravi", "contact"), ("RAVI", "contact"), ("x", "otp")]
    )
    assert [(s["start"], s["end"]) for s in spans] == [(0, 4), (10, 14)]
    assert missing == 1


# ---- Both decision modes ---------------------------------------------------------- #


def test_rulebook_mode_recomputes_the_route_from_the_context():
    # Restricted actuation at speed: the rulebook refuses whatever route the model picked.
    row = _row(
        id="u", utterance="unlock the doors", intent="unlock_doors", route="REFUSE", spans=[]
    )
    responses = {"u": {"text": _answer("unlock_doors", "LOCAL")}}
    e2e, rb, counts = predictions([row], responses)
    assert e2e[0]["route"] == "LOCAL" and rb[0]["route"] == "REFUSE"
    assert counts == {"rows": 1, "valid": 1, "invalid": {}, "spans_not_found_in_utterance": 0}


def test_answered_spans_make_the_rulebook_mask():
    row = _row(intent="navigate_to_contact_address", route="CLOUD_MASKED")
    answer = _answer("navigate_to_contact_address", "CLOUD", [("ravi gowda", "contact")])
    e2e, rb, _ = predictions([row], {"r1": {"text": answer}})
    assert rb[0]["route"] == "CLOUD_MASKED" and e2e[0]["route"] == "CLOUD"
    assert e2e[0]["spans"] == [{"start": 5, "end": 15, "label": "contact", "text": "ravi gowda"}]


def test_invalid_missing_and_failed_answers_fail_closed():
    rows = [_row(id="a"), _row(id="b"), _row(id="c")]
    responses = {
        "a": {"text": "no idea"},
        "c": {"text": _answer(), "error": "refusal (cyber)"},
    }
    e2e, rb, counts = predictions(rows, responses, on_invalid="REFUSE")
    assert {p["route"] for p in e2e + rb} == {"REFUSE"}
    assert {p["intent"] for p in e2e} == {INVALID_INTENT}
    assert counts["invalid"] == {"error": 1, "missing": 1, "no_json": 1}


# ---- Runs ---------------------------------------------------------------------- #


class FlakyBackend:
    """Fails the first call for ids in ``fail_once``, then answers like the oracle."""

    model = "flaky"

    def __init__(self, fail_once):
        self.fail_once, self.calls = set(fail_once), []

    async def complete(self, system, user, row):
        self.calls.append(row["id"])
        if row["id"] in self.fail_once:
            self.fail_once.discard(row["id"])
            return Completion(None, error="APIConnectionError: boom")
        return await OracleBackend().complete(system, user, row)


def test_run_appends_answers_and_resume_retries_only_failures(tmp_path):
    rows = [_row(id=f"r{i}") for i in range(5)]
    path = tmp_path / "responses.jsonl"
    backend = FlakyBackend({"r1", "r3"})
    asyncio.run(run(rows, backend, "sys", path, concurrency=2, progress=False))
    assert sorted(backend.calls) == [f"r{i}" for i in range(5)]
    assert {k for k, v in read_responses(path).items() if v["error"]} == {"r1", "r3"}

    backend.calls = []
    done = asyncio.run(run(rows, backend, "sys", path, concurrency=2, progress=False))
    assert sorted(backend.calls) == ["r1", "r3"]
    assert not any(v["error"] for v in done.values())


def test_oracle_rulebook_is_the_perfect_ceiling(pilot_rows, tmp_path, capsys):
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in pilot_rows), encoding="utf-8")
    out = tmp_path / "oracle"
    argv = ["--rows", str(rows_path), "--provider", "oracle", "--out-dir", str(out)]
    summary = llm.main([*argv, "--n-boot", "20"])
    assert summary["answers"]["invalid"] == {}
    rb = json.loads((out / "score_rulebook.json").read_text())
    assert rb["route"]["overall"]["accuracy"] == 1.0
    assert rb["intent"]["accuracy"] == 1.0 and rb["spans"]["leak_rate"] in (0.0, None)
    assert (out / "predictions_e2e.jsonl").exists() and (out / "summary.json").exists()

    with pytest.raises(SystemExit):  # an existing run needs --resume
        llm.main(argv)
    policy = tmp_path / "policy.json"
    Policy(capability={**DEFAULT_CAPABILITY, "read_messages": Capability.LOCAL_OK}).save(policy)
    with pytest.raises(SystemExit):  # and the same prompt
        llm.main([*argv, "--resume", "--policy", str(policy)])
    capsys.readouterr()


# ---- Backends against a local imitation of the APIs ------------------------------ #


class _FakeAPI(BaseHTTPRequestHandler):
    requests: list = []
    answer = _answer()

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, dict(self.headers), body))
        if "/v1/messages" in self.path:
            out = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": self.answer}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 30,
                    "cache_read_input_tokens": 2600,
                    "cache_creation_input_tokens": 0,
                },
            }
        else:
            out = {
                "id": "cmpl_1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"] + "-2026-09-03",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2640,
                    "completion_tokens": 30,
                    "total_tokens": 2670,
                    "prompt_tokens_details": {"cached_tokens": 2600},
                },
            }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_api(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _FakeAPI.requests = []
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_anthropic_backend_request_and_usage(fake_api, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", fake_api)
    backend = llm.AnthropicBackend("claude-opus-5-5")
    c = asyncio.run(backend.complete("SYSTEM", "USER", _row()))
    assert c.error is None and c.served_by == "claude-opus-5-5"
    assert parse_answer(c.text)[0].intent == "call_contact"
    assert c.usage == {"input": 40, "output": 30, "cache_read": 2600, "cache_write": 0}

    path, headers, body = _FakeAPI.requests[-1]
    assert "/v1/messages" in path
    assert body["system"] == [
        {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["output_config"] == {
        "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        "effort": "low",
    }
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")


def test_openai_compatible_backend_request_and_usage(fake_api):
    pytest.importorskip("openai")
    backend = llm.OpenAICompatBackend(
        "qwen3-4b",
        base_url=fake_api + "/v1",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        temperature=0.0,
    )
    c = asyncio.run(backend.complete("SYSTEM", "USER", _row()))
    assert c.error is None and c.served_by == "qwen3-4b-2026-09-03"
    assert not llm._served_by_other(c.served_by, "qwen3-4b")  # a dated id is the same model
    assert c.usage == {"input": 40, "output": 30, "cache_read": 2600, "cache_write": 0}

    path, _, body = _FakeAPI.requests[-1]
    assert path.endswith("/v1/chat/completions")
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    assert body["response_format"]["json_schema"]["schema"] == OUTPUT_SCHEMA
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.0


def test_cli_runs_a_sampled_openai_compatible_baseline(fake_api, pilot_rows, tmp_path, capsys):
    pytest.importorskip("openai")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in pilot_rows), encoding="utf-8")
    out = tmp_path / "local"
    summary = llm.main(
        [
            *("--rows", str(rows_path), "--split", "test", "--sample", "12"),
            *("--provider", "openai", "--model", "qwen3-4b", "--out-dir", str(out)),
            *("--base-url", fake_api + "/v1", "--price-in", "1", "--price-out", "2"),
            *("--price-cache-read", "0.5", "--n-boot", "20"),
        ]
    )
    assert summary["answers"]["rows"] == 12 and summary["answers"]["valid"] == 12
    assert summary["usage"]["served_by_another_model"] == 0
    # 12 x (40 x $1 + 30 x $2 + 2600 x $0.50) per million tokens
    assert summary["usage"]["estimated_cost_usd"] == pytest.approx(12 * 1400 / 1e6)
    assert len(read_responses(out / "responses.jsonl")) == 12
    capsys.readouterr()


def test_openai_backend_needs_a_key_for_remote_endpoints(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        llm.OpenAICompatBackend("gpt-x", base_url="https://api.example.com/v1")


def test_cost_estimate_uses_cache_prices():
    responses = [
        {"usage": {"input": 40, "output": 30, "cache_read": 2600, "cache_write": 0}, "latency_s": 1}
    ] * 1000
    u = llm.usage_summary(responses, "claude-opus-5-5", llm.CLAUDE_PRICES["claude-opus-5-5"])
    # 1000 x (40 x $4 + 30 x $20 + 2600 x $0.20) per million tokens
    assert u["estimated_cost_usd"] == pytest.approx(1.28)
    assert u["estimated_cost_usd_per_1k_requests"] == pytest.approx(1.28)
