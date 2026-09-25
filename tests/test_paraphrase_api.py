"""The reproducibility script: prompt rendering, response parsing, retry, no-credential exit.

Nothing here calls the API.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from autogate_bench import paraphrase_api as api
from autogate_bench.paraphrases import QUOTA, load_paraphrase_file, load_paraphrases, validate
from autogate_bench.seeds import Seed, SeedStyle, load_seeds

SEEDS = load_seeds()
TEMPLATE = api.DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
FIXTURE = Path(__file__).parent / "fixtures" / "paraphrases"
CANNED = [
    {"text": p.text, "variant": p.variant}
    for p in load_paraphrase_file(FIXTURE / "start_video_call.yaml")[0]
]


def canned(items=CANNED) -> str:
    return json.dumps({"paraphrases": list(items)})


# ---- prompt --------------------------------------------------------------- #


def test_template_has_every_placeholder():
    for name in api.PLACEHOLDERS:
        assert "{" + name + "}" in TEMPLATE, name


def test_render_fills_placeholders_and_keeps_slot_examples():
    out = api.render_prompt(TEMPLATE, "start_video_call", SEEDS["start_video_call"][0])
    assert not any("{" + n + "}" in out for n in api.PLACEHOLDERS)
    assert "`video call {contact}`" in out  # the seed, braces intact
    assert "slots: `{contact}`" in out
    assert "(communication): Start a video call with a contact" in out
    assert "style: `command`" in out
    for v, q in QUOTA.items():
        assert f"{q} `{v}`" in out
    assert "{contact} ge call maadu" in out  # literal example left alone


def test_render_seed_without_slots_and_with_braces_like_placeholders():
    out = api.render_prompt(TEMPLATE, "wipers_on", SEEDS["wipers_on"][1])
    assert "slots: none" in out
    tricky = Seed("add_reminder", "remind me about {note_text} at {time}", SeedStyle.COMMAND)
    out = api.render_prompt("{seed_text} / {slots} / {intent}", "add_reminder", tricky)
    assert out == "remind me about {note_text} at {time} / `{note_text}`, `{time}` / add_reminder"


def test_prompt_schema_matches_the_structured_output_schema():
    block = re.search(r"```json\n(.*?)\n```", TEMPLATE, re.S).group(1)
    assert json.loads(block) == api.OUTPUT_SCHEMA


# ---- parsing and validation ------------------------------------------------ #


def test_parse_canned_response_into_validated_paraphrases():
    items = api.parse_response(canned(), "start_video_call", 0)
    assert len(items) == 40
    assert items[0].intent == "start_video_call" and items[0].seed_index == 0
    assert api.check_seed("start_video_call", 0, items, SEEDS, {}) == []
    rep = validate({"start_video_call": {0: tuple(items)}}, SEEDS)
    assert rep.errors == []


@pytest.mark.parametrize("text", ["not json", "[]", '{"paraphrases": 3}', '{"paraphrases": [1]}'])
def test_parse_rejects_bad_shapes(text):
    with pytest.raises(api.GenerationError):
        api.parse_response(text, "start_video_call", 0)


def test_check_seed_reports_rule_quota_and_cross_intent_errors():
    bad = [*CANNED[:-1], {"text": "video call mom", "variant": "disfluent"}]
    errors = api.check_seed(
        "start_video_call", 0, api.parse_response(canned(bad), "start_video_call", 0), SEEDS, {}
    )
    assert any("item 40" in e and "slots" in e for e in errors)
    short = api.parse_response(canned(CANNED[:30]), "start_video_call", 0)
    assert any("wrong counts" in e for e in api.check_seed("start_video_call", 0, short, SEEDS, {}))
    others = load_paraphrases(FIXTURE)
    others["start_video_call"] = {}
    others["plan_day_itinerary"][0] = (
        *others["plan_day_itinerary"][0],
        api.parse_response(canned(CANNED[:1]), "plan_day_itinerary", 0)[0],
    )
    items = api.parse_response(canned(), "start_video_call", 0)
    errors = api.check_seed("start_video_call", 0, items, SEEDS, others)
    assert any("paraphrase of plan_day_itinerary" in e for e in errors)


def test_generate_seed_retries_once_with_the_errors():
    prompts = []
    answers = iter([canned(CANNED[:39]), canned()])

    def call(prompt):
        prompts.append(prompt)
        return next(answers)

    items, errors = api.generate_seed(call, TEMPLATE, "start_video_call", 0, SEEDS, {})
    assert errors == [] and len(items) == 40
    assert len(prompts) == 2
    assert "failed validation" in prompts[1] and "disfluent 5/6" in prompts[1]
    assert prompts[1].startswith(prompts[0])
    assert [p.variant for p in items] == sorted(
        (p.variant for p in items), key=["plain", "hinglish", "kannada_english", "disfluent"].index
    )


def test_generate_seed_gives_up_after_one_retry():
    calls = []

    def call(prompt):
        calls.append(prompt)
        return "oops"

    items, errors = api.generate_seed(call, TEMPLATE, "start_video_call", 0, SEEDS, {})
    assert items == [] and errors and len(calls) == 2


# ---- request shape, with a fake client ------------------------------------ #


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, message):
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.message = message

    def _stream(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.message)


def _message(stop_reason="end_turn", text=None):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        stop_reason=stop_reason, content=content, stop_details=SimpleNamespace(category="bio")
    )


def test_call_model_request_shape():
    client = FakeClient(_message(text=canned()))
    assert api.call_model(client, "hello", "claude-opus-5") == canned()
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": api.OUTPUT_SCHEMA}
    assert kw["messages"] == [{"role": "user", "content": "hello"}]
    assert kw["fallbacks"] == "default" and kw["betas"] == [api.FALLBACK_BETA]
    api.call_model(client, "hello", "claude-sonnet-5", fallbacks=False)
    assert "fallbacks" not in client.kwargs and client.kwargs["model"] == "claude-sonnet-5"


@pytest.mark.parametrize("stop, match", [("refusal", "declined"), ("max_tokens", "cut off")])
def test_call_model_raises_on_refusal_and_truncation(stop, match):
    with pytest.raises(api.GenerationError, match=match):
        api.call_model(FakeClient(_message(stop, "{}")), "hi", "claude-opus-5")


# ---- CLI ------------------------------------------------------------------ #


def test_dry_run_prints_prompt_without_credentials(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert api.main(["--intents", "wipers_on", "--dry-run"]) == 0
    assert "`turn the wipers on`" in capsys.readouterr().out


def test_existing_file_needs_resume_or_overwrite(tmp_path):
    (tmp_path / "wipers_on.yaml").write_text("# empty\n")
    with pytest.raises(SystemExit):
        api.main(["--intents", "wipers_on", "--out", str(tmp_path)])


@pytest.mark.parametrize("how", ["no_profile", "unreadable_profile"])
def test_refuses_to_run_without_credentials(tmp_path, monkeypatch, how):
    pytest.importorskip("anthropic")
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    if how == "unreadable_profile":
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profiles"))
    with pytest.raises(SystemExit, match="no Anthropic credentials"):
        api.main(["--intents", "wipers_on", "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
