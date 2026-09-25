"""Paraphrase files: loader, validator rules, CLI, and the test fixture."""

from pathlib import Path

import pytest

from autogate_bench.paraphrases import (
    PER_SEED,
    QUOTA,
    Paraphrase,
    Variant,
    format_paraphrase_file,
    load_paraphrase_file,
    load_paraphrases,
    main,
    normalize,
    status_lines,
    validate,
)
from autogate_bench.seeds import load_seeds

SEEDS = load_seeds()
FIXTURE = Path(__file__).parent / "fixtures" / "paraphrases"


def P(intent, index, text, variant="plain", line=1):
    return Paraphrase(intent, index, text, variant, line)


def errors_for(*items, strict=False):
    by: dict = {}
    for p in items:
        by.setdefault(p.intent, {}).setdefault(p.seed_index, []).append(p)
    rep = validate({i: {k: tuple(v) for k, v in s.items()} for i, s in by.items()}, SEEDS, strict)
    return rep


def messages(rep, level="errors"):
    return " | ".join(i.message for i in getattr(rep, level))


# ---- constants ------------------------------------------------------------ #


def test_quota_is_forty():
    assert QUOTA == {
        Variant.PLAIN: 24,
        Variant.HINGLISH: 6,
        Variant.KANNADA_ENGLISH: 4,
        Variant.DISFLUENT: 6,
    }
    assert PER_SEED == 40


def test_paraphrase_slots_match_seed_extraction():
    p = P("call_contact", 0, "call {contact} no wait call {contact} on {phone}")
    assert p.slots == ("contact", "contact", "phone")


def test_normalize():
    assert normalize("  What's   the time? ") == "whats the time"
    assert normalize("what’s the time") == "whats the time"


# ---- fixture -------------------------------------------------------------- #


def test_fixture_passes_strict_validation():
    rep = validate(load_paraphrases(FIXTURE), SEEDS, strict=True)
    assert rep.errors == []
    assert set(rep.counts) == {"start_video_call", "plan_day_itinerary"}
    for intent, counts in rep.counts.items():
        n = len(SEEDS[intent])
        assert dict(counts) == {str(v): q * n for v, q in QUOTA.items()}


def test_fixture_round_trips_through_the_formatter(tmp_path):
    for path in FIXTURE.glob("*.yaml"):
        by_seed = load_paraphrase_file(path)
        text = format_paraphrase_file(path.stem, by_seed, SEEDS[path.stem])
        assert text == path.read_text(encoding="utf-8")


def test_loader_records_lines_and_intent():
    by_seed = load_paraphrase_file(FIXTURE / "start_video_call.yaml")
    first = by_seed[0][0]
    assert first.intent == "start_video_call" and first.seed_index == 0
    lines = (FIXTURE / "start_video_call.yaml").read_text().splitlines()
    assert first.text in lines[first.line - 1]


# ---- loader errors -------------------------------------------------------- #


def _write(tmp_path, body, name="call_contact.yaml"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_loader_accepts_int_keys_and_empty_file(tmp_path):
    p = _write(tmp_path, '0:\n  - {text: "ring {contact}", variant: plain}\n')
    assert load_paraphrase_file(p) == {0: (P("call_contact", 0, "ring {contact}"),)}
    assert load_paraphrase_file(_write(tmp_path, "# nothing yet\n")) == {}


def test_missing_directory_is_empty(tmp_path):
    assert load_paraphrases(tmp_path / "nope") == {}


@pytest.mark.parametrize(
    "body",
    [
        "- just a list\n",
        '"x":\n  - {text: "ring {contact}", variant: plain}\n',
        '"0": {text: "ring {contact}", variant: plain}\n',
        '"0":\n  - ring {contact}\n',
        '"0":\n  - {text: "ring {contact}"}\n',
        '"0":\n  - {text: "ring {contact}", variant: plain, extra: 1}\n',
        '"0":\n  - {text: 12, variant: plain}\n',
        '"0":\n  - {text: "a", variant: plain}\n"0":\n  - {text: "b", variant: plain}\n',
    ],
)
def test_loader_rejects_malformed(tmp_path, body):
    with pytest.raises(ValueError):
        load_paraphrase_file(_write(tmp_path, body))


# ---- validator errors ----------------------------------------------------- #


def test_clean_line_has_no_errors():
    rep = errors_for(P("call_contact", 0, "give {contact} a ring"))
    assert rep.errors == []


@pytest.mark.parametrize(
    "p, fragment",
    [
        (P("order_pizza", 0, "order a pizza"), "unknown intent"),
        (P("call_contact", 9, "give {contact} a ring"), "unknown seed index"),
        (P("call_contact", 0, "give {contact} a ring", "tamil"), "unknown variant"),
        (P("call_contact", 0, "give {contact} a ring", "asr"), "made by code"),
        (P("call_contact", 0, "give them a ring"), "slots"),
        (P("call_contact", 0, "give {contact} a ring on {phone}"), "slots"),
        (P("call_contact", 0, "give {person} a ring"), "slots"),
        (P("call_contact", 0, "give {contact a ring"), "unbalanced"),
        (P("call_contact", 0, "give {contact} " + "a " * 23 + "ring"), "words"),
        (P("call_contact", 0, "give {contact} a ring naé"), "non-ASCII"),
        (P("call_contact", 0, "Give {contact} a ring"), "uppercase"),
        (P("call_contact", 0, "give {contact} a ring."), "punctuation"),
        (P("call_contact", 0, "give {contact}, a ring"), "punctuation"),
        (P("call_contact", 0, "give {contact} a ring? now"), "punctuation"),
        (P("call_contact", 0, "give {contact} a ring ?"), "space before"),
        (P("call_contact", 0, "give  {contact} a ring"), "double space"),
        (P("call_contact", 0, "give {contact} a ring at 98450"), "4+ digits"),
        (P("call_contact", 0, "give {contact} and rahul a ring"), "literal name"),
        (P("call_contact", 0, "ring {contact}"), "identical to a seed of call_contact"),
        (P("call_contact", 0, "take me home"), "identical to a seed of navigate_home"),
        (P("call_contact", 0, ""), "empty"),
    ],
)
def test_error_rules(p, fragment):
    assert fragment in messages(errors_for(p))


def test_trailing_question_mark_and_apostrophe_are_allowed():
    rep = errors_for(P("lookup_contact_info", 1, "what's the number for {contact}?"))
    assert rep.errors == []


def test_repeated_slot_is_allowed_in_a_restart():
    rep = errors_for(P("call_contact", 0, "call {contact} no wait call {contact}", "disfluent"))
    assert rep.errors == []


def test_duplicate_within_intent_after_normalising():
    rep = errors_for(
        P("call_contact", 0, "give {contact} a ring", line=3),
        P("call_contact", 2, "give  {contact}  a ring?", line=9),
    )
    assert "duplicate of line 3" in messages(rep)


def test_duplicate_across_intents():
    rep = errors_for(
        P("wipers_on", 0, "clear the rain off"), P("defrost_windshield", 0, "clear the rain off")
    )
    assert sum("paraphrase of" in e.message for e in rep.errors) == 2


def test_strict_quota():
    items = [P("call_contact", 0, f"give {{contact}} a ring number {w}") for w in ("one", "two")]
    rep = errors_for(*items, strict=True)
    assert "quota not met: plain 2/24" in messages(rep)
    assert {e.seed_index for e in rep.errors if "quota" in e.message} == set(range(5))
    assert "quota" not in messages(errors_for(*items))


# ---- validator warnings --------------------------------------------------- #


def test_near_duplicate_of_seed_and_of_sibling():
    rep = errors_for(
        P("call_contact", 0, "call {contact} now", line=1),
        P("call_contact", 0, "phone {contact} straight away please", line=2),
        P("call_contact", 0, "phone {contact} straight away", line=3),
    )
    warns = messages(rep, "warnings")
    assert "near-duplicate of the seed" in warns
    assert "near-duplicate of line 2" in warns


def test_intent_drift_warning():
    rep = errors_for(P("call_contact", 0, "wipers on full for {contact}"))
    assert "possible intent drift: closer to wipers_on" in messages(rep, "warnings")


def test_low_shape_diversity_warning():
    items = [
        P("wipers_on", 0, t, line=i)
        for i, t in enumerate(
            [
                "switch the wipers on",
                "switch on the wiper",
                "switch wipers to fast",
                "wipe the glass",
            ]
        )
    ]
    assert "low shape diversity" in messages(errors_for(*items), "warnings")
    items[2] = P("wipers_on", 0, "put the wipers on fast", line=2)
    items[3] = P("wipers_on", 0, "get the wipers going", line=3)
    items.append(P("wipers_on", 0, "start wiping the windscreen", line=4))
    assert "low shape diversity" not in messages(errors_for(*items), "warnings")


def test_style_warnings():
    warns = messages(errors_for(P("wipers_on", 0, "can you get the wipers going?")), "warnings")
    assert "'?' on a command seed" in warns
    warns = messages(errors_for(P("wipers_on", 0, "uh could you get the wipers going")), "warnings")
    assert "reads as a question on a command seed" in warns
    ok = errors_for(P("unlock_doors", 1, "could you let my friend in at the back"))
    assert "question" not in messages(ok, "warnings")
    long = P("wipers_on", 3, "wipers on the fastest setting right now please")
    assert "long for a fragment" in messages(errors_for(long), "warnings")
    assert "literal digit" in messages(errors_for(P("wipers_on", 0, "wipers to 2")), "warnings")


def test_warnings_can_be_skipped():
    rep = validate(
        {"call_contact": {0: (P("call_contact", 0, "call {contact} now"),)}},
        SEEDS,
        check_warnings=False,
    )
    assert rep.warnings == []


# ---- CLI ------------------------------------------------------------------ #


def test_cli_validate_fixture(capsys):
    assert main(["validate", "--dir", str(FIXTURE), "--strict"]) == 0
    out = capsys.readouterr().out
    assert "start_video_call" in out and "0 errors" in out
    assert "start_video_call.yaml:" in out  # warnings are located file:seed:line


def test_cli_validate_fails_on_errors(tmp_path, capsys):
    _write(tmp_path, '"0":\n  - {text: "Ring {contact}", variant: asr}\n')
    assert main(["validate", "--dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{tmp_path}/call_contact.yaml:0:2: error:" in out


def test_cli_validate_intents_filter(tmp_path, capsys):
    _write(tmp_path, '"0":\n  - {text: "Ring {contact}", variant: plain}\n')
    _write(tmp_path, '"0":\n  - {text: "get the wipers going", variant: plain}\n', "wipers_on.yaml")
    assert main(["validate", "--dir", str(tmp_path), "--intents", "wipers_on"]) == 0
    assert main(["validate", "--dir", str(tmp_path), "--intents", "fuel_range"]) == 1


def test_status(capsys):
    lines = status_lines(load_paraphrases(FIXTURE), SEEDS)
    by = {ln.split()[0]: ln for ln in lines if ln}
    assert by["start_video_call"].endswith("complete")
    assert "no file" in by["call_contact"]
    assert lines[-1].startswith("2/65 intents complete, 320/")
    assert main(["status", "--dir", str(FIXTURE)]) == 0
    assert "2/65 intents complete" in capsys.readouterr().out
