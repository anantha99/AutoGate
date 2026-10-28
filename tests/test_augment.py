"""Train-only augmentation: loader, every validator rule, rows, byte identity, router labels."""

import copy
import json
import re
from pathlib import Path

import pytest
import yaml

from autogate_bench.augment import (
    GOALS_FILE,
    OOD_KEYWORDS,
    PAIRS_FILE,
    Augmentation,
    Reference,
    augment_rows,
    load_augmentation,
    main,
    ood_keyword_hits,
    routes_differ,
    validate,
)
from autogate_bench.contexts import ALL_CONTEXTS
from autogate_bench.dataset import (
    Row,
    build_rows,
    read_jsonl,
    read_parquet,
    summarize,
    write_jsonl,
    write_parquet,
)
from autogate_bench.generate import main as generate
from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.paraphrases import load_paraphrases
from autogate_bench.rulebook import route, route_tags
from autogate_bench.seeds import load_seeds
from autogate_bench.splits import OOD_INTENTS, TEST, VAL
from autogate_eval.baselines.rules import content_tokens

FIXTURE = Path(__file__).parent / "fixtures" / "augment"
PARA_FIXTURE = Path(__file__).parent / "fixtures" / "paraphrases"
SEEDS = load_seeds()
PARA = load_paraphrases()  # the real paraphrases: the leak guards compare against them
REF = Reference.build(SEEDS, PARA)
RAW_GOALS = yaml.safe_load((FIXTURE / GOALS_FILE).read_text())
RAW_PAIRS = yaml.safe_load((FIXTURE / PAIRS_FILE).read_text())
AUG = load_augmentation(FIXTURE)


def write_aug(tmp_path, goals=(), pairs=()) -> Augmentation:
    if goals:
        (tmp_path / GOALS_FILE).write_text(yaml.safe_dump(list(goals), sort_keys=False))
    if pairs:
        (tmp_path / PAIRS_FILE).write_text(yaml.safe_dump(list(pairs), sort_keys=False))
    return load_augmentation(tmp_path)


def goal(**changes) -> dict:
    g = copy.deepcopy(RAW_GOALS[0])
    g.update(changes)
    return g


def sunroof(**changes) -> dict:
    g = copy.deepcopy(RAW_GOALS[1])
    g.update(changes)
    return g


def pair(**changes) -> dict:
    p = copy.deepcopy(RAW_PAIRS[0])
    for k, v in changes.items():
        if k in ("a", "b"):
            p[k] = {**p[k], **v}
        else:
            p[k] = v
    return p


def errors(tmp_path, goals=(), pairs=(), strict=True) -> list[str]:
    rep = validate(write_aug(tmp_path, goals, pairs), REF, strict=strict)
    return [i.message for i in rep.errors]


def has(msgs, fragment) -> bool:
    return any(fragment in m for m in msgs)


# --------------------------------------------------------------------------- #
# The fixture
# --------------------------------------------------------------------------- #


def test_fixture_passes_strict_validation():
    assert len(AUG.goals) == 2 and len(AUG.pairs) == 3
    rep = validate(AUG, REF, strict=True)
    assert rep.errors == [], [e.format() for e in rep.errors]


def test_loader_records_lines_and_fields():
    g = AUG.goals[0]
    assert g.name == "start_karaoke_session" and g.group == "media"
    assert g.line > 0 and all(u.line > g.line for u in g.utterances)
    p = AUG.pairs[0]
    assert p.differs_in == ("actuation",) and p.a.intent == "adjust_window"
    assert p.a.utterance.line > 0


def test_missing_directory_is_empty(tmp_path):
    assert not load_augmentation(tmp_path / "nope")
    assert not load_augmentation(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "name: x\n",  # not a list
        "- {name: x}\n",  # missing keys
        "- 3\n",
    ],
)
def test_loader_rejects_malformed_goals(tmp_path, body):
    (tmp_path / GOALS_FILE).write_text(body)
    with pytest.raises(ValueError):
        load_augmentation(tmp_path)


def test_loader_rejects_malformed_pairs(tmp_path):
    bad = pair()
    del bad["note"]
    (tmp_path / PAIRS_FILE).write_text(yaml.safe_dump([bad]))
    with pytest.raises(ValueError, match="note"):
        load_augmentation(tmp_path)
    bad = pair(differs_in="actuation")
    (tmp_path / PAIRS_FILE).write_text(yaml.safe_dump([bad]))
    with pytest.raises(ValueError, match="differs_in"):
        load_augmentation(tmp_path)


# --------------------------------------------------------------------------- #
# Goal rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"name": "StartKaraoke"}, "not snake_case"),
        ({"name": "play_video"}, "is an existing intent"),
        ({"group": "entertainment"}, "unknown group"),
        ({"capability": "maybe_cloud"}, "unknown capability"),
        ({"distraction": "extreme"}, "unknown distraction"),
        ({"actuation": "dangerous"}, "unknown actuation"),
        ({"consequence": "permanent"}, "unknown consequence"),
        ({"actuation": "comfort"}, "only vehicle_control intents"),  # media group
        ({"description": " "}, "empty description"),
        ({"rationale": ""}, "empty rationale"),
    ],
)
def test_goal_field_errors(tmp_path, changes, fragment):
    assert has(errors(tmp_path, [goal(**changes)]), fragment)


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"actuation": "none"}, "need an actuation class"),
        ({"capability": "needs_cloud"}, "must be local_ok"),
        ({"distraction": "medium"}, "must be low distraction"),
        ({"consequence": "none"}, "always has a side effect"),
    ],
)
def test_vehicle_goal_invariants(tmp_path, changes, fragment):
    assert has(errors(tmp_path, [sunroof(**changes)]), fragment)


def test_duplicate_goal_name(tmp_path):
    other = sunroof(name="start_karaoke_session")
    assert has(errors(tmp_path, [goal(), other]), "duplicate goal name")


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("Start karaoke now", "uppercase"),
        ("start karaoke, now", "disallowed punctuation"),
        ("start karaoke café", "non-ASCII"),
        (" ".join(["karaoke"] * 25), "25 words"),
        ("karaoke for {singer}", "unknown slot {singer}"),
        ("karaoke with rahul", "literal name rahul"),
        ("karaoke on channel 12345", "4+ digits"),
        ("start karaoke ?", "space before the question mark"),
    ],
)
def test_goal_text_rules(tmp_path, text, fragment):
    g = goal()
    g["utterances"][0]["text"] = text
    assert has(errors(tmp_path, [g]), fragment)


@pytest.mark.parametrize(
    "field_, value, fragment",
    [
        ("style", "rhetorical", "unknown style"),
        ("variant", "tamil_english", "unknown variant"),
        ("variant", "asr", "unknown variant"),
    ],
)
def test_goal_style_and_variant(tmp_path, field_, value, fragment):
    g = goal()
    g["utterances"][0][field_] = value
    assert has(errors(tmp_path, [g]), fragment)


def test_strict_quota_is_an_error_only_when_strict(tmp_path):
    g = goal()
    g["utterances"] = [u for u in g["utterances"] if u["style"] == "command"][:5]
    for u in g["utterances"]:
        u["variant"] = "plain"
    rep = validate(write_aug(tmp_path, [g]), REF, strict=True)
    msgs = [i.message for i in rep.errors]
    assert has(msgs, "5 utterances (want 8 to 12)")
    assert has(msgs, "1 styles (want at least 3)")
    assert has(msgs, "no hinglish or kannada_english")
    loose = validate(write_aug(tmp_path, [g]), REF, strict=False)
    assert not loose.errors
    assert has([i.message for i in loose.warnings], "5 utterances")


def test_too_many_utterances(tmp_path):
    g = goal()
    extra = [
        {"text": f"karaoke please {w}", "style": "fragment", "variant": "plain"}
        for w in ("alpha", "bravo")
    ]
    g["utterances"] += extra
    assert has(errors(tmp_path, [g]), "13 utterances")


# --------------------------------------------------------------------------- #
# Leak guards
# --------------------------------------------------------------------------- #


def test_ood_keyword_table_covers_every_ood_intent():
    assert set(OOD_KEYWORDS) == set(OOD_INTENTS)
    # the distinctive words of each OOD intent's name are caught
    for name in OOD_INTENTS:
        assert ood_keyword_hits(name.replace("_", " ")), name
    assert ood_keyword_hits("open the fuel flap") and ood_keyword_hits("avoid tolls")
    assert ood_keyword_hits("read me the headlines")
    assert not ood_keyword_hits("start a karaoke session")
    assert not ood_keyword_hits("open the sunroof")


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "toggle_fog_lamps"},
        {"description": "Play the news on the speakers"},
        {"name": "pair_headphones"},
        {"description": "Work out the trip cost to a city"},
        {"name": "set_ambient_glow"},
    ],
)
def test_goal_naming_an_ood_goal_is_rejected(tmp_path, changes):
    assert has(errors(tmp_path, [goal(**changes)]), "names an OOD goal")


def _close_variant(text: str, limit: float, above: bool = True) -> str:
    """``text`` plus one new word, so its token Jaccard with ``text`` is above ``limit``."""
    out = text + " right now"
    a, b = content_tokens(out), content_tokens(text)
    j = len(a & b) / len(a | b)
    assert (j > limit) == above, (text, j)
    return out


def test_goal_too_close_to_an_ood_utterance(tmp_path):
    seed = next(s.text for s in SEEDS["route_avoiding_tolls"] if len(content_tokens(s.text)) >= 3)
    g = goal()
    g["utterances"][0]["text"] = _close_variant(seed, 0.5)
    assert has(errors(tmp_path, [g]), "with OOD intent route_avoiding_tolls")


def _valtest_seed(min_tokens: int = 4) -> str:
    for intent, items in SEEDS.items():
        if intent in OOD_INTENTS:
            continue
        for i, s in enumerate(items):
            if REF.splits[f"{intent}/{i}"] in (VAL, TEST) and len(content_tokens(s.text)) >= 4:
                return s.text
    raise AssertionError("no val/test seed")


def test_goal_too_close_to_a_val_or_test_utterance(tmp_path):
    g = goal()
    g["utterances"][0]["text"] = _close_variant(_valtest_seed(), 0.7)
    assert has(errors(tmp_path, [g]), "> 0.7 with a")


def test_pair_too_close_to_a_val_or_test_utterance(tmp_path):
    text = _close_variant(_valtest_seed(), 0.7)
    msgs = errors(tmp_path, pairs=[pair(a={"text": text})])
    assert has(msgs, "> 0.7 with a")


def test_exact_duplicates(tmp_path):
    g = goal()
    g["utterances"][0]["text"] = SEEDS["play_video"][0].text
    g["utterances"][1]["text"] = PARA["play_music"][0][0].text
    g["utterances"][2]["text"] = g["utterances"][3]["text"]
    msgs = errors(tmp_path, [g])
    assert has(msgs, "identical to a seed of play_video")
    assert has(msgs, "identical to a paraphrase of play_music")
    assert has(msgs, "duplicate of goal start_karaoke_session utterance 2")


def test_duplicate_between_a_goal_and_a_pair(tmp_path):
    p = pair(a={"text": RAW_GOALS[0]["utterances"][0]["text"]})
    assert has(errors(tmp_path, [goal()], [p]), "duplicate of goal start_karaoke_session")


# --------------------------------------------------------------------------- #
# Pair rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"a": {"intent": "open_sunroof"}}, "a: unknown intent"),
        ({"b": {"intent": "open_fuel_door"}}, "b: open_fuel_door is an OOD intent"),
        ({"b": {"intent": "adjust_window"}}, "both sides are adjust_window"),
        ({"differs_in": ["actuation", "consequence"]}, "differs_in [actuation, consequence]"),
        ({"differs_in": ["colour"]}, "unknown tag in differs_in"),
        ({"b": {"text": "please unlock everything at the back"}}, "do not look alike"),
        ({"id": "Rear-Window"}, "not snake_case"),
        ({"note": ""}, "empty note"),
    ],
)
def test_pair_errors(tmp_path, changes, fragment):
    assert has(errors(tmp_path, pairs=[pair(**changes)]), fragment)


def test_duplicate_pair_id(tmp_path):
    other = copy.deepcopy(RAW_PAIRS[1])
    other["id"] = RAW_PAIRS[0]["id"]
    assert has(errors(tmp_path, pairs=[pair(), other]), "duplicate pair id")


def test_pair_that_never_changes_the_route_is_rejected(tmp_path):
    # set vs cancel cruise: comfort vs safety_critical, but both are always LOCAL
    p = {
        "id": "cruise_set_vs_cancel_01",
        "differs_in": ["actuation"],
        "a": {
            "intent": "set_cruise_speed",
            "text": "set the cruise to {speed} on this stretch",
            "style": "command",
            "variant": "plain",
        },
        "b": {
            "intent": "cancel_cruise_control",
            "text": "cancel the cruise on this stretch",
            "style": "command",
            "variant": "plain",
        },
        "note": "looks like a hard negative but is not one for routing",
    }
    msgs = errors(tmp_path, pairs=[p])
    assert msgs == [
        "set_cruise_speed and cancel_cruise_control route the same in every context: "
        "not a hard negative"
    ]


def test_routes_differ():
    i = INTENTS_BY_NAME
    assert routes_differ(i["adjust_window"], i["unlock_doors"])
    assert not routes_differ(i["set_cruise_speed"], i["cancel_cruise_control"])
    assert not routes_differ(i["unlock_doors"], i["disable_child_locks"])


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #

FIX_PARA = load_paraphrases(PARA_FIXTURE)
PLAIN = build_rows(rng_seed=0, paraphrases=FIX_PARA)
AUGMENTED = build_rows(rng_seed=0, paraphrases=FIX_PARA, augment=AUG)
AUG_ROWS = [r for r in AUGMENTED if r.is_augmented]
GOALS = {g.name: g for g in AUG.goals}


def test_augmentation_only_appends_rows():
    assert AUGMENTED[: len(PLAIN)] == PLAIN
    assert AUGMENTED[len(PLAIN) :] == AUG_ROWS and AUG_ROWS
    assert {r.split for r in AUG_ROWS} == {"train"}
    assert len({r.id for r in AUGMENTED}) == len(AUGMENTED)


@pytest.mark.parametrize("paraphrases", [None, "fixture"])
def test_jsonl_without_augmentation_is_a_byte_prefix(tmp_path, paraphrases):
    para = FIX_PARA if paraphrases else None
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(build_rows(rng_seed=0, paraphrases=para), a)
    write_jsonl(build_rows(rng_seed=0, paraphrases=para, augment=AUG), b)
    plain, aug = a.read_bytes(), b.read_bytes()
    assert aug.startswith(plain) and len(aug) > len(plain)
    # every non-train row and every existing train row is byte-identical
    new_lines = aug[len(plain) :].decode().splitlines()
    assert all(json.loads(line)["split"] == "train" for line in new_lines)
    assert all(
        json.loads(line)["source"] in ("synthetic_goal", "minimal_pair") for line in new_lines
    )


def test_generate_cli_without_augment_is_unchanged(tmp_path):
    generate(["--out", str(tmp_path / "a"), "--rng-seed", "0"])
    m = generate(["--out", str(tmp_path / "b"), "--rng-seed", "0", "--augment", str(FIXTURE)])
    plain = (tmp_path / "a" / "rows.jsonl").read_bytes()
    assert (tmp_path / "b" / "rows.jsonl").read_bytes().startswith(plain)
    assert m["args"]["augment"].endswith("tests/fixtures/augment")
    assert m["augment"]["synthetic_goals"] == 2 and m["augment"]["minimal_pairs"] == 3
    assert m["augment"]["split"] == {"train": m["augment"]["rows"]}
    m0 = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert "augment" not in m0 and "augment" not in m0["args"]
    assert set(m0["source"]) == {"seed", "paraphrase", "asr"}


def test_generate_cli_rejects_missing_augment_dir(tmp_path):
    with pytest.raises(SystemExit):
        generate(["--out", str(tmp_path / "x"), "--augment", str(tmp_path / "none")])


def test_generation_refuses_invalid_augmentation(tmp_path):
    bad = write_aug(tmp_path, [goal(name="play_video")])
    with pytest.raises(ValueError, match="augmentation errors"):
        build_rows(rng_seed=0, augment=bad)


def test_goal_rows_are_labelled_from_their_tags():
    goal_rows = [r for r in AUG_ROWS if r.source == "synthetic_goal"]
    assert {r.intent for r in goal_rows} == set(GOALS)
    for r in goal_rows:
        g = GOALS[r.intent]
        assert r.is_synthetic and r.pair_id is None
        assert r.seed_id == f"goal:{g.name}" and r.id.startswith(f"goal:{g.name}/u")
        assert (r.capability, r.distraction, r.actuation, r.consequence) == (
            g.capability,
            g.distraction,
            g.actuation,
            g.consequence,
        )
        assert r.group == g.group
        d = route_tags(g.capability, g.distraction, g.actuation, r.context(), r.has_sensitive_spans)
        assert (r.route, r.reason, r.demand) == (str(d.route), str(d.reason), str(d.demand))
        assert "{" not in r.utterance
        for s in r.spans:
            assert r.utterance[s.start : s.end] == s.text
    # the karaoke goal is high distraction: refused somewhere, local somewhere
    karaoke = {r.route for r in goal_rows if r.intent == "start_karaoke_session"}
    assert {"REFUSE", "LOCAL"} <= karaoke


def test_pair_rows_are_labelled_as_their_intent():
    pair_rows = [r for r in AUG_ROWS if r.source == "minimal_pair"]
    assert {r.pair_id for r in pair_rows} == {p.id for p in AUG.pairs}
    for r in pair_rows:
        intent = INTENTS_BY_NAME[r.intent]
        assert not r.is_synthetic and r.seed_id == f"pair:{r.pair_id}"
        assert r.actuation == str(intent.actuation) and r.distraction == str(intent.distraction)
        d = route(intent, r.context(), r.has_sensitive_spans)
        assert (r.route, r.reason, r.demand) == (str(d.route), str(d.reason), str(d.demand))
    by_side = {(r.pair_id, r.id.split("/")[1].split("-")[0]) for r in pair_rows}
    assert len(by_side) == 2 * len(AUG.pairs)


def test_augmented_rows_use_three_contexts_and_no_asr():
    assert not any(r.variant == "asr" for r in AUG_ROWS)
    per_utt: dict = {}
    for r in AUG_ROWS:
        per_utt.setdefault(r.id.rsplit("-", 1)[0], []).append(r)
    assert all(len(rs) == 3 for rs in per_utt.values())
    # route-boundary-weighted: a flipping utterance lands on at least two routes
    for rs in per_utt.values():
        if rs[0].context_flipped:
            assert len({r.route for r in rs}) >= 2


def test_augmented_rows_round_trip(tmp_path):
    p = tmp_path / "rows.jsonl"
    write_jsonl(AUGMENTED, p)
    assert read_jsonl(p) == AUGMENTED
    pq = tmp_path / "rows.parquet"
    write_parquet(AUGMENTED, pq)
    assert read_parquet(pq) == AUGMENTED
    r = AUG_ROWS[0]
    assert Row.from_dict(r.to_dict()) == r
    assert "is_synthetic" not in PLAIN[0].to_dict()


def test_summary_reports_augmentation():
    s = summarize(AUGMENTED)
    assert s["source"]["synthetic_goal"] + s["source"]["minimal_pair"] == len(AUG_ROWS)
    assert s["augment"]["synthetic_goals"] == 2 and s["augment"]["minimal_pairs"] == 3
    assert "augment" not in summarize(PLAIN)


def test_augment_rows_directly_matches_build_rows():
    rows = augment_rows(AUG, load_seeds(), FIX_PARA, rng_seed=0)
    assert rows == AUG_ROWS


# --------------------------------------------------------------------------- #
# Router labels, rules baseline, scorer
# --------------------------------------------------------------------------- #


class _WhitespaceTokenizer:
    """Just enough of a fast tokenizer for ``encode_row``."""

    def __call__(self, text, **_):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_synthetic_rows_ignore_the_intent_head():
    from autogate_router.data import encode_row
    from autogate_router.labels import DEFAULT_LABELS, IGNORE_INDEX

    tok = _WhitespaceTokenizer()
    for r in AUG_ROWS:
        ex = encode_row(r.to_dict(), tok)
        if r.is_synthetic:
            assert ex["intent_label"] == IGNORE_INDEX
        else:
            assert ex["intent_label"] == DEFAULT_LABELS.intent_index[r.intent]
        assert all(v != IGNORE_INDEX for v in ex["tag_labels"].values())
        assert ex["route_label"] == DEFAULT_LABELS.route_index[r.route]
    # a synthetic row that happens to reuse an intent's name is still ignored
    row = {**AUG_ROWS[0].to_dict(), "intent": "play_video"}
    assert encode_row(row, tok)["intent_label"] == IGNORE_INDEX


def test_rules_baseline_and_scorer_accept_augmented_rows(tmp_path, capsys):
    from autogate_eval.baselines.rules import main as rules_main
    from autogate_eval.score import main as score_main

    rows = tmp_path / "rows.jsonl"
    write_jsonl(AUGMENTED, rows)
    preds = tmp_path / "preds.jsonl"
    result = rules_main(["--rows", str(rows), "--train-only", "--emit", str(preds)])
    assert result["intent_accuracy_by_split"]["train"] is not None
    out = score_main(["--rows", str(rows), "--preds", str(preds), "--n-boot", "5"])
    assert out["route"]["overall"]["n"] == len(AUGMENTED)
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_validate_stats_review(tmp_path, capsys):
    assert main(["validate", "--dir", str(FIXTURE), "--strict"]) == 0
    assert "0 errors" in capsys.readouterr().out
    assert main(["stats", "--dir", str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "distraction  high" in out and "pairs per differs_in tag" in out
    review = tmp_path / "review.md"
    assert main(["review", "--dir", str(FIXTURE), "--out", str(review)]) == 0
    text = review.read_text()
    assert "### start_karaoke_session" in text and "### voice_vs_video_call_01" in text
    assert "nearest existing intent" in text and "val/test" in text


def test_cli_validate_fails_on_errors_and_handles_empty(tmp_path, capsys):
    write_aug(tmp_path, [goal(name="play_video")])
    assert main(["validate", "--dir", str(tmp_path)]) == 1
    assert "is an existing intent" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["validate", "--dir", str(empty)]) == 0
    assert "no augmentation files" in capsys.readouterr().out


def test_every_context_is_checked_for_route_differences():
    i = INTENTS_BY_NAME
    diff = routes_differ(i["call_contact"], i["start_video_call"])
    assert 0 < len(diff) < 2 * len(ALL_CONTEXTS)
