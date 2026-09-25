"""Rules baseline: intent matching, PII regexes, and valid routes."""

import pytest

from autogate_bench import Route
from autogate_bench.dataset import build_rows
from autogate_eval.baselines.rules import (
    RulesBaseline,
    content_tokens,
    detect_sensitive,
    evaluate_baseline,
    format_baseline,
    main,
    predict,
)

ROWS = build_rows(rng_seed=0)
BASELINE = RulesBaseline()


def test_content_tokens():
    assert content_tokens("Can you dial {phone} for me?") == {"dial"}
    assert content_tokens("what's the score in 2024") == {"whats", "score", "<num>"}


def test_intent_accuracy_on_seed_only_pilot():
    correct = sum(BASELINE.predict_intent(r) == r.intent for r in ROWS)
    assert correct / len(ROWS) >= 0.9


def test_predictions_are_valid_routes():
    preds = [predict(r) for r in ROWS]
    assert all(isinstance(p, Route) for p in preds)
    assert preds == [BASELINE.predict(r) for r in ROWS]
    # the dict form of a row works too
    assert predict(ROWS[0].to_dict()) == preds[0]


@pytest.mark.parametrize(
    "text",
    [
        "can you dial +91 98450 12345 for me",
        "dial 9845012345",
        "reply 482913",
        "quick note otp 4821",
        "remind me to pay the card ending 4421",
        "call Priya about dinner",
    ],
)
def test_pii_detector_catches(text):
    assert detect_sensitive(text)


@pytest.mark.parametrize(
    "text",
    [
        "set the temperature to 22",
        "tune to fm 93.5",
        "book a table for 4 at 8 15 pm",
        "call priya natarajan",  # lowercase names are missed by design
        "Call my wife",  # a sentence-initial capital is not a name
    ],
)
def test_pii_detector_passes(text):
    assert not detect_sensitive(text)


def test_evaluate_and_cli(tmp_path, capsys):
    from autogate_bench.dataset import write_jsonl

    result = evaluate_baseline(ROWS, BASELINE, n_boot=20)
    assert result["intent_accuracy"] >= 0.9
    assert result["pii_detection"]["precision"] == 1.0
    assert "intent accuracy" in format_baseline(result)

    path = tmp_path / "rows.jsonl"
    write_jsonl(ROWS, path)
    out = main(["--rows", str(path), "--train-only"])
    assert out["intent_accuracy_by_split"]["train"] == 1.0
    assert "| overall |" in capsys.readouterr().out
