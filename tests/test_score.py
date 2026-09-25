"""autogate_eval.score and the rules baseline's --emit (no torch needed)."""

import json

import pytest

from autogate_eval.baselines.rules import detect_sensitive, detect_spans
from autogate_eval.baselines.rules import main as rules_main
from autogate_eval.score import format_score, join, score, span_scores
from autogate_eval.score import main as score_main


def _row(id_, route, utterance="", spans=(), split="test", intent="call_contact"):
    return {
        "id": id_,
        "route": route,
        "actuation": "none",
        "split": split,
        "intent": intent,
        "utterance": utterance,
        "spans": list(spans),
        "context_invariant": False,
        "context_flipped": True,
        "sensitive": bool(spans),
        "adversarial": False,
    }


NAME = {"start": 5, "end": 15, "label": "contact", "text": "ravi gowda"}


def test_join_keeps_row_order_and_rejects_unknown_ids():
    rows = [_row("a", "LOCAL"), _row("b", "CLOUD"), _row("c", "DEFER")]
    kept, preds = join(rows, [{"id": "c", "route": "DEFER"}, {"id": "a", "route": "LOCAL"}])
    assert [r["id"] for r in kept] == ["a", "c"] and [p["id"] for p in preds] == ["a", "c"]
    with pytest.raises(ValueError, match="not in the rows"):
        join(rows, [{"id": "zzz", "route": "LOCAL"}])
    with pytest.raises(ValueError, match="duplicate"):
        join(rows, [{"id": "a", "route": "LOCAL"}, {"id": "a", "route": "CLOUD"}])


def test_leak_rate_counts_uncovered_spans_on_cloud_routes():
    rows = [
        _row("a", "CLOUD_MASKED", "text ravi gowda hi", [NAME]),
        _row("b", "CLOUD_MASKED", "text ravi gowda hi", [NAME]),
        _row("c", "LOCAL", "text ravi gowda hi", [NAME]),
    ]
    preds = [
        {"id": "a", "route": "CLOUD_MASKED", "spans": [dict(NAME)]},  # covered exactly
        {"id": "b", "route": "CLOUD", "spans": [{**NAME, "end": 9, "text": "ravi"}]},  # partial
        {"id": "c", "route": "LOCAL", "spans": []},  # stays in the car: not at risk
    ]
    s = span_scores(rows, preds)
    assert s["spans_sent_to_cloud"] == 2 and s["leaked_spans"] == 1 and s["leak_rate"] == 0.5
    assert s["precision"] == 0.5 and s["recall"] == pytest.approx(1 / 3)


def test_score_reports_route_intent_and_spans():
    rows = [_row("a", "LOCAL"), _row("b", "REFUSE", split="val")]
    preds = [
        {"id": "a", "route": "LOCAL", "intent": "call_contact", "spans": []},
        {"id": "b", "route": "LOCAL", "intent": "take_note", "spans": []},
    ]
    result = score(rows, preds, n_boot=10)
    assert result["route"]["overall"]["critical_misses"] == 1
    assert result["intent"]["accuracy"] == 0.5
    assert result["intent"]["by_split"] == {"val": 0.0, "test": 1.0}
    assert "leak rate" in format_score(result)
    # route-only predictions skip the intent and span sections
    bare = score(rows, [{"id": "a", "route": "LOCAL"}], n_boot=10)
    assert bare["n_scored"] == 1 and "intent" not in bare and "spans" not in bare


def test_detect_spans_agrees_with_detect_sensitive(pilot_rows):
    for r in pilot_rows:
        spans = detect_spans(r["utterance"])
        if spans:
            assert detect_sensitive(r["utterance"])
        for s in spans:
            assert r["utterance"][s["start"] : s["end"]] == s["text"]
    assert detect_spans("dial +91 98450 12345 now") == [
        {"start": 5, "end": 20, "label": "phone", "text": "+91 98450 12345"}
    ]
    assert detect_spans("pay the card ending 4421")[0]["label"] == "card"


def test_rules_emit_is_scored(tmp_path, pilot_jsonl, capsys):
    out = tmp_path / "rules_preds.jsonl"
    rules_main(["--rows", str(pilot_jsonl), "--emit", str(out)])
    preds = [json.loads(line) for line in out.read_text().splitlines()]
    assert set(preds[0]) == {
        "id",
        "route",
        "route_probs",
        "intent",
        "intent_probs_top3",
        "spans",
        "confidence",
    }
    assert preds[0]["confidence"] is None and preds[0]["route_probs"] is None
    result = score_main(["--rows", str(pilot_jsonl), "--preds", str(out), "--n-boot", "20"])
    # same routes as the baseline's own report
    assert result["route"]["overall"]["accuracy"] == pytest.approx(0.958, abs=1e-3)
    assert result["intent"]["accuracy"] == pytest.approx(0.992, abs=1e-3)
    assert "scored 1182 of 1182 rows" in capsys.readouterr().out
