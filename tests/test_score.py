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
    assert set(result["spans"]["by_split"]) == {"val", "test"}
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


# --------------------------------------------------------------------------- #
# Tag report
# --------------------------------------------------------------------------- #

# speed high + workload low = MEDIUM demand: restricted actuation and high
# distraction are refused, medium distraction is allowed.
HIGHWAY = dict(
    speed_bucket="high",
    connectivity="good",
    driver_workload="low",
    privacy_mode="standard",
    passenger_present=False,
    local_model_tier="tiny",
)


def _tag_row(id_, route, split, tier="tiny", **tags):
    return {
        **_row(id_, route, split=split),
        **HIGHWAY,
        "local_model_tier": tier,
        "actuation": tags["act"],
        "capability": tags["cap"],
        "distraction": tags["dis"],
        "consequence": tags["cons"],
    }


def _tags(cap, dis, act, cons):
    return {"capability": cap, "distraction": dis, "actuation": act, "consequence": cons}


def _tag_case():
    rows = [
        _tag_row(
            "1", "REFUSE", "test", cap="local_ok", dis="low", act="restricted", cons="reversible"
        ),
        _tag_row("2", "REFUSE", "test", cap="needs_cloud", dis="high", act="none", cons="none"),
        _tag_row("3", "LOCAL", "test", "small", cap="local_ok", dis="low", act="none", cons="none"),
        _tag_row(
            "4", "CLOUD", "ood", cap="needs_cloud", dis="low", act="none", cons="irreversible"
        ),
        _tag_row(
            "5", "LOCAL", "ood", cap="local_ok", dis="low", act="safety_critical", cons="reversible"
        ),
    ]
    preds = [
        # restricted read as comfort: dangerous miss; tag route LOCAL, corrected REFUSE
        _tags("local_ok", "low", "comfort", "reversible"),
        # high read as medium: dangerous miss; tag route CLOUD, corrected REFUSE
        _tags("needs_cloud", "medium", "none", "irreversible"),
        # capability wrong, but needs_small_local on a small tier is still LOCAL
        _tags("needs_small_local", "low", "none", "none"),
        # irreversible read as reversible: dangerous miss, never route-changing
        _tags("needs_cloud", "low", "none", "reversible"),
        # safety_critical read as restricted: not a dangerous miss, but REFUSE instead of LOCAL
        _tags("local_ok", "low", "restricted", "reversible"),
    ]
    return rows, [
        {"id": r["id"], "route": r["route"], "spans": [], "tags": t}
        for r, t in zip(rows, preds, strict=True)
    ]


def test_tag_report_known_numbers():
    rows, preds = _tag_case()
    result = score(rows, preds, n_boot=10)
    tags = result["tags"]["overall"]
    act = tags["actuation"]
    assert act["n"] == 5 and act["accuracy"] == pytest.approx(0.6)
    assert act["macro_f1"] == pytest.approx(0.25)  # none 1.0; comfort, restricted, sc 0.0
    assert (act["dangerous_misses"], act["n_dangerous"], act["dangerous_miss_rate"]) == (1, 2, 0.5)
    assert act["route_changing_errors"] == 2
    assert act["route_changing_cost"] == 6 + 5  # REFUSE->LOCAL, safety-critical LOCAL->REFUSE
    assert act["route_changing_cost_if_corrected"] == 0
    dis = tags["distraction"]
    assert dis["accuracy"] == pytest.approx(0.8) and dis["macro_f1"] == pytest.approx(1 / 3)
    assert dis["dangerous_miss_rate"] == 1.0
    assert (dis["route_changing_errors"], dis["route_changing_cost"]) == (1, 10)  # REFUSE->CLOUD
    cap = tags["capability"]
    assert cap["accuracy"] == pytest.approx(0.8) and cap["dangerous_miss_rate"] is None
    assert cap["route_changing_errors"] == 0
    cons = tags["consequence"]
    assert cons["accuracy"] == pytest.approx(0.6) and cons["dangerous_miss_rate"] == 1.0
    assert cons["route_changing_errors"] == 0

    test, ood = result["tags"]["by_split"]["test"], result["tags"]["by_split"]["ood"]
    assert test["actuation"]["accuracy"] == pytest.approx(2 / 3)
    assert test["actuation"]["route_changing_cost"] == 6
    assert ood["actuation"]["dangerous_miss_rate"] == 0.0  # restricted is still dangerous
    assert (ood["actuation"]["route_changing_errors"], ood["actuation"]["route_changing_cost"]) == (
        1,
        5,
    )
    assert ood["consequence"]["dangerous_misses"] == 1
    text = format_score(result)
    assert "| split: ood | actuation | 2 | 0.500 |" in text
    assert "route-changing errors" in text


def test_tag_report_absent_without_tags_and_in_json(tmp_path):
    rows, preds = _tag_case()
    plain = score(rows, [{k: v for k, v in p.items() if k != "tags"} for p in preds], n_boot=10)
    assert "tags" not in plain and "route-changing" not in format_score(plain)
    rows_path, preds_path = tmp_path / "rows.jsonl", tmp_path / "preds.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    preds_path.write_text("".join(json.dumps(p) + "\n" for p in preds))
    out = tmp_path / "score.json"
    score_main(
        ["--rows", str(rows_path), "--preds", str(preds_path), "--n-boot", "10", "--json", str(out)]
    )
    saved = json.loads(out.read_text())
    assert saved["tags"]["overall"]["actuation"]["route_changing_errors"] == 2
    assert set(saved["tags"]["by_split"]) == {"test", "ood"}
