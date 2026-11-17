"""autogate_decide: cache, examples, routing tables, decoders, abstention, a full fake run."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from autogate_bench.rulebook import route_tags
from autogate_decide import route as rt
from autogate_decide.data import QUESTIONS, build_utterances, intent_folds
from autogate_decide.embed import EmbeddingCache, build_cache
from autogate_decide.encoders import ENCODERS, FakeEncoder

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def fake_cache(tmp_path_factory, pilot_jsonl):
    cache = build_cache(FakeEncoder(), [pilot_jsonl])
    out = tmp_path_factory.mktemp("cache")
    cache.save(out)
    return out


def test_fake_encoder_is_deterministic_and_normalised():
    enc = FakeEncoder()
    a = enc.encode(["turn on the wipers", "call mom"])
    b = enc.encode(["turn on the wipers", "call mom"])
    assert a.shape == (2, ENCODERS["fake"].dim)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_cache_roundtrip(fake_cache, pilot_rows):
    cache = EmbeddingCache.load(fake_cache)
    texts = list(dict.fromkeys(r["utterance"] for r in pilot_rows))
    assert cache.texts == texts
    assert cache.manifest["determinism_max_abs_diff"] == 0.0
    assert cache.manifest["all_finite"]
    assert cache.lookup(texts[:3]).shape == (3, cache.dim)
    with pytest.raises(KeyError):
        cache.lookup(["not an utterance in the pilot"])


def test_utterances_dedupe_contexts(pilot_rows):
    utts = build_utterances(pilot_rows)
    assert len(utts) * 3 == len(pilot_rows)  # three contexts per utterance
    for i, r in enumerate(pilot_rows[:50]):
        u = utts.row_utt[i]
        assert utts.texts[u] == r["utterance"]
        assert QUESTIONS["actuation"][utts.answers["actuation"][u]] == r["actuation"]
        assert utts.answers["sensitive"][u] == int(r["has_sensitive_spans"])


def test_intent_folds_hold_out_whole_intents(pilot_rows):
    utts = build_utterances(pilot_rows)
    train = utts.mask("train")
    folds = intent_folds(utts, train, k=3)
    assert np.array_equal(np.sum(folds, axis=0)[train], np.ones(train.sum()))
    for held in folds:
        assert not set(utts.intent[held]) & set(utts.intent[train & ~held])


def test_route_tables_match_rulebook():
    t = rt.route_tables()
    rng = np.random.default_rng(0)
    for _ in range(300):
        i, j = rng.integers(len(rt.CONTEXTS)), rng.integers(len(rt.COMBOS))
        cap, dis, act, sens = rt.COMBOS[j]
        want = route_tags(cap, dis, act, rt.CONTEXTS[i], sens == "yes").route
        assert rt.ROUTES[t.route[i, j]] == str(want)
        assert t.feasible[i, t.route[i, j]]
        assert t.cost[i, j, t.route[i, j]] == 0


def test_expected_cost_with_certain_answers_is_the_rulebook_route():
    t = rt.route_tables()
    n = len(rt.COMBOS)
    for i in range(0, len(rt.CONTEXTS), 7):
        pc = np.eye(n)
        ctx = np.full(n, i)
        assert np.array_equal(rt.decode_expected_cost(pc, ctx, t), t.route[i])
        assert np.allclose(rt.p_critical(pc, ctx, t.route[i], t), 0.0)


def test_expected_cost_never_picks_an_infeasible_route():
    t = rt.route_tables()
    rng = np.random.default_rng(1)
    pc = rng.dirichlet(np.ones(len(rt.COMBOS)), size=500)
    ctx = rng.integers(len(rt.CONTEXTS), size=500)
    route = rt.decode_expected_cost(pc, ctx, t)
    assert t.feasible[ctx, route].all()


def test_combo_probs_sum_to_one_and_joint_mix():
    rng = np.random.default_rng(2)
    probs = {q: rng.dirichlet(np.ones(len(QUESTIONS[q])), size=4) for q in rt.COMBO_QUESTIONS}
    vocab = [("local_ok", "low", "comfort"), ("needs_cloud", "high", "none")]
    joint = rng.dirichlet(np.ones(2), size=4)
    for mode, lam in (("product", 0.0), ("joint", 0.0), ("mix", 0.3)):
        pc = rt.combo_probs(probs, mode, joint, vocab, lam)
        assert pc.shape == (4, 96)
        assert np.allclose(pc.sum(1), 1.0)
    pj = rt.combo_probs(probs, "joint", joint, vocab)
    other = [k for k, c in enumerate(rt.COMBOS) if c[:3] not in vocab]
    assert np.allclose(pj[:, other], 0.0)


def test_conservative_decoder_matches_decode_tags():
    from autogate_router.tag_routing import decode_tags

    probs = {
        "capability": np.array([[0.7, 0.1, 0.1, 0.1]]),
        "distraction": np.array([[0.6, 0.1, 0.3]]),
        "actuation": np.array([[0.5, 0.1, 0.1, 0.3]]),
        "consequence": np.array([[0.4, 0.3, 0.3]]),
        "sensitive": np.array([[0.4, 0.6]]),
    }
    combo, tags = rt.conservative_combo(probs)
    per = {q: dict(zip(QUESTIONS[q], probs[q][0], strict=True)) for q in QUESTIONS}
    want = decode_tags({k: v for k, v in per.items() if k != "sensitive"})
    assert tags[0] == {**want, "sensitive": "yes"}
    assert rt.COMBOS[combo[0]] == (
        want["capability"],
        want["distraction"],
        want["actuation"],
        "yes",
    )


def test_pick_threshold_meets_target(pilot_rows):
    rng = np.random.default_rng(3)
    routes = [r["route"] if rng.random() < 0.8 else "CLOUD" for r in pilot_rows]
    risk = rng.random(len(pilot_rows))
    thr = rt.pick_threshold(pilot_rows, routes, risk, 0.05)
    res = rt.apply_threshold(pilot_rows, routes, risk, thr)
    assert res["critical_miss_rate"] is None or res["critical_miss_rate"] <= 0.05


def test_scorer_uses_sensitive_field(pilot_rows):
    from autogate_eval.score import predicted_sensitive, tag_scores

    assert predicted_sensitive({"sensitive": True}) is True
    assert predicted_sensitive({"sensitive": False, "spans": [{"start": 0}]}) is False
    assert predicted_sensitive({"spans": [{"start": 0}]}) is True
    rows = [r for r in pilot_rows if r["has_sensitive_spans"]][:20]
    preds = [
        {
            "id": r["id"],
            "route": r["route"],
            "tags": {t: r[t] for t in ("capability", "distraction", "actuation", "consequence")},
            "sensitive": True,
        }
        for r in rows
    ]
    out = tag_scores(rows, preds)
    assert all(m["route_changing_errors"] == 0 for m in out["overall"].values())


def test_questions_yaml_has_every_option_and_no_ood_key_words():
    from autogate_bench.augment import OOD_KEYWORDS
    from autogate_decide.run import load_questions

    questions = load_questions()
    for q, vocab in QUESTIONS.items():
        assert tuple(questions[q]) == vocab
    for q, opts in questions.items():
        for text in opts.values():
            words = set(re.findall(r"[a-z]+", text.lower()))
            for intent, entries in OOD_KEYWORDS.items():
                assert intent not in text
                for entry in entries:
                    assert not all(any(w.startswith(k) for w in words) for k in entry), (
                        q,
                        text,
                        entry,
                    )


def test_full_run_with_fake_encoder(tmp_path, fake_cache, pilot_jsonl):
    from autogate_decide.run import RunConfig, run
    from autogate_eval.score import _read_jsonl, load_rows, score

    cfg = RunConfig(
        score=("val", "test"),
        seeds=(0, 1),
        hidden_grid=(0,),
        weight_decay_grid=(0.1,),
        steps=30,
        folds=2,
        lambda_grid=(0.0, 1.0),
        n_boot=20,
    )
    out = tmp_path / "run"
    rep = run(fake_cache, pilot_jsonl, out, cfg)
    assert set(rep["seeds"]) == {0, 1}
    for dec in ("conservative", "expected_cost"):
        preds = _read_jsonl(out / f"pred_{dec}.jsonl")
        assert {p["decoder"] for p in preds} == {dec}
        rows = [r for r in load_rows(pilot_jsonl) if r["split"] in ("val", "test")]
        assert len(preds) == len(rows)
        assert score(rows, preds, n_boot=10)["route"]["overall"]["n"] == len(rows)
    config = json.loads((out / "config.json").read_text())
    assert config["lam"] in (0.0, 1.0)
    assert "report.md" in {p.name for p in out.iterdir()}

    again = RunConfig(**{**cfg.__dict__, "selected": config["selected"], "lam": config["lam"]})
    rep2 = run(fake_cache, pilot_jsonl, tmp_path / "again", again)
    a = rep["seeds"][0]["splits"]["test"]["conservative"]["gate"]
    b = rep2["seeds"][0]["splits"]["test"]["conservative"]["gate"]
    assert a["swe"] == b["swe"]  # reusing a committed selection reproduces the run


def test_zeroshot_run_with_fake_encoder(tmp_path, fake_cache, pilot_jsonl):
    from autogate_decide.run import RunConfig, run

    cfg = RunConfig(method="zeroshot", score=("val",), n_boot=10)
    rep = run(fake_cache, pilot_jsonl, tmp_path / "zs", cfg)
    assert set(rep["seeds"][0]["temperatures"]["zeroshot"]) == set(QUESTIONS)
