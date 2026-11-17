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


@pytest.fixture(scope="module")
def fake_token_cache(fake_cache):
    from autogate_decide.embed import build_token_cache

    cache = EmbeddingCache.load(fake_cache)
    build_token_cache(FakeEncoder(), cache.texts, cache.manifest).save(fake_cache / "tokens")
    return fake_cache


def test_token_cache_roundtrip_and_views(fake_token_cache):
    from autogate_decide.embed import TokenCache

    tc = TokenCache.load(fake_token_cache / "tokens")
    view = tc.lookup(tc.texts[:5])
    assert len(view) == 5
    sub = view[np.array([True, False, True, False, False])]
    x, mask = sub.padded([0, 1])
    assert x.shape[0] == 2 and mask.any(1).all()
    a, b = tc.offsets[2], tc.offsets[3]
    assert np.array_equal(x[1, : b - a], tc.states[a:b])


def test_attention_head_trains_and_predicts(fake_token_cache, pilot_rows):
    from autogate_decide.embed import TokenCache
    from autogate_decide.heads import HeadConfig, train_head

    utts = build_utterances(pilot_rows)
    x = TokenCache.load(fake_token_cache / "tokens").lookup(utts.texts)
    y = utts.answers["actuation"]
    head = train_head(x, y, 4, HeadConfig(epochs=3, weight_decay=0.01), seed=0)
    again = train_head(x, y, 4, HeadConfig(epochs=3, weight_decay=0.01), seed=0)
    p = head.probs(x[:20])
    assert p.shape == (20, 4) and np.allclose(p.sum(1), 1.0)
    assert np.allclose(head.logits(x[:20]), again.logits(x[:20]))  # deterministic per seed
    assert (head.probs(x).argmax(1) == y).mean() > 0.5


def test_tokens_run_with_fake_encoder(tmp_path, fake_token_cache, pilot_jsonl):
    from autogate_decide.run import RunConfig, run

    cfg = RunConfig(
        method="tokens",
        score=("val",),
        seeds=(0,),
        hidden_grid=(0,),
        weight_decay_grid=(0.1,),
        epochs=1,
        folds=2,
        lambda_grid=(0.5,),
        n_boot=10,
    )
    rep = run(fake_token_cache, pilot_jsonl, tmp_path / "tok", cfg)
    assert rep["seeds"][0]["splits"]["val"]["conservative"]["gate"]["n"] > 0


def test_safety_lane_is_built_from_train_seeds_and_never_fires_on_refuse(pilot_rows):
    from autogate_decide.registry import build_lane

    lane = build_lane(pilot_rows)
    assert lane.keywords
    for r in pilot_rows:
        if r["split"] in ("train", "val") and r["route"] == "REFUSE":
            assert not lane.fires([r["utterance"]])[0]
    sc_train = [
        r["utterance"]
        for r in pilot_rows
        if r["split"] == "train" and r["actuation"] == "safety_critical"
    ]
    assert lane.fires(sc_train).mean() > 0.5


def test_registry_replaces_tags_of_matched_utterances(fake_cache, pilot_rows):
    from autogate_decide.registry import apply_registry, build_registry, choose_tau

    utts = build_utterances(pilot_rows)
    x = EmbeddingCache.load(fake_cache).lookup(utts.texts)
    tau, scan = choose_tau(pilot_rows, utts, x, folds=2)
    assert tau > 1.0 or tau in [s["tau"] for s in scan]  # all-seed pilot: no queries
    reg = build_registry(pilot_rows, utts, x, 0.999, "fair")
    probs = {q: np.full((len(utts), len(v)), 1.0 / len(v)) for q, v in QUESTIONS.items()}
    out, hit = apply_registry(probs, reg, x)
    seeds = [utts.row_utt[i] for i, r in enumerate(pilot_rows) if r["split"] == "train"]
    assert hit[seeds].all()  # every training seed matches itself
    i = seeds[0]
    assert QUESTIONS["actuation"][out["actuation"][i].argmax()] == pilot_rows[0]["actuation"]
    assert np.allclose(out["sensitive"], probs["sensitive"])  # heads keep sensitivity
    reg_ood = build_registry(pilot_rows, utts, x, 0.999, "registered")
    assert len(reg_ood.intents) > len(reg.intents)


def test_registered_run_with_lane(tmp_path, fake_cache, pilot_rows):
    from autogate_decide.run import RunConfig, run

    # half the seeds stand in for paraphrases, so registered mode has OOD rows left to score
    rows = tmp_path / "rows.jsonl"
    with rows.open("w", encoding="utf-8") as f:
        for r in pilot_rows:
            odd = int(r["seed_id"].rsplit("/", 1)[1]) % 2
            f.write(json.dumps({**r, "source": "paraphrase" if odd else "seed"}) + "\n")

    cfg = RunConfig(
        score=("ood",),
        seeds=(0,),
        hidden_grid=(0,),
        weight_decay_grid=(0.1,),
        steps=20,
        folds=2,
        lambda_grid=(0.5,),
        n_boot=10,
        registry="registered",
        lane=True,
    )
    rep = run(fake_cache, rows, tmp_path / "reg", cfg)
    assert rep["phase3"]["registry_hit_rate"]["ood"] > 0
    assert rep["phase3"]["lane"]["val"]["fired_on_refuse"] == 0
    n_ood = sum(r["split"] == "ood" for r in pilot_rows)
    n_scored = rep["seeds"][0]["splits"]["ood"]["conservative"]["gate"]["n"]
    assert 0 < n_scored < n_ood  # the registered OOD seeds are not scored


def test_seed_folds_keep_intents_and_hold_out_whole_seeds(pilot_rows):
    from autogate_decide.data import seed_folds

    utts = build_utterances(pilot_rows)
    train = utts.mask("train")
    folds = seed_folds(utts, train, k=3)
    assert np.array_equal(np.sum(folds, axis=0)[train], np.ones(train.sum()))
    for held in folds:
        assert not set(utts.seed_id[held]) & set(utts.seed_id[train & ~held])


def test_layer_mix_head_and_blend_selection(tmp_path, fake_cache, pilot_jsonl):
    from autogate_decide.embed import TokenCache, build_token_cache
    from autogate_decide.heads import HeadConfig, train_head
    from autogate_decide.run import RunConfig, run

    cache = EmbeddingCache.load(fake_cache)
    tc = build_token_cache(FakeEncoder(), cache.texts, cache.manifest, layers=[1, 2])
    d = tmp_path / "c"
    cache.save(d)
    tc.save(d / "tokens")
    x = TokenCache.load(d / "tokens").lookup(cache.texts[:40])
    assert x.n_layers == 2 and x.dim == 2 * ENCODERS["fake"].dim
    y = np.arange(40) % 3
    head = train_head(x, y, 3, HeadConfig(epochs=2), seed=0)
    assert head.params["mix"].shape == (2,)
    assert np.allclose(head.probs(x).sum(1), 1.0)

    cfg = RunConfig(
        method="tokens",
        score=("val",),
        seeds=(0,),
        hidden_grid=(0,),
        weight_decay_grid=(0.01, 1.0),
        epochs_grid=(1, 2),
        selection="blend",
        folds=2,
        seed_folds=2,
        lambda_grid=(0.5,),
        n_boot=10,
    )
    rep = run(d, pilot_jsonl, tmp_path / "blend", cfg)
    sel = rep["config"]["selected"]["actuation"]
    trials = rep["config"]["selection_trials"]["actuation"]
    assert len(trials) == 4 and all("seed_nll" in t for t in trials)
    best = min(t["score"] for t in trials)
    chosen = next(
        t
        for t in trials
        if (t["weight_decay"], t["epochs"]) == (sel["weight_decay"], sel["epochs"])
    )
    assert chosen["score"] <= best + max(t["score_se"] for t in trials)
