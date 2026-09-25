"""Dataset rows: labels, ids, splits, determinism, and the writers."""

import json
import random
from collections import defaultdict
from pathlib import Path

import pytest

from autogate_bench import IntentGroup, route
from autogate_bench.dataset import (
    Row,
    asr_texts,
    build_rows,
    read_jsonl,
    read_parquet,
    summarize,
    write_jsonl,
    write_parquet,
)
from autogate_bench.generate import main as generate
from autogate_bench.intents import INTENTS_BY_NAME
from autogate_bench.paraphrases import Paraphrase, load_paraphrases
from autogate_bench.spans import spans_from_bio
from autogate_bench.splits import OOD_INTENTS, assign_splits, seed_id

ROWS = build_rows(rng_seed=0)


def test_rows_exist():
    assert len(ROWS) > 1000


def test_route_matches_rulebook_recomputed_from_row_fields():
    for r in ROWS:
        d = route(INTENTS_BY_NAME[r.intent], r.context(), r.has_sensitive_spans)
        assert (r.route, r.reason, r.demand) == (str(d.route), str(d.reason), str(d.demand))
        assert r.context_prefix == r.context().to_prefix()


def test_ids_unique():
    assert len({r.id for r in ROWS}) == len(ROWS)


def test_every_seed_has_one_split():
    by_seed = defaultdict(set)
    for r in ROWS:
        by_seed[r.seed_id].add(r.split)
    assert all(len(s) == 1 for s in by_seed.values())


def test_ood_intents_only_in_ood():
    for r in ROWS:
        assert (r.intent in OOD_INTENTS) == (r.split == "ood")
    assert {r.intent for r in ROWS if r.split == "ood"} == set(OOD_INTENTS)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_group_in_every_split(split):
    assert {r.group for r in ROWS if r.split == split} == {str(g) for g in IntentGroup}


def test_every_in_distribution_intent_is_in_train():
    train = {r.intent for r in ROWS if r.split == "train"}
    assert train == set(INTENTS_BY_NAME) - set(OOD_INTENTS)


def test_split_proportions():
    seeds = {r.seed_id: r.split for r in ROWS}
    n_id = sum(s != "ood" for s in seeds.values())
    for split, lo, hi in (("train", 0.75, 0.85), ("val", 0.07, 0.13), ("test", 0.07, 0.13)):
        assert lo <= sum(s == split for s in seeds.values()) / n_id <= hi


def test_assign_splits_is_order_independent_and_deterministic():
    ids = [seed_id(n, i) for n in INTENTS_BY_NAME for i in range(4)]
    a = assign_splits(ids, random.Random(1))
    b = assign_splits(list(reversed(ids)), random.Random(1))
    assert a == b
    assert a != assign_splits(ids, random.Random(2))


def test_assign_splits_rejects_unknown_ood_intent():
    with pytest.raises(ValueError, match="unknown OOD"):
        assign_splits(["call_contact/0"], random.Random(0), ood_intents=["nope"])


def test_slices_are_consistent():
    for r in ROWS:
        assert r.context_invariant != r.context_flipped
        assert r.sensitive == r.has_sensitive_spans == bool(r.spans)
        if r.style == "question" and r.actuation == "restricted":
            assert r.adversarial
        if r.adversarial and r.actuation != "restricted":
            assert r.route == "REFUSE"
    # every utterance of a flipped intent appears in >= 2 routes
    by_utt = defaultdict(set)
    for r in ROWS:
        by_utt[r.id.rsplit("-", 1)[0]].add((r.route, r.context_flipped))
    for routes in by_utt.values():
        flipped = next(iter(routes))[1]
        assert (len(routes) >= 2) == flipped


def test_spans_and_bio_agree():
    for r in ROWS:
        for s in r.spans:
            assert r.utterance[s.start : s.end] == s.text
        assert len(r.bio) == len(r.tokens)
        assert spans_from_bio(r.utterance, r.bio) == list(r.spans)


def test_generation_is_byte_identical(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(build_rows(rng_seed=3), a)
    write_jsonl(build_rows(rng_seed=3), b)
    assert a.read_bytes() == b.read_bytes()
    c = tmp_path / "c.jsonl"
    write_jsonl(build_rows(rng_seed=4), c)
    assert a.read_bytes() != c.read_bytes()


def test_jsonl_round_trip(tmp_path):
    p = tmp_path / "rows.jsonl"
    write_jsonl(ROWS, p)
    assert read_jsonl(p) == ROWS
    first = json.loads(p.read_text().splitlines()[0])
    assert all(not isinstance(v, dict) for v in first.values())  # flat record


def test_parquet_round_trip(tmp_path):
    p = tmp_path / "rows.parquet"
    write_parquet(ROWS, p)
    assert read_parquet(p) == ROWS


def test_row_dict_round_trip():
    r = next(r for r in ROWS if r.spans)
    assert Row.from_dict(r.to_dict()) == r


def test_cli_writes_all_three_files(tmp_path, capsys):
    out = tmp_path / "pilot"
    manifest = generate(["--out", str(out), "--rng-seed", "0"])
    assert {p.name for p in out.iterdir()} == {"rows.jsonl", "rows.parquet", "manifest.json"}
    assert read_jsonl(out / "rows.jsonl") == ROWS
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["n_rows"] == len(ROWS)
    assert sum(manifest["split"].values()) == len(ROWS)
    assert sum(manifest["route"].values()) == len(ROWS)
    assert manifest == {**manifest, **summarize(ROWS)}
    assert manifest["args"]["rng_seed"] == 0
    assert "commit" in manifest["git"]


# ---- paraphrases and ASR copies ------------------------------------------- #

FIXTURE = Path(__file__).parent / "fixtures" / "paraphrases"
PARA = load_paraphrases(FIXTURE)
PROWS = build_rows(rng_seed=0, paraphrases=PARA, asr_per_seed=4)


def test_seed_only_rows_unchanged_by_the_paraphrase_code(tmp_path):
    assert [r for r in PROWS if r.source == "seed"] == ROWS
    assert build_rows(rng_seed=0, paraphrases={}, asr_per_seed=4) == ROWS
    p = tmp_path / "rows.jsonl"
    write_jsonl(ROWS, p)
    first = json.loads(p.read_text().splitlines()[0])
    assert "source" not in first and "variant" not in first  # seed-only bytes unchanged
    assert all(r.source == "seed" and r.variant == "plain" for r in read_jsonl(p))


def test_paraphrase_rows_source_variant_and_ids():
    para = [r for r in PROWS if r.source != "seed"]
    assert {r.intent for r in para} == set(PARA)
    for r in para:
        uid = r.id.rsplit("-", 2)[0]
        seed_id_, rest = uid.split("/p", 1)
        assert seed_id_ == r.seed_id
        k = int(rest.split("/")[0])
        index = int(r.seed_id.rsplit("/", 1)[1])
        source_p = PARA[r.intent][index][k]
        if r.source == "paraphrase":
            assert "/a" not in rest and r.variant == source_p.variant
        else:
            assert r.source == "asr" and r.variant == "asr"
            assert rest.split("/")[1].startswith("a") and source_p.variant == "plain"


def test_paraphrase_rows_counts():
    utts = {r.id.rsplit("-", 1)[0]: r for r in PROWS}
    per_seed = defaultdict(lambda: defaultdict(set))
    for key, r in utts.items():
        per_seed[r.seed_id][r.source].add(key.rsplit("-", 1)[0])
    for intent, by_seed in PARA.items():
        for index in by_seed:
            s = per_seed[seed_id(intent, index)]
            assert len(s["paraphrase"]) == 40
            assert len(s["asr"]) == 4


def test_paraphrases_inherit_split_style_and_labels():
    seed_split = {r.seed_id: (r.split, r.style) for r in ROWS}
    for r in PROWS:
        assert (r.split, r.style) == seed_split[r.seed_id]
        d = route(INTENTS_BY_NAME[r.intent], r.context(), r.has_sensitive_spans)
        assert (r.route, r.reason, r.demand) == (str(d.route), str(d.reason), str(d.demand))
        assert r.sensitive == r.has_sensitive_spans == bool(r.spans)
        for s in r.spans:
            assert r.utterance[s.start : s.end] == s.text
        assert spans_from_bio(r.utterance, r.bio) == list(r.spans)
        assert "{" not in r.utterance
    splits = {r.split for r in PROWS if r.intent == "plan_day_itinerary" and r.source != "seed"}
    assert splits == {"train", "test"}
    assert {r.split for r in PROWS if r.intent == "start_video_call"} == {"train"}


def test_paraphrase_rows_keep_every_slice_invariant():
    by_utt = defaultdict(set)
    for r in PROWS:
        if r.style == "question" and r.actuation == "restricted":
            assert r.adversarial
        if r.adversarial and r.actuation != "restricted":
            assert r.route == "REFUSE"
        by_utt[r.id.rsplit("-", 1)[0]].add((r.route, r.context_flipped))
    for routes in by_utt.values():
        assert (len(routes) >= 2) == next(iter(routes))[1]
    assert len({r.id for r in PROWS}) == len(PROWS)


def test_paraphrase_generation_is_deterministic_and_round_trips(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(PROWS, a)
    write_jsonl(build_rows(rng_seed=0, paraphrases=load_paraphrases(FIXTURE)), b)
    assert a.read_bytes() == b.read_bytes()
    assert read_jsonl(a) == PROWS
    pq = tmp_path / "rows.parquet"
    write_parquet(PROWS, pq)
    assert read_parquet(pq) == PROWS


def test_asr_per_seed_zero_and_invalid_paraphrases():
    rows = build_rows(rng_seed=0, paraphrases=PARA, asr_per_seed=0)
    assert not any(r.source == "asr" for r in rows)
    bad = {"start_video_call": {0: (Paraphrase("start_video_call", 0, "call mom", "plain"),)}}
    with pytest.raises(ValueError, match="paraphrase errors"):
        build_rows(rng_seed=0, paraphrases=bad)


def test_asr_texts_spread_across_paraphrases():
    plain = [(k, p) for k, p in enumerate(PARA["start_video_call"][0]) if p.variant == "plain"]
    out = asr_texts(plain, random.Random(0), 4)
    assert len(out) == 4 and len({k for k, _, _ in out}) == 4
    assert all(a == 0 for _, a, _ in out)
    many = asr_texts(plain[:2], random.Random(0), 5)
    assert {k for k, _, _ in many} <= {plain[0][0], plain[1][0]} and len(many) == 5
    assert len({t for _, _, t in many}) == 5


def test_manifest_counts_sources_and_variants(tmp_path, capsys):
    manifest = generate(["--out", str(tmp_path), "--paraphrases", str(FIXTURE)])
    assert manifest["source"]["seed"] == len(ROWS)
    assert sum(manifest["source"].values()) == manifest["n_rows"] == len(PROWS)
    assert sum(manifest["variant"].values()) == len(PROWS)
    filled_asr = {r.id.rsplit("-", 1)[0] for r in PROWS if r.source == "asr"}
    assert manifest["utterances_by_source"]["asr"] == len(filled_asr)
    assert manifest["args"]["paraphrases"].endswith("fixtures/paraphrases")
    assert manifest["args"]["asr_per_seed"] == 4
    seed_only = summarize(ROWS)
    assert seed_only["source"] == {"seed": len(ROWS), "paraphrase": 0, "asr": 0}
