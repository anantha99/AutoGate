"""Dataset rows: labels, ids, splits, determinism, and the writers."""

import json
import random
from collections import defaultdict

import pytest

from autogate_bench import IntentGroup, route
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
        by_utt[(r.seed_id, r.id.rsplit("-", 2)[1])].add((r.route, r.context_flipped))
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
