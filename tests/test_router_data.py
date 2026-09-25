"""Router labels, class weights, rows loading and subword BIO alignment."""

import json

import pytest

from autogate_bench.intents import INTENTS as BENCH_INTENTS
from autogate_bench.schema import Route
from autogate_router.data import (
    NO_OFFSET,
    class_weights_from_cost_matrix,
    input_text,
    load_rows,
    spans_from_tags,
    subword_bio,
)
from autogate_router.labels import (
    BIO_TAGS,
    DEFAULT_LABELS,
    IGNORE_INDEX,
    INTENTS,
    ROUTES,
    Labels,
    load_labels,
    save_labels,
)

# --------------------------------------------------------------------------- #
# Labels (no torch needed)
# --------------------------------------------------------------------------- #


def test_vocabularies():
    assert ROUTES == tuple(str(r) for r in Route)
    assert INTENTS == tuple(i.name for i in BENCH_INTENTS)
    assert len(INTENTS) == 65
    assert len(BIO_TAGS) == 11 and BIO_TAGS[0] == "O"
    assert BIO_TAGS[1:3] == ("B-contact", "I-contact")


def test_labels_round_trip(tmp_path):
    save_labels(tmp_path)
    assert json.loads((tmp_path / "labels.json").read_text())["routes"] == list(ROUTES)
    assert load_labels(tmp_path) == DEFAULT_LABELS
    custom = Labels(routes=tuple(reversed(ROUTES)))
    save_labels(tmp_path, custom)
    loaded = load_labels(tmp_path / "labels.json")
    assert loaded.routes == tuple(reversed(ROUTES))
    assert loaded.route_index["REFUSE"] == 0


def test_class_weights():
    w = class_weights_from_cost_matrix()
    assert len(w) == 5
    assert all(x > 0 for x in w)
    assert sum(w) / len(w) == pytest.approx(1.0)
    idx = {r: i for i, r in enumerate(ROUTES)}
    assert w[idx["LOCAL"]] < w[idx["REFUSE"]]
    # the documented formula on configs/cost_matrix.json
    raw = [(8.25 + 2.25) / 2, 2.25, 4.0, 2.75, 7.5]
    mean = sum(raw) / 5
    assert w == pytest.approx([x / mean for x in raw])


def test_load_rows_filters_and_samples(pilot_jsonl, pilot_rows):
    assert len(load_rows(pilot_jsonl)) == len(pilot_rows)
    val = load_rows(pilot_jsonl, "val")
    assert val and all(r["split"] == "val" for r in val)
    both = load_rows(pilot_jsonl, "test,ood")
    assert {r["split"] for r in both} == {"test", "ood"}
    sample = load_rows(pilot_jsonl, "train", limit=40, seed=1)
    assert len(sample) == 40 and sample == load_rows(pilot_jsonl, "train", limit=40, seed=1)
    assert len({r["intent"] for r in sample}) > 10  # sampled, not the first 40 lines


def test_load_rows_parquet(tmp_path, pilot_rows):
    from autogate_bench.dataset import Row, write_parquet

    path = tmp_path / "rows.parquet"
    write_parquet([Row.from_dict(r) for r in pilot_rows[:30]], path)
    rows = load_rows(path)
    assert [r["id"] for r in rows] == [r["id"] for r in pilot_rows[:30]]


def test_input_text():
    row = {"context_prefix": "[speed=low]", "utterance": "call amma"}
    assert input_text(row) == ("[speed=low] call amma", 12)
    assert input_text(row, use_context=False) == ("call amma", 0)


def test_bio_from_offsets_without_tokenizer():
    utterance = "call priya natarajan ok"
    spans = [{"start": 5, "end": 20, "label": "contact", "text": "priya natarajan"}]
    offsets = [NO_OFFSET, (0, 4), (5, 8), (8, 10), (10, 20), (20, 23)]  # raw, untrimmed
    tags = subword_bio(offsets, spans)
    names = ["-" if t == IGNORE_INDEX else BIO_TAGS[t] for t in tags]
    assert names == ["-", "O", "B-contact", "I-contact", "I-contact", "O"]
    rebuilt = spans_from_tags(utterance, offsets, [n if n != "-" else "O" for n in names])
    assert rebuilt == spans


def test_spans_from_tags_lenient_i_and_trim():
    text = "text 2182 now"
    offsets = [(0, 4), (4, 9), (9, 13)]  # leading-space tokens, as byte-level BPE gives
    got = spans_from_tags(text, offsets, ["O", "I-otp", "O"])
    assert got == [{"start": 5, "end": 9, "label": "otp", "text": "2182"}]


# --------------------------------------------------------------------------- #
# With a real fast tokenizer (the tiny BPE one)
# --------------------------------------------------------------------------- #


def _sensitive(rows, n=50):
    out = [r for r in rows if r["spans"]][:n]
    assert len(out) == n
    return out


@pytest.mark.parametrize("use_context", [True, False])
def test_subword_bio_round_trips_char_spans(tiny_tokenizer, pilot_rows, use_context):
    from autogate_router.data import build_examples

    rows = _sensitive(pilot_rows)
    for row, ex in zip(rows, build_examples(rows, tiny_tokenizer, 96, use_context), strict=True):
        tags = ["O" if t == IGNORE_INDEX else BIO_TAGS[t] for t in ex["span_labels"]]
        assert spans_from_tags(row["utterance"], ex["offsets"], tags) == row["spans"], row["id"]


def test_prefix_tokens_are_ignored(tiny_tokenizer, pilot_rows):
    from autogate_router.data import build_examples

    rows = _sensitive(pilot_rows, 20)
    for row, ex in zip(rows, build_examples(rows, tiny_tokenizer, 96, True), strict=True):
        text, start = input_text(row)
        enc = tiny_tokenizer(text, return_offsets_mapping=True)
        n_prefix = 0
        for (_a, b), lab, um in zip(
            enc["offset_mapping"], ex["span_labels"], ex["utterance_mask"], strict=True
        ):
            if b <= start:  # entirely inside the prefix (or special)
                n_prefix += 1
                assert lab == IGNORE_INDEX and um == 0
            else:
                assert lab != IGNORE_INDEX and um == 1
        assert n_prefix > 0
        assert sum(ex["utterance_mask"]) < len(ex["input_ids"])


def test_no_context_has_no_prefix_tokens(tiny_tokenizer, pilot_rows):
    from autogate_router.data import build_examples

    rows = pilot_rows[:50]
    for ex in build_examples(rows, tiny_tokenizer, 96, False):
        assert "[speed" not in tiny_tokenizer.decode(ex["input_ids"])
        assert all(ex["utterance_mask"])
        assert IGNORE_INDEX not in ex["span_labels"]
        assert ex["offsets"][0][0] == 0


def test_continuation_subwords_get_i(tiny_tokenizer, pilot_rows):
    from autogate_router.data import build_examples

    rows = _sensitive(pilot_rows)
    seen_multi = False
    for row, ex in zip(rows, build_examples(rows, tiny_tokenizer, 96, True), strict=True):
        for s in row["spans"]:
            inside = [
                t
                for (a, b), t in zip(ex["offsets"], ex["span_labels"], strict=True)
                if (a, b) != NO_OFFSET and a < s["end"] and b > s["start"]
            ]
            assert BIO_TAGS[inside[0]] == f"B-{s['label']}"
            assert all(BIO_TAGS[t] == f"I-{s['label']}" for t in inside[1:])
            seen_multi |= len(inside) > 1
    assert seen_multi


def test_collate_pads(tiny_tokenizer, pilot_rows):
    torch = pytest.importorskip("torch")
    from autogate_router.data import RouterDataset, collate

    ds = RouterDataset(pilot_rows[:8], tiny_tokenizer)
    batch = collate([ds[i] for i in range(len(ds))], tiny_tokenizer.pad_token_id)
    n = max(len(ex["input_ids"]) for ex in ds.examples)
    assert batch["input_ids"].shape == (8, n)
    assert batch["route_labels"].dtype == torch.long and batch["route_labels"].shape == (8,)
    short = min(range(8), key=lambda i: len(ds[i]["input_ids"]))
    k = len(ds[short]["input_ids"])
    if k < n:
        assert batch["attention_mask"][short, k:].sum() == 0
        assert (batch["span_labels"][short, k:] == IGNORE_INDEX).all()
