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
    ACTUATIONS,
    BIO_TAGS,
    CAPABILITIES,
    CONSEQUENCES,
    DEFAULT_LABELS,
    DISTRACTIONS,
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


def test_tag_vocabularies_follow_enum_order():
    from autogate_bench.schema import ActuationClass, Capability, Consequence

    assert CAPABILITIES == tuple(str(c) for c in Capability)
    assert DISTRACTIONS == ("low", "medium", "high")
    assert ACTUATIONS == ("none", "safety_critical", "comfort", "restricted")
    assert ACTUATIONS == tuple(str(a) for a in ActuationClass)
    assert CONSEQUENCES == tuple(str(c) for c in Consequence)
    assert list(DEFAULT_LABELS.tags) == ["capability", "distraction", "actuation", "consequence"]
    assert DEFAULT_LABELS.tag_index["actuation"]["restricted"] == 3


def test_labels_json_has_tags_and_round_trips(tmp_path):
    save_labels(tmp_path)
    d = json.loads((tmp_path / "labels.json").read_text())
    assert d["capabilities"] == list(CAPABILITIES)
    assert d["distractions"] == list(DISTRACTIONS)
    assert d["actuations"] == list(ACTUATIONS)
    assert d["consequences"] == list(CONSEQUENCES)
    loaded = load_labels(tmp_path)
    assert loaded == DEFAULT_LABELS and loaded.has_tags


def test_old_labels_json_without_tags_loads(tmp_path):
    old = {"routes": list(ROUTES), "intents": list(INTENTS), "bio_tags": list(BIO_TAGS)}
    (tmp_path / "labels.json").write_text(json.dumps(old))
    loaded = load_labels(tmp_path)
    assert not loaded.has_tags and loaded.tags == {} and loaded.tag_index == {}
    assert loaded.routes == ROUTES and loaded.to_dict() == old
    assert loaded == DEFAULT_LABELS.without_tags()
    with pytest.raises(ValueError, match="all of"):
        Labels(tags={"actuation": ACTUATIONS})


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


# --------------------------------------------------------------------------- #
# Tag labels and training-time augmentation
# --------------------------------------------------------------------------- #


def test_tag_labels_encoded_and_collated(tiny_tokenizer, pilot_rows):
    pytest.importorskip("torch")
    from autogate_router.data import RouterDataset, build_examples, collate

    rows = pilot_rows[:8]
    exs = build_examples(rows, tiny_tokenizer)
    for row, ex in zip(rows, exs, strict=True):
        for name in ("capability", "distraction", "actuation", "consequence"):
            assert DEFAULT_LABELS.tags[name][ex["tag_labels"][name]] == row[name]
    batch = collate(RouterDataset(rows, tiny_tokenizer).examples, tiny_tokenizer.pad_token_id)
    for name in ("capability", "distraction", "actuation", "consequence"):
        assert batch[f"{name}_labels"].shape == (8,)
    # labels without tags (an old checkpoint) encode no tag labels
    old = build_examples(rows[:2], tiny_tokenizer, labels=DEFAULT_LABELS.without_tags())
    assert all("tag_labels" not in ex for ex in old)


def test_word_dropout_keeps_span_offsets_exact(pilot_rows):
    import random

    from autogate_router.data import drop_words

    changed = 0
    for i, row in enumerate(r for r in pilot_rows if r["spans"]):
        out = drop_words(row, 0.5, random.Random(i))
        assert len(out["spans"]) == len(row["spans"])
        for old, new in zip(row["spans"], out["spans"], strict=True):
            assert out["utterance"][new["start"] : new["end"]] == old["text"] == new["text"]
            assert new["label"] == old["label"]
        assert (
            out["utterance"] == out["utterance"].strip()
            or row["utterance"] != row["utterance"].strip()
        )
        assert "  " not in out["utterance"] or "  " in row["utterance"]
        changed += out["utterance"] != row["utterance"]
    assert changed > 50
    every = drop_words({"utterance": "a b c", "spans": []}, 1.0, random.Random(0))
    assert every["utterance"] == "a b c"  # never drops every word


@pytest.mark.parametrize("use_context", [True, False])
def test_word_dropout_subword_spans_round_trip(tiny_tokenizer, pilot_rows, use_context):
    from autogate_router.data import Augment, RouterDataset

    rows = _sensitive(pilot_rows)
    ds = RouterDataset(
        rows, tiny_tokenizer, use_context=use_context, augment=Augment(word_dropout=0.4, seed=3)
    )
    for epoch in (1, 2):
        ds.set_epoch(epoch)
        for i, row in enumerate(rows):
            aug_row, _ = ds.augment.apply(row, epoch, i)
            ex = ds[i]
            tags = ["O" if t == IGNORE_INDEX else BIO_TAGS[t] for t in ex["span_labels"]]
            got = spans_from_tags(aug_row["utterance"], ex["offsets"], tags)
            assert got == aug_row["spans"], row["id"]
            assert [s["text"] for s in got] == [s["text"] for s in row["spans"]]


def test_context_dropout_removes_prefix_tokens(tiny_tokenizer, pilot_rows):
    from autogate_router.data import Augment, RouterDataset

    rows = pilot_rows[:20]
    ds = RouterDataset(rows, tiny_tokenizer, augment=Augment(context_dropout=1.0))
    ds.set_epoch(1)
    for i in range(len(rows)):
        ex = ds[i]
        assert "[speed" not in tiny_tokenizer.decode(ex["input_ids"])
        assert all(ex["utterance_mask"])
        assert len(ex["input_ids"]) < len(ds.examples[i]["input_ids"])
    half = RouterDataset(rows * 5, tiny_tokenizer, augment=Augment(context_dropout=0.5))
    half.set_epoch(1)
    dropped = sum(all(half[i]["utterance_mask"]) for i in range(len(half)))
    assert 20 < dropped < 80


def test_augmentation_is_seeded_per_epoch(tiny_tokenizer, pilot_rows):
    from autogate_router.data import Augment, RouterDataset

    rows = pilot_rows[:40]
    aug = Augment(context_dropout=0.3, word_dropout=0.3, seed=7)

    def draw(epoch):
        ds = RouterDataset(rows, tiny_tokenizer, augment=aug)
        ds.set_epoch(epoch)
        return [ds[i]["input_ids"] for i in range(len(ds))]

    assert draw(1) == draw(1)
    assert draw(1) != draw(2)
    # off by default: the dataset serves the plain encodings
    plain = RouterDataset(rows, tiny_tokenizer, augment=Augment())
    assert plain.augment is None and plain[0] is plain.examples[0]
    with pytest.raises(ValueError):
        Augment(word_dropout=1.5)


def test_val_is_never_augmented(tiny_tokenizer, pilot_rows):
    from autogate_router.data import build_examples
    from autogate_router.train import TrainConfig, build_datasets

    cfg = TrainConfig(context_dropout=1.0, word_dropout=0.9)
    train_rows = [r for r in pilot_rows if r["split"] == "train"][:30]
    val_rows = [r for r in pilot_rows if r["split"] == "val"][:30]
    train_ds, val_ds = build_datasets(cfg, tiny_tokenizer, train_rows, val_rows)
    assert train_ds.augment is not None and val_ds.augment is None
    plain = build_examples(val_rows, tiny_tokenizer)
    for epoch in (1, 2, 3):
        val_ds.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        assert [val_ds[i] for i in range(len(val_ds))] == plain
    assert train_ds[0]["input_ids"] != train_ds.examples[0]["input_ids"]
