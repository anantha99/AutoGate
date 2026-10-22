"""Router model, training loop, config parsing, save/load (tiny random backbone)."""

import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from autogate_router.data import (  # noqa: E402
    build_examples,
    class_weights_from_cost_matrix,
    collate,
)
from autogate_router.model import AutoGateRouter, RouterConfig, pool  # noqa: E402
from autogate_router.train import TrainConfig, parse_config, span_token_f1  # noqa: E402

SMALL_LORA = dict(lora_r=4, lora_alpha=8, lora_dropout=0.0, head_dropout=0.0)


def _model(backbone, **kw):
    kw.setdefault("tag_class_weights", {"actuation": [0.5, 2.0, 0.5, 1.0]})
    cfg = RouterConfig(
        backbone=str(backbone), route_class_weights=class_weights_from_cost_matrix(), **kw
    )
    torch.manual_seed(0)
    return AutoGateRouter(cfg)


def _batch(tokenizer, rows, use_context=True):
    return collate(build_examples(rows, tokenizer, 96, use_context), tokenizer.pad_token_id)


def test_pool_mean_and_last():
    h = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]])
    mean = pool(h, mask, "mean")
    assert torch.allclose(mean[0], (h[0, 1] + h[0, 2]) / 2)
    assert torch.all(mean[1] == 0)
    last = pool(h, mask, "last")
    assert torch.equal(last[0], h[0, 2])


@pytest.mark.parametrize(
    "kw", [dict(use_lora=True, pooling="mean"), dict(use_lora=False, pooling="last")]
)
def test_forward_shapes_and_loss(tiny_backbone, tiny_tokenizer, pilot_rows, kw):
    model = _model(tiny_backbone, **SMALL_LORA, **kw)
    batch = _batch(tiny_tokenizer, pilot_rows[:6])
    out = model(**batch)
    b, t = batch["input_ids"].shape
    assert out["route_logits"].shape == (b, 5)
    assert out["intent_logits"].shape == (b, 65)
    assert out["span_logits"].shape == (b, t, 11)
    for name, n in (("capability", 4), ("distraction", 3), ("actuation", 4), ("consequence", 3)):
        assert out[f"{name}_logits"].shape == (b, n)
        assert torch.isfinite(out[f"{name}_loss"]) and out[f"{name}_loss"] > 0
    assert torch.isfinite(out["loss"])
    tags = (
        1.0 * out["actuation_loss"]
        + 1.0 * out["distraction_loss"]
        + 0.5 * out["capability_loss"]
        + 0.3 * out["consequence_loss"]
    )
    assert torch.allclose(out["tags_loss"], tags)
    expected = out["route_loss"] + 0.5 * out["intent_loss"] + 1.0 * out["span_loss"] + tags
    assert torch.allclose(out["loss"], expected)
    logits_only = model.eval()(batch["input_ids"], batch["attention_mask"], batch["utterance_mask"])
    assert "loss" not in logits_only


def test_no_tag_heads_is_the_v0_model(tiny_backbone, tiny_tokenizer, pilot_rows):
    model = _model(tiny_backbone, tag_heads=False, **SMALL_LORA)
    assert not model.has_tag_heads and set(model.head_modules()) == {"route", "intent", "span"}
    out = model(**_batch(tiny_tokenizer, pilot_rows[:4]))  # tag labels in the batch are ignored
    assert not any(k.startswith(("capability", "actuation")) for k in out)
    expected = out["route_loss"] + 0.5 * out["intent_loss"] + out["span_loss"]
    assert torch.allclose(out["loss"], expected)


def test_label_smoothing_and_tag_weights(tiny_backbone, tiny_tokenizer, pilot_rows):
    from autogate_router.data import tag_class_weights

    batch = _batch(tiny_tokenizer, pilot_rows[:8])
    weights = tag_class_weights(pilot_rows)
    plain = _model(tiny_backbone, **SMALL_LORA).eval()
    smooth = _model(tiny_backbone, label_smoothing=0.1, tag_class_weights=weights, **SMALL_LORA)
    smooth.eval()
    assert torch.equal(smooth.actuation_weights, torch.tensor(weights["actuation"]))
    with torch.no_grad():
        a, b = plain(**batch), smooth(**batch)
    assert torch.equal(a["route_logits"], b["route_logits"])
    assert not torch.isclose(a["route_loss"], b["route_loss"])
    assert not torch.isclose(a["intent_loss"], b["intent_loss"])
    assert not torch.isclose(a["actuation_loss"], b["actuation_loss"])
    assert torch.equal(a["span_loss"], b["span_loss"])  # no smoothing on spans
    with pytest.raises(ValueError, match="lambda_tags"):
        RouterConfig(lambda_tags={"colour": 1.0})


def test_tag_class_weights_formula():
    from autogate_router.data import tag_class_weights

    rows = (
        [{"actuation": "none", "distraction": "low"}] * 6
        + [{"actuation": "comfort", "distraction": "medium"}] * 2
        + [{"actuation": "safety_critical", "distraction": "high"}]
        + [{"actuation": "restricted", "distraction": "high"}]
    )
    for r in rows:
        r.update(capability="local_ok", consequence="none")
    w = tag_class_weights(rows)
    raw = [1 / math.sqrt(0.6), 2.0 / math.sqrt(0.1), 1 / math.sqrt(0.2), 2.0 / math.sqrt(0.1)]
    assert w["actuation"] == pytest.approx([x / (sum(raw) / 4) for x in raw])
    raw = [1 / math.sqrt(0.6), 1 / math.sqrt(0.2), 1.5 / math.sqrt(0.2)]
    assert w["distraction"] == pytest.approx([x / (sum(raw) / 3) for x in raw])
    # absent classes are counted once; all weights finite and averaging 1
    for name in ("capability", "consequence"):
        assert sum(w[name]) / len(w[name]) == pytest.approx(1.0)
        assert all(math.isfinite(x) for x in w[name])


def test_lora_freezes_backbone(tiny_backbone):
    model = _model(tiny_backbone, **SMALL_LORA)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in trainable)
    assert all("lora_" in n or n.split(".")[0].endswith("_head") for n in trainable)


def test_training_steps_reduce_loss(tiny_backbone, tiny_tokenizer, pilot_rows):
    model = _model(tiny_backbone, **SMALL_LORA).train()
    batch = _batch(tiny_tokenizer, pilot_rows[::18][:64])
    assert batch["input_ids"].shape[0] == 64
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    assert model.has_tag_heads and "actuation_labels" in batch
    losses, tag_losses = [], []
    for _ in range(5):
        out = model(**batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
        tag_losses.append(out["tags_loss"].item())
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]
    assert tag_losses[-1] < tag_losses[0]


@pytest.mark.parametrize("use_lora", [True, False])
def test_save_load_round_trip(tmp_path, tiny_backbone, tiny_tokenizer, pilot_rows, use_lora):
    model = _model(tiny_backbone, use_lora=use_lora, **SMALL_LORA)
    # move the adapter off its zero init so the round trip means something
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.1)
    model.eval()
    batch = _batch(tiny_tokenizer, pilot_rows[:5])
    inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "utterance_mask")}
    with torch.no_grad():
        before = model(**inputs)
    model.save_pretrained(tmp_path / "ckpt", tiny_tokenizer)
    assert (tmp_path / "ckpt" / "labels.json").is_file()
    assert (tmp_path / "ckpt" / ("adapter" if use_lora else "backbone")).is_dir()
    saved = json.loads((tmp_path / "ckpt" / "router_config.json").read_text())
    assert saved["use_lora"] is use_lora and saved["max_length"] == 96
    loaded = AutoGateRouter.from_pretrained(tmp_path / "ckpt")
    with torch.no_grad():
        after = loaded(**inputs)
    assert loaded.has_tag_heads and loaded.config.tag_heads
    assert loaded.config.tag_class_weights == model.config.tag_class_weights
    assert torch.equal(loaded.actuation_weights, model.actuation_weights)
    for k in model.output_names:
        assert torch.equal(before[k], after[k]), k
    assert "actuation_logits" in model.output_names
    if use_lora:
        merged = AutoGateRouter.from_pretrained(tmp_path / "ckpt", merge_lora=True)
        with torch.no_grad():
            m = merged(**inputs)
        assert torch.allclose(m["route_logits"], before["route_logits"], atol=1e-5)


def test_span_token_f1():
    gold = [[-100, 0, 1, 2, 0]]
    assert span_token_f1([[5, 0, 1, 2, 0]], gold)["f1"] == 1.0
    r = span_token_f1([[0, 0, 1, 0, 3]], gold)
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_parse_config_yaml_and_flags(tmp_path):
    yml = tmp_path / "c.yaml"
    yml.write_text("epochs: 7\nlr: 1.0e-3\nlora_target_modules: [q_proj, v_proj]\nmax-steps: 9\n")
    cfg = parse_config(["--config", str(yml), "--lr", "5e-4", "--no-context", "--no-lora"])
    assert isinstance(cfg, TrainConfig)
    assert cfg.epochs == 7 and cfg.lr == 5e-4 and cfg.max_steps == 9
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]
    assert cfg.context is False and cfg.lora is False
    cfg = parse_config(["--lora-target-modules", "q_proj,o_proj", "--max-steps", "none"])
    assert cfg.lora_target_modules == ["q_proj", "o_proj"] and cfg.max_steps is None
    assert cfg.context and cfg.lora and cfg.pooling == "mean"
    yml.write_text("not_a_field: 1\n")
    with pytest.raises(ValueError, match="unknown config key"):
        parse_config(["--config", str(yml)])


def test_shipped_configs_parse():
    full = parse_config(["--config", "configs/train_qwen3_0.6b.yaml"])
    assert (full.epochs, full.batch_size, full.lr, full.lora_r, full.lora_alpha) == (
        3,
        32,
        2e-4,
        16,
        32,
    )
    assert full.backbone == "Qwen/Qwen3-0.6B" and full.lora and full.context
    assert not full.tag_heads  # v0 reproduces v0
    assert (full.context_dropout, full.word_dropout, full.label_smoothing) == (0.0, 0.0, 0.0)
    smoke = parse_config(["--config", "configs/train_smoke.yaml"])
    assert smoke.device == "cpu" and math.isfinite(smoke.lr)
    tags = parse_config(["--config", "configs/train_qwen3_0.6b_tags.yaml"])
    assert tags.tag_heads and tags.out_dir == "checkpoints/qwen3-0.6b-tags"
    assert (tags.epochs, tags.lr, tags.lora_r, tags.lora_alpha, tags.lora_dropout) == (
        2,
        1e-4,
        8,
        16,
        0.1,
    )
    assert tags.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert (tags.head_dropout, tags.weight_decay, tags.lambda_intent) == (0.3, 0.05, 0.2)
    assert (tags.label_smoothing, tags.context_dropout, tags.word_dropout) == (0.1, 0.2, 0.1)
    smoke_tags = parse_config(["--config", "configs/train_smoke_tags.yaml"])
    assert smoke_tags.tag_heads and smoke_tags.device == "cpu"
    assert (smoke_tags.label_smoothing, smoke_tags.context_dropout) == (0.1, 0.2)


def test_old_checkpoint_without_tag_heads_loads(
    tmp_path, tiny_backbone, tiny_tokenizer, pilot_rows
):
    """A checkpoint written before the tag heads: no tag keys in its config or labels."""
    model = _model(tiny_backbone, tag_heads=False, use_lora=False, tag_class_weights=None).eval()
    ckpt = model.save_pretrained(tmp_path / "old", tiny_tokenizer)
    cfg = json.loads((ckpt / "router_config.json").read_text())
    for key in ("tag_heads", "lambda_tags", "label_smoothing", "tag_class_weights"):
        cfg.pop(key)
    (ckpt / "router_config.json").write_text(json.dumps(cfg))
    labels = json.loads((ckpt / "labels.json").read_text())
    for key in ("capabilities", "distractions", "actuations", "consequences"):
        labels.pop(key)
    (ckpt / "labels.json").write_text(json.dumps(labels))

    loaded = AutoGateRouter.from_pretrained(ckpt)
    assert not loaded.has_tag_heads and not loaded.labels.has_tags
    assert loaded.output_names == ("route_logits", "intent_logits", "span_logits")
    batch = _batch(tiny_tokenizer, pilot_rows[:5])
    inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "utterance_mask")}
    with torch.no_grad():
        before, after = model(**inputs), loaded(**inputs)
    assert torch.equal(before["route_logits"], after["route_logits"])
    # training-style batches (with tag labels) still run through it
    assert torch.isfinite(loaded(**batch)["loss"])
