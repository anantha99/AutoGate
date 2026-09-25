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
    assert torch.isfinite(out["loss"])
    expected = out["route_loss"] + 0.5 * out["intent_loss"] + 1.0 * out["span_loss"]
    assert torch.allclose(out["loss"], expected)
    logits_only = model.eval()(batch["input_ids"], batch["attention_mask"], batch["utterance_mask"])
    assert "loss" not in logits_only


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
    losses = []
    for _ in range(5):
        out = model(**batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert losses[-1] < losses[0]


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
    for k in ("route_logits", "intent_logits", "span_logits"):
        assert torch.equal(before[k], after[k]), k
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
    smoke = parse_config(["--config", "configs/train_smoke.yaml"])
    assert smoke.device == "cpu" and math.isfinite(smoke.lr)
