"""End to end on CPU: train -> predict -> score -> calibrate -> export (tiny backbone)."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from autogate_eval.score import main as score_main  # noqa: E402
from autogate_router.calibrate import THRESHOLDS  # noqa: E402
from autogate_router.calibrate import main as calibrate_main  # noqa: E402
from autogate_router.predict import PREDICTION_FIELDS, TAG_PREDICTION_FIELDS  # noqa: E402
from autogate_router.predict import main as predict_main  # noqa: E402
from autogate_router.train import main as train_main  # noqa: E402


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, pilot_jsonl, tiny_backbone):
    """configs/train_smoke.yaml on the pilot rows with the tiny fixture backbone."""
    out = tmp_path_factory.mktemp("smoke")
    history = train_main(
        [
            "--config",
            "configs/train_smoke.yaml",
            "--rows",
            str(pilot_jsonl),
            "--backbone",
            str(tiny_backbone),
            "--out-dir",
            str(out),
        ]
    )
    return out, history


def test_train_writes_best_and_history(smoke_run):
    out, history = smoke_run
    assert (out / "best" / "heads.safetensors").is_file()
    assert (out / "best" / "labels.json").is_file()
    assert (out / "best" / "tokenizer").is_dir()
    saved = json.loads((out / "history.json").read_text())
    assert saved["epochs"] and saved["best"]["swe"] == min(e["swe"] for e in saved["epochs"])
    steps = saved["steps"]
    assert steps[-1]["loss"] < steps[0]["loss"]
    for key in ("macro_f1", "swe", "intent_accuracy", "span_token_f1", "critical_miss_rate"):
        assert key in saved["epochs"][-1]
    assert history["total_steps"] == steps[-1]["step"]


def test_predict_score_calibrate(smoke_run, pilot_jsonl, tmp_path, capsys):
    out, _ = smoke_run
    ckpt = out / "best"

    calibrate_main(["--checkpoint", str(ckpt), "--rows", str(pilot_jsonl), "--n-boot", "20"])
    cal = json.loads((ckpt / "calibration.json").read_text())
    assert cal["temperature"] > 0 and cal["split"] == "val"
    assert cal["nll_after"] <= cal["nll_before"] + 1e-9
    assert 0 <= cal["ece_after"] <= 1
    scan = json.loads((ckpt / "threshold_scan.json").read_text())
    assert [s["threshold"] for s in scan["thresholds"]] == list(THRESHOLDS)
    cov = [s["coverage"] for s in scan["thresholds"]]
    assert cov == sorted(cov, reverse=True)
    for s in scan["thresholds"]:
        assert s["n_covered"] + s["clarify"]["n"] == scan["n"]
    assert "smallest threshold with covered critical miss rate < 1%" in capsys.readouterr().out

    preds_path = tmp_path / "predictions.jsonl"
    predict_main(
        [
            "--checkpoint",
            str(ckpt),
            "--rows",
            str(pilot_jsonl),
            "--split",
            "test,ood",
            "--out",
            str(preds_path),
        ]
    )
    preds = [json.loads(line) for line in preds_path.read_text().splitlines()]
    assert len(preds) == 270  # pilot test + ood
    for p in preds:
        assert tuple(p) == PREDICTION_FIELDS + TAG_PREDICTION_FIELDS
        assert p["route"] == p["route_e2e"]  # the default route source
        assert len(p["route_probs"]) == 5 and abs(sum(p["route_probs"]) - 1) < 1e-4
        assert (
            p["route"]
            == ["LOCAL", "CLOUD", "CLOUD_MASKED", "DEFER", "REFUSE"][
                max(range(5), key=p["route_probs"].__getitem__)
            ]
        )
        assert len(p["intent_probs_top3"]) == 3
        assert 0 < p["confidence"] <= 1
        for s in p["spans"]:
            assert s["end"] > s["start"] and s["label"] in {
                "contact",
                "phone",
                "address",
                "otp",
                "card",
            }

    result = score_main(["--rows", str(pilot_jsonl), "--preds", str(preds_path), "--n-boot", "20"])
    assert result["n_scored"] == 270
    assert set(result["route"]["by_split"]) == {"test", "ood"}
    assert "intent" in result and "spans" in result
    assert set(result["tags"]["by_split"]) == {"test", "ood"}
    assert set(result["tags"]["overall"]) == {
        "capability",
        "distraction",
        "actuation",
        "consequence",
    }


def test_export_onnx_matches_torch(smoke_run, tmp_path, pilot_jsonl):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from autogate_router.export import main as export_main

    out, _ = smoke_run
    report = export_main(
        [
            "--checkpoint",
            str(out / "best"),
            "--out-dir",
            str(tmp_path / "onnx"),
            "--rows",
            str(pilot_jsonl),
        ]
    )
    fp32, int8 = report["models"]["fp32"], report["models"]["int8"]
    assert fp32["max_abs_diff"] <= 1e-3
    tag_outputs = [
        "capability_logits",
        "distraction_logits",
        "actuation_logits",
        "consequence_logits",
    ]
    assert report["outputs"] == ["route_logits", "intent_logits", "span_logits", *tag_outputs]
    assert set(fp32["outputs"]) == set(report["outputs"])
    import onnxruntime as ort

    sess = ort.InferenceSession(fp32["path"], providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == report["outputs"]
    assert sess.get_outputs()[6].shape[1] == 3  # consequence: none, reversible, irreversible
    assert int8["bytes"] < fp32["bytes"]
    assert (tmp_path / "onnx" / "labels.json").is_file()
    assert (tmp_path / "onnx" / "tokenizer").is_dir()


def test_lora_checkpoint_trains_and_exports(tmp_path, pilot_jsonl, tiny_backbone):
    pytest.importorskip("onnxruntime")
    from autogate_router.export import export

    out = tmp_path / "lora"
    train_main(
        [
            "--config",
            "configs/train_smoke.yaml",
            "--rows",
            str(pilot_jsonl),
            "--backbone",
            str(tiny_backbone),
            "--out-dir",
            str(out),
            "--lora",
            "--no-context",
            "--pooling",
            "last",
            "--max-steps",
            "8",
        ]
    )
    assert (out / "best" / "adapter").is_dir()
    cfg = json.loads((out / "best" / "router_config.json").read_text())
    assert cfg["use_lora"] and not cfg["use_context"] and cfg["pooling"] == "last"
    report = export(out / "best", tmp_path / "onnx")
    assert report["models"]["fp32"]["max_abs_diff"] <= 1e-3


TAGS = {
    "capability": {"local_ok", "needs_small_local", "needs_cloud", "cloud_preferred"},
    "distraction": {"low", "medium", "high"},
    "actuation": {"none", "safety_critical", "comfort", "restricted"},
    "consequence": {"none", "reversible", "irreversible"},
}


def test_predict_write_all_schema(smoke_run, pilot_jsonl, pilot_rows, tmp_path):
    from autogate_router.tag_routing import combine, route_from_tags, row_context

    out, _ = smoke_run
    prefix = tmp_path / "preds"
    written = predict_main(
        [
            "--checkpoint",
            str(out / "best"),
            "--rows",
            str(pilot_jsonl),
            "--split",
            "test,ood",
            "--write-all",
            str(prefix),
            "--route-source",
            "tags",
        ]
    )
    names = ["preds_e2e.jsonl", "preds_tags.jsonl", "preds_combined.jsonl"]
    assert [p.name for p in written] == names
    files = {
        n.removeprefix("preds_").removesuffix(".jsonl"): [
            json.loads(line) for line in (tmp_path / n).read_text().splitlines()
        ]
        for n in names
    }
    rows = {r["id"]: r for r in pilot_rows}
    assert {len(v) for v in files.values()} == {270}
    for e2e, tag, comb in zip(files["e2e"], files["tags"], files["combined"], strict=True):
        assert e2e["id"] == tag["id"] == comb["id"]
        assert tuple(e2e) == PREDICTION_FIELDS + TAG_PREDICTION_FIELDS
        # identical apart from the top-level route
        assert {**e2e, "route": None} == {**tag, "route": None} == {**comb, "route": None}
        assert e2e["route"] == e2e["route_e2e"]
        assert tag["route"] == tag["route_tags"]
        assert comb["route"] == comb["route_combined"]
        assert set(e2e["tags"]) == set(TAGS) == set(e2e["tag_probs"])
        for name, value in e2e["tags"].items():
            assert value in TAGS[name] and set(e2e["tag_probs"][name]) == TAGS[name]
            assert abs(sum(e2e["tag_probs"][name].values()) - 1) < 1e-4
        t = e2e["tags"]
        row = rows[e2e["id"]]
        want = route_from_tags(
            t["capability"], t["distraction"], t["actuation"], row_context(row), bool(e2e["spans"])
        ).route
        assert e2e["route_tags"] == want
        assert e2e["route_combined"] == combine(e2e["route_e2e"], want, t["actuation"])


def test_old_checkpoint_predicts_without_tags(tmp_path, pilot_jsonl, tiny_backbone, tiny_tokenizer):
    """A pre-tag-head checkpoint still predicts the v0 schema; tag route sources are refused."""
    from autogate_router.model import AutoGateRouter, RouterConfig

    model = AutoGateRouter(
        RouterConfig(backbone=str(tiny_backbone), use_lora=False, tag_heads=False)
    )
    ckpt = model.save_pretrained(tmp_path / "old", tiny_tokenizer)
    cfg = json.loads((ckpt / "router_config.json").read_text())
    for key in ("tag_heads", "lambda_tags", "label_smoothing", "tag_class_weights"):
        cfg.pop(key)
    (ckpt / "router_config.json").write_text(json.dumps(cfg))
    labels = json.loads((ckpt / "labels.json").read_text())
    for key in ("capabilities", "distractions", "actuations", "consequences"):
        labels.pop(key)
    (ckpt / "labels.json").write_text(json.dumps(labels))

    args = ["--checkpoint", str(ckpt), "--rows", str(pilot_jsonl), "--split", "test"]
    path = predict_main([*args, "--out", str(tmp_path / "p.jsonl")])
    preds = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(preds) == 129 and all(tuple(p) == PREDICTION_FIELDS for p in preds)
    with pytest.raises(ValueError, match="tag heads"):
        predict_main([*args, "--route-source", "tags", "--out", str(tmp_path / "t.jsonl")])


def test_tags_smoke_end_to_end(tmp_path, pilot_jsonl, tiny_backbone, capsys):
    """train_smoke_tags.yaml: train, calibrate, predict --write-all, score x3, export."""
    pytest.importorskip("onnxruntime")
    from autogate_router.export import main as export_main

    out = tmp_path / "tags"
    rows = ["--rows", str(pilot_jsonl)]
    history = train_main(
        ["--config", "configs/train_smoke_tags.yaml", *rows]
        + ["--backbone", str(tiny_backbone), "--out-dir", str(out)]
    )
    ckpt = out / "best"
    cfg = json.loads((ckpt / "router_config.json").read_text())
    assert cfg["tag_heads"] and cfg["label_smoothing"] == 0.1
    assert set(cfg["tag_class_weights"]) == {
        "capability",
        "distraction",
        "actuation",
        "consequence",
    }
    last = history["epochs"][-1]
    for name in ("capability", "distraction", "actuation", "consequence"):
        assert 0 <= last[f"{name}_accuracy"] <= 1
    assert history["steps"][-1]["tags"] < history["steps"][0]["tags"]

    calibrate_main(["--checkpoint", str(ckpt), *rows, "--n-boot", "10"])
    predict_main(
        ["--checkpoint", str(ckpt), *rows, "--split", "test,ood"]
        + ["--write-all", str(tmp_path / "preds")]
    )
    scores = {}
    runs = tmp_path / "runs"
    (runs / "qwen3-0.6b-tags").mkdir(parents=True)
    for src in ("e2e", "tags", "combined"):
        js = runs / "qwen3-0.6b-tags" / f"score_test_ood_{src}.json"
        score_main(
            [*rows, "--preds", str(tmp_path / f"preds_{src}.jsonl"), "--split", "test,ood"]
            + ["--n-boot", "10", "--json", str(js)]
        )
        scores[src] = json.loads(js.read_text())
        assert scores[src]["n_scored"] == 270
        assert set(scores[src]["route"]["by_split"]) == {"test", "ood"}
    # the tag report depends on the tags, not on the route source
    assert scores["e2e"]["tags"] == scores["tags"]["tags"] == scores["combined"]["tags"]
    # the notebook's comparison cell reads these files
    from test_kaggle_notebook import run_comparison

    table = run_comparison(runs, capsys)
    for src in ("e2e", "tags", "combined"):
        assert f"| qwen3-0.6b-tags [{src}] | 129 |" in table
    assert "### Tags: qwen3-0.6b-tags" in table and "| ood | actuation |" in table

    report = export_main(
        ["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "onnx"), *rows]
        + ["--n-examples", "16"]
    )
    assert "actuation_logits" in report["outputs"]
    assert report["models"]["fp32"]["max_abs_diff"] <= 1e-3
