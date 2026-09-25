"""End to end on CPU: train -> predict -> score -> calibrate -> export (tiny backbone)."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from autogate_eval.score import main as score_main  # noqa: E402
from autogate_router.calibrate import THRESHOLDS  # noqa: E402
from autogate_router.calibrate import main as calibrate_main  # noqa: E402
from autogate_router.predict import PREDICTION_FIELDS  # noqa: E402
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
    assert len(preds) == 213  # pilot test + ood
    for p in preds:
        assert tuple(p) == PREDICTION_FIELDS
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
    assert result["n_scored"] == 213
    assert set(result["route"]["by_split"]) == {"test", "ood"}
    assert "intent" in result and "spans" in result


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
