"""A8 (docs/plan-a8-finetune-a6.md): holdout config, holdout application, selection rules,
and the fine-tune options. CPU only, no downloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autogate_bench.augment import DEFAULT_AUGMENT_DIR, PAIRS_FILE, load_augmentation, load_pairs
from autogate_bench.dataset import build_rows
from autogate_bench.intents import INTENTS
from autogate_decide.holdout import DEFAULT_HOLDOUT_PATH, Holdout, apply, leaking_goals

FIXTURE = Path(__file__).parent / "fixtures" / "augment"
TAGS = {i.name: (str(i.capability), str(i.distraction), str(i.actuation)) for i in INTENTS}


@pytest.fixture(scope="module")
def aug_rows():
    return [r.to_dict() for r in build_rows(rng_seed=0, augment=load_augmentation(FIXTURE))]


def test_committed_holdout_mirrors_ood_tags_and_keeps_probe_clean(pilot_rows):
    h = Holdout.load(DEFAULT_HOLDOUT_PATH)
    split = {}
    for r in pilot_rows:
        split.setdefault(r["intent"], set()).add(r["split"])
    assert len(h.proxy) == 8 and len(set(h.proxy.values())) == 8
    for ood, proxy in h.proxy.items():
        assert split[ood] == {"ood"}
        assert "train" in split[proxy]
        assert TAGS[proxy] == TAGS[ood], (ood, proxy)
    assert set(h.keywords) == set(h.proxy.values())
    pairs = {p.id: p for p in load_pairs(DEFAULT_AUGMENT_DIR / PAIRS_FILE)}
    assert 0.17 <= len(h.probe_pairs) / len(pairs) <= 0.23
    for pid in h.probe_pairs:
        p = pairs[pid]
        assert not {p.a.intent, p.b.intent} & h.proxy_intents, pid


def test_apply_moves_proxy_rows_and_drops_leaks(aug_rows):
    goal = next(r for r in aug_rows if r.get("is_synthetic"))
    pair_rows = [r for r in aug_rows if r.get("pair_id")]
    pair = pair_rows[0]
    proxy_intent = pair["intent"]
    h = Holdout(
        proxy={"x": proxy_intent},
        keywords={proxy_intent: (tuple(goal["intent"].split("_")[:1]),)},
        probe_pairs=(),
    )
    out, rep = apply(aug_rows, h, {goal["intent"]: goal["intent"].replace("_", " ")})
    assert not [r for r in out if r["intent"] == proxy_intent and r["split"] in ("train", "val")]
    assert any(r["split"] == "proxy" for r in out)
    assert not [r for r in out if r.get("pair_id") == pair["pair_id"]]
    assert pair["pair_id"] in rep["dropped_pairs"]
    assert goal["intent"] in rep["dropped_goals"]
    assert {r["split"] for r in out if r["intent"] == proxy_intent} <= {"proxy", "test"}
    final, frep = apply(aug_rows, h, {}, proxy=False)  # the final model: nothing dropped
    assert len(final) == len(aug_rows) and not frep["dropped_pairs"]


def test_probe_pairs_are_never_trained(aug_rows):
    pid = next(r["pair_id"] for r in aug_rows if r.get("pair_id"))
    h = Holdout(proxy={}, keywords={}, probe_pairs=(pid,))
    for proxy in (True, False):
        out, _ = apply(aug_rows, h, {}, proxy=proxy)
        assert {r["split"] for r in out if r.get("pair_id") == pid} == {"probe"}


def test_goal_slot_fills_do_not_count_as_leaks(aug_rows):
    # a proxy intent whose rows share only an invented name with a goal is not a leak
    h = Holdout(proxy={"x": "call_contact"}, keywords={}, probe_pairs=())
    leaks = leaking_goals(aug_rows, h)
    for why in leaks.values():
        assert "Jaccard" in why


# --------------------------------------------------------------------------- #
# Selection rules
# --------------------------------------------------------------------------- #


def _rows_file(tmp_path: Path) -> tuple[Path, list[str]]:
    rows = []
    for intent in ("a", "b", "c", "d"):
        for j in range(20):
            sc = j % 2 == 0
            rows.append(
                {
                    "id": f"{intent}/{j}",
                    "intent": intent,
                    "route": "LOCAL" if sc else "REFUSE",
                    "actuation": "safety_critical" if sc else "none",
                    "distraction": "high" if not sc else "low",
                    "split": "proxy",
                }
            )
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, [r["id"] for r in rows]


def _run(tmp_path: Path, name: str, ids, wrong: float, val_swe: float, seed: int = 0) -> Path:
    """A fake finetune output: a fraction ``wrong`` of proxy rows predicted CLOUD."""
    d = tmp_path / name
    d.mkdir()
    rng = np.random.default_rng(seed)
    hist = []
    for epoch in (1, 2):
        bad = rng.random(len(ids)) < wrong
        routes = [
            "CLOUD" if b else ("LOCAL" if i % 2 == 0 else "REFUSE")
            for i, b in zip(range(len(ids)), bad, strict=True)
        ]
        dis = [
            "low" if b else ("low" if i % 2 == 0 else "high")
            for i, b in zip(range(len(ids)), bad, strict=True)
        ]
        (d / f"epoch{epoch}_proxy.json").write_text(
            json.dumps({"ids": list(ids), "routes": routes, "distraction": dis})
        )
        hist.append(
            {
                "epoch": epoch,
                "swe": val_swe,
                "critical_miss_rate": 0.05,
                "proxy": {
                    "critical_miss_rate": float(np.mean(bad)),
                    "swe": float(np.mean(bad)) * 8,
                },
            }
        )
    (d / "finetune.json").write_text(json.dumps({"history": hist}))
    return d


def test_selection_rules_on_fake_runs(tmp_path):
    from autogate_decide import a8

    rows, ids = _rows_file(tmp_path)
    frz = _run(tmp_path, "frz", ids, 0.05, 0.60)
    ctl = _run(tmp_path, "ctl", ids, 0.60, 0.33, seed=1)
    good = _run(tmp_path, "good", ids, 0.02, 0.34, seed=2)
    off = _run(tmp_path, "off", ids, 0.00, 0.50, seed=3)  # fails the val filter
    assert a8.check_proxy(frz, ctl, rows)["valid"]
    sel = a8.select(ctl, frz, [good, off], rows)
    assert sel["winner"] == "good"
    assert a8.select(ctl, frz, [off], rows)["outcome"] == "C"
    assert a8.outcome([ctl], [good], frz, rows)["outcome"] == "A"
    worse = _run(tmp_path, "worse", ids, 0.55, 0.33, seed=4)
    assert a8.outcome([ctl], [worse], frz, rows)["outcome"] == "B"


# --------------------------------------------------------------------------- #
# Fine-tune options
# --------------------------------------------------------------------------- #


def test_finetune_a8_options(tmp_path, tiny_backbone, aug_rows):
    pytest.importorskip("peft")
    from autogate_decide.embed import build_cache
    from autogate_decide.encoders import EncoderSpec, FakeEncoder
    from autogate_decide.finetune import finetune
    from autogate_decide.holdout import pick_proxy

    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in aug_rows))
    proxy = pick_proxy(aug_rows, seed=0)
    hold = tmp_path / "holdout.yaml"
    probe = next(r["pair_id"] for r in aug_rows if r.get("pair_id"))
    hold.write_text(json.dumps({"seed": 0, "proxy": proxy, "keywords": {}, "probe_pairs": [probe]}))
    import transformers

    dim = transformers.AutoConfig.from_pretrained(tiny_backbone).hidden_size
    cache = build_cache(FakeEncoder(EncoderSpec("fake", "fake", None, "", dim)), [rows])
    cache.save(tmp_path / "cache")
    kw = dict(
        model_id=str(tiny_backbone),
        revision=None,
        prefix="",
        text_only=False,
        batch_size=32,
        lr=1e-3,
        epochs=2,
        holdout=hold,
        frozen_cache=tmp_path / "cache",
    )
    rep = finetune(rows, tmp_path / "b", full=True, distill_weight=1.0, save_encoder=False, **kw)
    assert all("proxy" in h for h in rep["history"])
    assert (tmp_path / "b" / "epoch2_proxy.json").exists()
    assert not (tmp_path / "b" / "encoder").exists()
    assert rep["holdout_report"]["rows_moved"]["proxy"] > 0
    frz = finetune(rows, tmp_path / "f", freeze_encoder=True, save_encoder=False, **kw)
    assert frz["mode"] == "frozen" and len(frz["history"]) == 2
    last = finetune(
        rows,
        tmp_path / "l",
        full=True,
        keep="last",
        holdout_mode="final",
        **{**kw, "holdout": hold},
    )
    assert last["best"]["epoch"] == 2 and (tmp_path / "l" / "encoder").is_dir()
