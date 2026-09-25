"""Cost matrix loading and the route metrics."""

import copy
import json

import pytest

from autogate_bench import Route
from autogate_bench.dataset import build_rows
from autogate_eval.cost_matrix import (
    DEFAULT_COST_MATRIX_PATH,
    CostMatrix,
    cost,
    default_cost_matrix,
)
from autogate_eval.metrics import evaluate, format_report, mcc_from_confusion, report

L, C, CM, D, R = Route.LOCAL, Route.CLOUD, Route.CLOUD_MASKED, Route.DEFER, Route.REFUSE
RAW = json.loads(DEFAULT_COST_MATRIX_PATH.read_text())


# ---- cost matrix ---------------------------------------------------------- #


def test_default_matrix_matches_the_prd():
    assert cost(L, C, safety_critical=True) == 10
    assert cost(L, D, safety_critical=True) == 8
    assert cost(L, R, safety_critical=True) == 5
    assert cost(L, C, safety_critical=False) == 2
    assert cost(L, R, safety_critical=False) == 3
    assert cost(CM, C, safety_critical=False) == 8
    assert cost(C, CM, safety_critical=False) == 1
    assert cost(D, C, safety_critical=False) == 4
    assert cost(R, L, safety_critical=False) == 6
    assert cost(R, CM, safety_critical=False) == 10
    for r in Route:
        assert cost(r, r, True) == cost(r, r, False) == 0
    # safety_critical only splits the LOCAL row
    assert cost(R, C, True) == cost(R, C, False)


def test_missing_cell_is_rejected(tmp_path):
    bad = copy.deepcopy(RAW)
    del bad["costs"]["DEFER"]["CLOUD"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exactly the columns"):
        CostMatrix.load(p)


def test_missing_row_is_rejected():
    bad = copy.deepcopy(RAW)
    del bad["costs"]["LOCAL_SAFETY_CRITICAL"]
    with pytest.raises(ValueError, match="rows must be exactly"):
        CostMatrix.from_dict(bad)


@pytest.mark.parametrize(("row", "col"), [("LOCAL_OTHER", "LOCAL"), ("REFUSE", "REFUSE")])
def test_nonzero_diagonal_is_rejected(row, col):
    bad = copy.deepcopy(RAW)
    bad["costs"][row][col] = 1
    with pytest.raises(ValueError, match="must be 0"):
        CostMatrix.from_dict(bad)


def test_negative_cost_is_rejected():
    bad = copy.deepcopy(RAW)
    bad["costs"]["CLOUD"]["DEFER"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        CostMatrix.from_dict(bad)


# ---- metrics --------------------------------------------------------------- #


def _row(route, actuation="none", **flags):
    return {"route": str(route), "actuation": actuation, "split": "test", **flags}


def test_perfect_predictions():
    rows = build_rows(rng_seed=0)
    m = evaluate(rows, [r.route for r in rows])
    assert m["swe"] == 0 and m["swe_ci95"] == [0.0, 0.0]
    assert m["macro_f1"] == 1.0 and m["mcc"] == pytest.approx(1.0)
    assert m["critical_misses"] == 0 and m["critical_miss_rate"] == 0.0
    rep = report(rows, [r.route for r in rows], n_boot=50)
    for sub in [*rep["by_split"].values(), *rep["by_slice"].values()]:
        assert sub["n"] > 0 and sub["swe"] == 0 and sub["macro_f1"] == 1.0


def test_hand_built_case():
    rows = [
        _row(L, "safety_critical"),  # -> CLOUD: cost 10, critical
        _row(L, "comfort"),  # -> REFUSE: cost 3
        _row(R, "restricted"),  # -> LOCAL: cost 6, critical
        _row(CM),  # -> CLOUD: cost 8 (leaked span, not "critical")
        _row(D),  # -> DEFER: cost 0
        _row(R),  # -> DEFER: cost 4, safe miss
    ]
    preds = [C, R, L, C, D, D]
    m = evaluate(rows, preds, n_boot=200)
    assert m["swe"] == pytest.approx(31 / 6)
    assert m["critical_misses"] == 2 and m["n_critical"] == 3
    assert m["critical_miss_rate"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(1 / 6)
    assert m["confusion"]["matrix"] == [
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0],
    ]
    # F1: LOCAL 0, CLOUD 0, CLOUD_MASKED 0, DEFER 2/3, REFUSE 0 -> 2/15
    assert m["macro_f1"] == pytest.approx((2 / 3) / 5)
    lo, hi = m["swe_ci95"]
    assert lo <= m["swe"] <= hi


def test_bootstrap_is_deterministic():
    rows = [_row(L), _row(C), _row(R), _row(D)] * 5
    preds = [L, CM, D, D] * 5
    assert evaluate(rows, preds)["swe_ci95"] == evaluate(rows, preds)["swe_ci95"]


def test_mcc_known_values():
    import numpy as np

    assert mcc_from_confusion(np.diag([3, 4, 5, 0, 0])) == pytest.approx(1.0)
    assert mcc_from_confusion(np.array([[0, 5], [5, 0]])) == pytest.approx(-1.0)
    assert mcc_from_confusion(np.array([[5, 0], [0, 0]])) == 0.0


def test_length_mismatch():
    with pytest.raises(ValueError):
        evaluate([_row(L)], [L, L])


def test_report_structure_and_format():
    rows = [
        _row(L, sensitive=True, adversarial=False, split="train"),
        _row(R, sensitive=False, adversarial=True, split="test"),
    ]
    rep = report(rows, [L, L], slices=("sensitive", "adversarial"), n_boot=20)
    assert list(rep["by_split"]) == ["train", "test"]
    assert rep["by_slice"]["sensitive"]["n"] == 1
    assert rep["by_slice"]["adversarial"]["critical_misses"] == 1
    text = format_report(rep)
    assert "| overall | 2 |" in text and "slice: adversarial" in text
    empty = report(rows, [L, L], slices=("sensitive",), by_split=False, n_boot=5)
    assert "by_split" not in empty


def test_default_matrix_is_cached():
    assert default_cost_matrix() is default_cost_matrix()
