"""The Kaggle/Colab notebook and script: syntax, required steps, and the comparison cell."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
SCRIPT = ROOT / "scripts" / "kaggle_train.sh"


def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _code(predicate):
    return [s for kind, s in _cells() if kind == "code" and predicate(s)]


def comparison_source() -> str:
    (src,) = _code(lambda s: "### Routes:" in s)
    return src


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


@needs_bash
def test_bash_cells_and_script_parse():
    bash = _code(lambda s: s.startswith("%%bash"))
    assert len(bash) == 4  # install+generate, rules, tags run, ablation
    for src in bash:
        body = src.split("\n", 1)[1]
        r = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True)
        assert r.returncode == 0, r.stderr
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_python_cells_compile():
    py = _code(lambda s: not s.startswith("%%"))
    assert len(py) == 3  # setup, Drive, comparison
    for i, src in enumerate(py):
        compile(src, f"cell{i}", "exec")


def test_notebook_steps():
    text = "\n".join(s for _, s in _cells())
    assert 'os.environ["BRANCH"] = "claude/peaceful-pasteur-nessbd"' in text
    assert "/kaggle/working" in text and "/content" in text  # platform detection
    assert "drive.mount" in text and "pip uninstall -y -q torchao" in text
    assert "configs/train_qwen3_0.6b_tags_r4.yaml" in text
    assert 'os.environ["RUN_NAME"] = "qwen3-0.6b-tags-r5"' in text
    assert 'os.environ["AUGMENT"] = "data/augment"' in text
    assert '[ -d "$AUGMENT" ]' in text and '--augment "$AUGMENT"' in text
    assert '"$RUNS/$RUN_NAME"' in text and '"$RUNS/$RUN_NAME-no-context"' in text
    # test + ood for both runs, val for the main run only
    assert text.count("--write-all") == 3 and text.count("autogate_router.export") == 2
    assert "--split test,ood --json" in text
    (main_run,) = _code(lambda s: '"$RUNS/$RUN_NAME";' in s)
    assert '--split val --batch-size 128 --write-all "$OUT/predictions_val"' in main_run
    assert '--split val --json "$OUT/score_val_$src.json"' in main_run
    (ablation,) = _code(lambda s: '"$RUNS/$RUN_NAME-no-context";' in s)
    assert "predictions_val" not in ablation
    kinds = [k for k, _ in _cells()]
    i_main = next(i for i, (_, s) in enumerate(_cells()) if '"$RUNS/$RUN_NAME";' in s)
    assert kinds[i_main - 1] == "markdown"  # what the tags run tests
    assert _cells()[-1][1] == comparison_source()  # the comparison is the last cell
    script = SCRIPT.read_text(encoding="utf-8")
    assert comparison_source() in script
    assert 'run "$RUN_NAME-no-context" --no-context' in script
    assert "--write-all" in script and "for src in e2e tags combined" in script
    assert "RUN_NAME=${RUN_NAME:-qwen3-0.6b-tags-r5}" in script
    assert "AUGMENT=${AUGMENT:-data/augment}" in script
    assert 'GEN_ARGS+=(--augment "$AUGMENT")' in script
    assert '--write-all "$out/predictions_val"' in script and "score_val_$src.json" in script


def run_comparison(runs: Path, capsys) -> str:
    import os

    old = os.environ.get("RUNS")
    os.environ["RUNS"] = str(runs)
    try:
        exec(compile(comparison_source(), "comparison", "exec"), {})
    finally:
        if old is None:
            os.environ.pop("RUNS")
        else:
            os.environ["RUNS"] = old
    return capsys.readouterr().out


def test_comparison_cell_on_hand_made_scores(tmp_path, capsys):
    def score(macro, tags=None):
        split = {"n": 10, "macro_f1": macro, "swe": 1.0, "critical_miss_rate": 0.25}
        out = {
            "route": {"by_split": {"test": split, "ood": {**split, "macro_f1": macro / 2}}},
            "spans": {"by_split": {"test": {"leak_rate": 0.5}, "ood": {"leak_rate": None}}},
        }
        if tags:
            out["tags"] = tags
        return out

    tag = {
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "dangerous_miss_rate": 0.1,
        "dangerous_misses": 1,
        "n_dangerous": 10,
        "route_changing_errors": 2,
        "route_changing_cost": 7.0,
    }
    tags = {"by_split": {"test": {"actuation": tag}, "ood": {"actuation": tag}}}
    files = {
        "rules/score_test_ood.json": score(0.6),
        "qwen3-0.6b/score_test_ood.json": score(0.7),
        "qwen3-0.6b-tags/score_test_ood_e2e.json": score(0.8, tags),
        "qwen3-0.6b-tags/score_test_ood_tags.json": score(0.81, tags),
        "qwen3-0.6b-tags/score_test_ood_combined.json": score(0.82, tags),
    }
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(json.dumps(data))
    out = run_comparison(tmp_path, capsys)
    test_table = out.split("### Routes: test")[1].split("### Routes: ood")[0]
    labels = [
        line.split("|")[1].strip() for line in test_table.splitlines() if line.startswith("| ")
    ]
    assert labels == [
        "system",
        "---",
        "rules",
        "qwen3-0.6b (v0)",
        "qwen3-0.6b-tags [e2e]",
        "qwen3-0.6b-tags [tags]",
        "qwen3-0.6b-tags [combined]",
    ]
    assert "| rules | 10 | 0.600 | 1.000 | 25.0% | 50.0% |" in out
    assert "| qwen3-0.6b-tags [e2e] | 10 | 0.400 | 1.000 | 25.0% | - |" in out  # ood
    assert out.count("### Tags:") == 1
    assert "| ood | actuation | 0.900 | 0.800 | 10.0% (1/10) | 2 (7) |" in out
