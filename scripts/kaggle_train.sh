#!/usr/bin/env bash
# Train, calibrate, score and export the AutoGate router (with tag heads) on a Kaggle or Colab
# GPU notebook, unattended: rules baseline, the tags run, its --no-context ablation, and a
# comparison of every score file under $RUNS (including a v0 run in $RUNS/qwen3-0.6b, if any).
#
#   bash scripts/kaggle_train.sh            # or: curl/clone the repo first, see below
#
# Needs: a GPU (T4 or better) and internet (pip, GitHub, Hugging Face). Works on Kaggle and Colab.
# For a private repo, add a Kaggle secret GITHUB_TOKEN and export it before running
# (the notebook version does this for you).
#
# Environment knobs (all optional):
#   REPO_URL      default https://github.com/anantha99/AutoGate.git
#   BRANCH        default claude/peaceful-pasteur-nessbd
#   WORK          default /kaggle/working on Kaggle, /content on Colab, else ~/autogate_work
#   RUNS          where artifacts go, default $WORK/runs-v1 (point it at Google Drive on Colab)
#   EPOCHS        default 3
#   RUN_ABLATION  default 1 (also train the --no-context ablation)
#   RUN_EXPORT    default 1 (ONNX fp32 + int8 for each run)
#   GITHUB_TOKEN  token for a private clone
#   CONFIG        default configs/train_qwen3_0.6b_tags_r4.yaml
#   RUN_NAME      default qwen3-0.6b-tags-r4
#   ROWS          default data/generated/full/rows.jsonl (generated if missing)
#   TRAIN_ARGS    extra flags for every training run (e.g. "--batch-size 16 --grad-accum 2")
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/anantha99/AutoGate.git}
BRANCH=${BRANCH:-claude/peaceful-pasteur-nessbd}
if [ -z "${WORK:-}" ]; then
    if [ -d /kaggle/working ]; then WORK=/kaggle/working
    elif [ -d /content ]; then WORK=/content
    else WORK="$HOME/autogate_work"; fi
fi
mkdir -p "$WORK"
EPOCHS=${EPOCHS:-3}
RUN_ABLATION=${RUN_ABLATION:-1}
RUN_EXPORT=${RUN_EXPORT:-1}
CONFIG=${CONFIG:-configs/train_qwen3_0.6b_tags_r4.yaml}
RUN_NAME=${RUN_NAME:-qwen3-0.6b-tags-r4}
TRAIN_ARGS=${TRAIN_ARGS:-}
RUNS=${RUNS:-$WORK/runs-v1}
mkdir -p "$RUNS"

log() { printf '\n=== %s | %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- clone + install
cd "$WORK"
if [ -d AutoGate/.git ]; then
    log "updating existing clone"
    git -C AutoGate fetch --depth 1 origin "$BRANCH"
    git -C AutoGate checkout -q FETCH_HEAD
else
    url="$REPO_URL"
    if [ -n "${GITHUB_TOKEN:-}" ] && [[ "$REPO_URL" == https://* ]]; then
        url="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
    fi
    log "cloning $REPO_URL @ $BRANCH"
    git clone -q --depth 1 --branch "$BRANCH" "$url" AutoGate
fi
cd AutoGate
git log --oneline -1

log "installing (keeps Kaggle's CUDA torch)"
pip install -q -e ".[train,export]"
pip uninstall -y -q torchao 2>/dev/null || true  # Colab preinstalls an old torchao that makes peft refuse to build LoRA layers
python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__, "peft", peft.__version__)
PY

# ---------------------------------------------------------------- data
ROWS=${ROWS:-data/generated/full/rows.jsonl}
if [ ! -f "$ROWS" ]; then
    log "generating the full dataset"
    python -m autogate_bench.generate --out data/generated/full --rng-seed 0 \
        --paraphrases data/paraphrases
fi
wc -l "$ROWS"

# ---------------------------------------------------------------- rules baseline (reference)
log "rules baseline on test + ood"
mkdir -p "$RUNS/rules"
python -m autogate_eval.baselines.rules --rows "$ROWS" --emit "$RUNS/rules/predictions.jsonl" \
    > "$RUNS/rules/baseline_report.txt"
python -m autogate_eval.score --rows "$ROWS" --preds "$RUNS/rules/predictions.jsonl" \
    --split test,ood --json "$RUNS/rules/score_test_ood.json" | tee "$RUNS/rules/score_test_ood.txt"

# ---------------------------------------------------------------- one full run
run() {
    local name=$1
    shift
    local out="$RUNS/$name"
    mkdir -p "$out"

    log "[$name] train ($EPOCHS epochs) $*"
    # shellcheck disable=SC2086  # TRAIN_ARGS is split on purpose
    python -m autogate_router.train --config "$CONFIG" \
        --rows "$ROWS" --out-dir "$out" --epochs "$EPOCHS" $TRAIN_ARGS "$@" 2>&1 | tee "$out/train.log"

    log "[$name] calibrate on val"
    python -m autogate_router.calibrate --checkpoint "$out/best" --rows "$ROWS" \
        --batch-size 128 2>&1 | tee "$out/calibrate.log"

    log "[$name] predict on test + ood (e2e, tags and combined routes in one pass)"
    python -m autogate_router.predict --checkpoint "$out/best" --rows "$ROWS" \
        --split test,ood --batch-size 128 --write-all "$out/predictions_test_ood"

    for src in e2e tags combined; do
        log "[$name] score ($src)"
        python -m autogate_eval.score --rows "$ROWS" --preds "$out/predictions_test_ood_$src.jsonl" \
            --split test,ood --json "$out/score_test_ood_$src.json" | tee "$out/score_test_ood_$src.txt"
    done

    if [ "$RUN_EXPORT" = 1 ]; then
        log "[$name] export ONNX + int8"
        python -m autogate_router.export --checkpoint "$out/best" --out-dir "$out/onnx" \
            --rows "$ROWS" --split val --n-examples 256 2>&1 | tee "$out/export.log"
    fi
}

run "$RUN_NAME"
if [ "$RUN_ABLATION" = 1 ]; then
    run "$RUN_NAME-no-context" --no-context
fi

# ---------------------------------------------------------------- comparison
log "comparison (markdown tables, also in $RUNS/comparison.md)"
RUNS="$RUNS" python - <<'PY' | tee "$RUNS/comparison.md"
import glob
import json
import os

RUNS = os.environ.get("RUNS", "runs")
SOURCES = ("", "e2e", "tags", "combined")


def _num(x, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


systems = []  # (sort key, label, run, score)
for path in glob.glob(os.path.join(RUNS, "*", "score_test_ood*.json")):
    run = os.path.basename(os.path.dirname(path))
    src = os.path.basename(path)[: -len(".json")].removeprefix("score_test_ood").lstrip("_")
    if src not in SOURCES:
        continue
    label = run + (f" [{src}]" if src else "")
    if run == "qwen3-0.6b" and not src:
        label = "qwen3-0.6b (v0)"
    rank = {"rules": 0, "qwen3-0.6b": 1}.get(run, 2)
    with open(path, encoding="utf-8") as f:
        systems.append(((rank, run, SOURCES.index(src)), label, run, json.load(f)))
systems.sort(key=lambda s: s[0])
if not systems:
    print(f"no score_test_ood*.json under {RUNS}")

for split in ("test", "ood"):
    print(f"\n### Routes: {split}\n")
    print("| system | n | macro-F1 | SWE | critical miss | span leak |")
    print("| --- | --- | --- | --- | --- | --- |")
    for _, label, _, sc in systems:
        m = sc["route"].get("by_split", {}).get(split)
        if not m or not m.get("n"):
            continue
        spans = (sc.get("spans") or {}).get("by_split", {}).get(split) or {}
        print(
            f"| {label} | {m['n']} | {_num(m['macro_f1'])} | {_num(m['swe'])} "
            f"| {_num(m['critical_miss_rate'], True)} | {_num(spans.get('leak_rate'), True)} |"
        )

seen = set()
for _, _, run, sc in systems:
    if run in seen or not sc.get("tags"):
        continue
    seen.add(run)  # the tag report is the same for every route source of a run
    print(f"\n### Tags: {run}\n")
    print("| split | tag | accuracy | macro-F1 | dangerous miss | route-changing errors (cost) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for split in ("test", "ood"):
        for tag, t in sc["tags"]["by_split"].get(split, {}).items():
            dm = _num(t["dangerous_miss_rate"], True)
            if t["n_dangerous"]:
                dm += f" ({t['dangerous_misses']}/{t['n_dangerous']})"
            print(
                f"| {split} | {tag} | {_num(t['accuracy'])} | {_num(t['macro_f1'])} | {dm} "
                f"| {t['route_changing_errors']} ({t['route_changing_cost']:.0f}) |"
            )
PY

# ---------------------------------------------------------------- summary
log "artifacts"
for d in "$RUNS"/*/; do
    echo "$d"
    ls -1 "$d" | sed 's/^/    /'
done
cat <<TXT

Where things are (per run directory under $RUNS):
  best/                      checkpoint: LoRA adapter, heads, router_config.json, labels.json,
                             tokenizer/, calibration.json, threshold_scan.json
  history.json, train.log    loss curve and per-epoch val metrics
  calibrate.log              temperature, ECE, threshold table
  predictions_test_ood_{e2e,tags,combined}.jsonl
                             predictions on test + ood; route = the route head (e2e), the
                             rulebook on the predicted tags (tags), or the more cautious (combined)
  score_test_ood_{e2e,tags,combined}.{txt,json}
                             route report, intent accuracy, span F1, leak rate, per-tag report
  onnx/                      router.onnx (+ .data), router.int8.onnx (+ .data), export_report.json
$RUNS/rules holds the rules baseline scored the same way; $RUNS/comparison.md has the tables.
TXT
