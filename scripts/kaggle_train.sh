#!/usr/bin/env bash
# Train, calibrate, score and export the AutoGate router on a Kaggle GPU notebook, unattended.
#
#   bash scripts/kaggle_train.sh            # or: curl/clone the repo first, see below
#
# Needs: a GPU accelerator (T4 or better) and Internet enabled (pip, GitHub, Hugging Face).
# For a private repo, add a Kaggle secret GITHUB_TOKEN and export it before running
# (the notebook version does this for you).
#
# Environment knobs (all optional):
#   REPO_URL      default https://github.com/anantha99/AutoGate.git
#   BRANCH        default claude/peaceful-pasteur-nessbd
#   WORK          default /kaggle/working (artifacts land in $WORK/runs)
#   EPOCHS        default 3
#   RUN_ABLATION  default 1 (also train the --no-context ablation)
#   RUN_EXPORT    default 1 (ONNX fp32 + int8 for each run)
#   GITHUB_TOKEN  token for a private clone
#   CONFIG        default configs/train_qwen3_0.6b.yaml
#   ROWS          default data/generated/full/rows.jsonl (generated if missing)
#   TRAIN_ARGS    extra flags for every training run (e.g. "--batch-size 16 --grad-accum 2")
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/anantha99/AutoGate.git}
BRANCH=${BRANCH:-claude/peaceful-pasteur-nessbd}
WORK=${WORK:-/kaggle/working}
EPOCHS=${EPOCHS:-3}
RUN_ABLATION=${RUN_ABLATION:-1}
RUN_EXPORT=${RUN_EXPORT:-1}
CONFIG=${CONFIG:-configs/train_qwen3_0.6b.yaml}
TRAIN_ARGS=${TRAIN_ARGS:-}
RUNS="$WORK/runs"
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

    log "[$name] predict on test + ood"
    python -m autogate_router.predict --checkpoint "$out/best" --rows "$ROWS" \
        --split test,ood --batch-size 128 --out "$out/predictions_test_ood.jsonl"

    log "[$name] score"
    python -m autogate_eval.score --rows "$ROWS" --preds "$out/predictions_test_ood.jsonl" --split test,ood \
        --json "$out/score_test_ood.json" | tee "$out/score_test_ood.txt"

    if [ "$RUN_EXPORT" = 1 ]; then
        log "[$name] export ONNX + int8"
        python -m autogate_router.export --checkpoint "$out/best" --out-dir "$out/onnx" \
            --rows "$ROWS" --split val --n-examples 256 2>&1 | tee "$out/export.log"
    fi
}

run qwen3-0.6b
if [ "$RUN_ABLATION" = 1 ]; then
    run qwen3-0.6b-no-context --no-context
fi

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
  predictions_test_ood.jsonl predictions on test + ood
  score_test_ood.{txt,json}  route report, intent accuracy, span F1, leak rate
  onnx/                      router.onnx (+ .data), router.int8.onnx (+ .data), export_report.json
$RUNS/rules holds the rules baseline scored the same way for comparison.
TXT
