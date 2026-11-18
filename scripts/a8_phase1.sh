#!/bin/bash
# A8 Phase 1 (docs/plan-a8-finetune-a6.md): controls, proxy check, selection, second seeds.
# Run only after the owner's audit of docs/augment-review.md. Safe to re-run after an
# interruption: every step is skipped when its output exists (fine-tunes resume per epoch).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run --no-sync python -u -m"
FULL=data/generated/full/rows.jsonl
AUG=data/generated/full_aug/rows.jsonl
CACHE=data/generated/decide/embeddinggemma-2-a8
HOLD=configs/decide/a8_holdout.yaml
BETA=$(python3 -c "import json; print(json.load(open('configs/decide/a8_beta.json'))['beta'])")
OUT=runs/a8
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

step() { echo "== $(date -u +%H:%M:%S) $*"; }

# 0. data: the augmented set after the audit; the base set must not move
if [ ! -f "$OUT/data.ok" ]; then
  step "regenerate the augmented dataset"
  $PY autogate_bench.generate --out data/generated/full_aug --paraphrases data/paraphrases \
    --augment data/augment > /dev/null
  [ "$(sha256sum "$FULL" | cut -c1-8)" = "8cc12a4b" ] || { echo "base set moved"; exit 1; }
  n=$(wc -l < "$FULL")
  cmp -s <(head -n "$n" "$AUG") "$FULL" || { echo "augmented set is not a prefix extension"; exit 1; }
  sha256sum "$AUG" > "$OUT/data.ok"
fi
if [ ! -f "$CACHE/manifest.json" ]; then
  step "frozen-encoder cache for the augmented texts"
  $PY autogate_decide.embed --encoder embeddinggemma-2 --rows "$FULL" "$AUG" --out "$CACHE"
fi

ft() {  # name rows extra-args...
  local name=$1 rows=$2; shift 2
  if [ ! -f "$OUT/$name/finetune.json" ]; then
    step "fine-tune $name"
    $PY autogate_decide.finetune --rows "$rows" --out "$OUT/$name" --holdout "$HOLD" \
      --holdout-mode proxy --no-save --epochs 3 "$@"
  fi
}

# 1. controls
ft R7-frz "$AUG" --freeze-encoder --frozen-cache "$CACHE"
ft R7-ctl "$FULL" --full

# 2. proxy validity check
if [ ! -f "$OUT/check_proxy.json" ]; then
  $PY autogate_decide.a8 check-proxy --frz "$OUT/R7-frz" --ctl "$OUT/R7-ctl" --rows "$AUG" \
    --out "$OUT/check_proxy.json" > /dev/null
fi
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/check_proxy.json'))['valid'] else 1)" \
  || { step "STOP: the proxy does not reproduce A7's ordering; ask the owner"; exit 2; }

# 3. candidates
ft R7a "$AUG" --full
ft R7b "$AUG" --full --distill-weight "$BETA" --frozen-cache "$CACHE"

# 4. selection
if [ ! -f "$OUT/select.json" ]; then
  $PY autogate_decide.a8 select --ctl "$OUT/R7-ctl" --frz "$OUT/R7-frz" \
    --candidates "$OUT/R7a" "$OUT/R7b" --rows "$AUG" --out "$OUT/select.json" > /dev/null
fi
WIN=$(python3 -c "import json; d=json.load(open('$OUT/select.json')); print(d.get('winner',''))")
[ -n "$WIN" ] || { step "STOP: outcome C (nothing passes the val filter); ask the owner"; exit 3; }

# 5. second fine-tune seeds
ft R7-ctl-s1 "$FULL" --full --seed 1
if [ "$WIN" = R7a ]; then ft R7a-s1 "$AUG" --full --seed 1
else ft R7b-s1 "$AUG" --full --distill-weight "$BETA" --frozen-cache "$CACHE" --seed 1; fi

# 6. outcome
$PY autogate_decide.a8 outcome --ctl "$OUT/R7-ctl" "$OUT/R7-ctl-s1" \
  --winner "$OUT/$WIN" "$OUT/$WIN-s1" --frz "$OUT/R7-frz" --rows "$AUG" \
  --out "$OUT/outcome.json" > /dev/null
step "PHASE 1 DONE: outcome $(python3 -c "import json; print(json.load(open('$OUT/outcome.json'))['outcome'])")"
