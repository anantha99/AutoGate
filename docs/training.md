# Training the router

The router is a small decoder backbone (default `Qwen/Qwen3-0.6B`, any
`transformers.AutoModel` works via `--backbone`) fine-tuned with LoRA, with
three heads on one forward pass. Code: `src/autogate_router/`. The numbers
it has to beat are the rules baseline's in `docs/dataset-v0.md`.

## Install

```
uv sync --extra dev --extra train --extra export
```

`train` is torch, transformers, peft and accelerate; `export` is onnx and
onnxruntime. Under uv, torch always resolves from the PyTorch CPU index
(`[[tool.uv.index]] pytorch-cpu` in `pyproject.toml`), so a CPU box never
pulls CUDA wheels. On a GPU machine use pip on top of the machine's CUDA
torch: `pip install -e ".[train,export]"` (this is what the Kaggle script
does).

## Input

One row becomes one sequence:

```
[speed=high][conn=weak][workload=medium][privacy=standard][passenger=yes][local=tiny] arindam mukherjee is waiting on that otp it's 2182
```

that is `context_prefix + " " + utterance`, at most 96 tokens (the longest
row in the full dataset is 66 Qwen3 tokens). `--no-context` drops the prefix
and feeds the bare utterance, the PRD's ablation: without the context the
route is not determined by the input, so the gap between the two runs is the
value of the context.

## Heads and losses

| head | input | output | loss |
| --- | --- | --- | --- |
| route | pooled utterance representation | 5 logits, `Route` order | cross-entropy, class-weighted (below) |
| intent | the same pooled vector | 65 logits, `INTENTS` order | cross-entropy, auxiliary |
| span | every token's hidden state | 11 BIO logits | token cross-entropy, prefix/special/padding ignored |

`loss = route + λ_intent · intent + λ_span · span`, with `--lambda-intent
0.5` and `--lambda-span 1.0` by default. The intent head is there because
the policy gate needs the intent's tags (actuation class, consequence); the
span head finds what CLOUD_MASKED has to mask.

**Pooling** (`--pooling`). `mean` (default) averages the hidden states of
the utterance tokens only; the context prefix and padding are masked out.
The prefix still shapes those states through attention, since a causal model
reads the prefix before the utterance. `last` takes the hidden state of the
last utterance token, the one that has attended to everything. Inputs are
right-padded, so the last utterance token is well defined.

**Labels** are fixed in `autogate_router.labels` and written to
`labels.json` in every checkpoint, so a checkpoint never depends on the
import order of `Route` or `INTENTS`:
`O, B-contact, I-contact, B-phone, I-phone, B-address, I-address, B-otp,
I-otp, B-card, I-card`.

### Span labels at the subword level

Span targets come from the row's character-offset `spans`, not from its
word-level `bio`. The text goes through the fast tokenizer with
`return_offsets_mapping`; each token's character range is trimmed of
whitespace (byte-level BPE tokens carry their leading space) and made
relative to the utterance. A token that overlaps a span is `B-<label>` if
it is the span's first token and `I-<label>` otherwise, so the
continuation subwords of a name are `I-`. Utterance tokens outside every
span are `O`; prefix tokens, special tokens and padding are `-100`.
Prediction runs the same offsets backwards (`data.spans_from_tags`): B/I
runs merge into one character span, trimmed of whitespace. With the real
Qwen3 tokenizer all 8,640 sensitive rows of the full dataset round-trip
exactly, with and without the prefix; the tests check the same on 50 pilot
rows with the tiny tokenizer.

### Route class weights

The route loss weights each true class by how expensive it is to get wrong,
from `configs/cost_matrix.json`:

```
w_c = mean over p != c of cost(c, p)          (the four wrong predictions)
w_LOCAL = (w_LOCAL_SAFETY_CRITICAL + w_LOCAL_OTHER) / 2
w <- w / mean(w)                              (so the weights average 1)
```

The matrix splits the true-LOCAL row by whether the intent is
safety-critical. A per-class weight cannot see that, so LOCAL takes the
mean of its two rows. With the shipped matrix:

| route | raw mean cost | weight |
| --- | --- | --- |
| LOCAL | (8.25 + 2.25) / 2 = 5.25 | 1.207 |
| CLOUD | 2.25 | 0.517 |
| CLOUD_MASKED | 4.00 | 0.920 |
| DEFER | 2.75 | 0.632 |
| REFUSE | 7.50 | 1.724 |

`--no-class-weights` trains with a plain cross-entropy instead;
`--cost-matrix` points at another matrix.

## Run it

Every field of `TrainConfig` (`src/autogate_router/train.py`) is a flag
(`lora_r` becomes `--lora-r`; booleans take `--no-<name>`). `--config` loads
defaults from YAML, and flags given on the command line override it.

**Smoke (CPU, a few seconds of training, nothing downloaded).** A tiny
random Qwen3 (hidden 32, 2 layers, 2 heads) with a BPE tokenizer trained on
the rows:

```
uv run python -m autogate_bench.generate --out data/generated/pilot --rng-seed 0
uv run python -m autogate_router.tiny --out checkpoints/tiny-qwen3 --rows data/generated/pilot/rows.jsonl
uv run python -m autogate_router.train --config configs/train_smoke.yaml
```

The smoke config fine-tunes the whole tiny model (`lora: false`) because
LoRA keeps the embeddings frozen and a frozen random embedding leaves the
adapters little to learn; `--lora` runs the LoRA path anyway. Its numbers
only show that the loop works (loss 6.27 to 1.53 over 216 steps; val
macro-F1 around 0.2).

**Full (one GPU).** Qwen3-0.6B, LoRA r=16, alpha=32 on the attention and MLP
projections, 3 epochs, batch 32, lr 2e-4:

```
uv run python -m autogate_bench.generate --out data/generated/full --rng-seed 0 --paraphrases data/paraphrases
python -m autogate_router.train --config configs/train_qwen3_0.6b.yaml \
    --rows data/generated/full/rows.jsonl --out-dir checkpoints/qwen3-0.6b
```

**No-context ablation.** The same with the prefix dropped:

```
python -m autogate_router.train --config configs/train_qwen3_0.6b.yaml \
    --rows data/generated/full/rows.jsonl --out-dir checkpoints/qwen3-0.6b-no-context --no-context
```

**Kaggle.** `scripts/kaggle_train.sh` (or `notebooks/kaggle_train.ipynb`,
the same steps as cells) runs everything unattended on a GPU notebook with
Internet on: clone, install, generate, rules baseline, then for the main run
and the ablation train, calibrate, predict and score test + ood, and export.
Everything lands in `/kaggle/working/runs/<run>/`. Knobs are environment
variables (`EPOCHS`, `RUN_ABLATION`, `RUN_EXPORT`, `TRAIN_ARGS`, ...; see the
script header); for a private repo set a `GITHUB_TOKEN` secret.

Other useful flags: `--no-lora` (full fine-tuning), `--lora-target-modules
q_proj,v_proj`, `--head-lr` (a separate learning rate for the heads),
`--grad-accum`, `--precision {auto,bf16,fp16,fp32}` (auto: bf16 on GPUs that
support it, fp16 with a grad scaler otherwise, fp32 on CPU), `--max-steps`
and `--limit-rows` (a seeded sample of that many rows from each of train
and val) for quick runs.

## The loop

AdamW (weight decay on matrices only), linear warmup (`--warmup-ratio`,
default 6% of the steps) then linear decay to 0, gradient accumulation,
gradient clipping at 1.0, autocast on CUDA, fixed seeds. A log line every
`--log-every` optimizer steps gives the windowed total and per-head losses.
At the end of each epoch (and when `--max-steps` stops early) the val split
is scored: route SWE, macro-F1, accuracy, MCC and critical miss rate from
`autogate_eval.metrics.evaluate`, intent accuracy, and span token-F1 (micro
F1 over non-`O` subword tags, exact tag match). The checkpoint with the
lowest val SWE (ties: higher macro-F1) is saved.

## Artifacts

| path | what |
| --- | --- |
| `<out>/train_config.json` | the resolved config |
| `<out>/history.json` | the step log (loss curve), per-epoch val metrics, the best epoch |
| `<out>/best/router_config.json` | backbone name, pooling, use_context, max_length, LoRA settings, class weights |
| `<out>/best/labels.json` | the three label vocabularies, in logit order |
| `<out>/best/heads.safetensors` | the three heads |
| `<out>/best/adapter/` | the LoRA adapter (the base backbone is loaded by name) |
| `<out>/best/backbone/` | the full backbone instead, with `--no-lora` |
| `<out>/best/tokenizer/` | the tokenizer |
| `<out>/best/calibration.json` | temperature and ECE before/after (from `calibrate`) |
| `<out>/best/threshold_scan.json` | the confidence-threshold scan (from `calibrate`) |
| `<out>/best/onnx/` | `router.onnx`, `router.int8.onnx`, `export_report.json`, plus config, labels, tokenizer (from `export`) |

## Predict and score

```
python -m autogate_router.predict --checkpoint checkpoints/qwen3-0.6b/best \
    --rows data/generated/full/rows.jsonl --split test,ood --out predictions.jsonl
python -m autogate_eval.score --rows data/generated/full/rows.jsonl --preds predictions.jsonl
```

`predictions.jsonl` has one line per row:

```json
{"id": "send_text_message/3-0-1", "route": "CLOUD_MASKED",
 "route_probs": [0.01, 0.02, 0.95, 0.01, 0.01],
 "intent": "send_text_message",
 "intent_probs_top3": [["send_text_message", 0.91], ["reply_to_message", 0.05], ["take_note", 0.01]],
 "spans": [{"start": 0, "end": 17, "label": "contact", "text": "arindam mukherjee"}],
 "confidence": 0.93}
```

`route_probs` is the plain softmax in `labels.json` order; `confidence` is
the max route probability after the temperature in `calibration.json` when
that file sits beside the checkpoint, the plain max otherwise.

`autogate_eval.score` scores **any** file with this schema; only `id` and
`route` are required. It joins on `id` (rows without a prediction are left
out, so a test + ood file can be scored against the full rows file; `--split`
restricts further) and prints the route report from `metrics.report` /
`format_report`, intent accuracy when predictions carry `intent`, and, when
they carry `spans`, exact-match span precision/recall/F1 and the **leak
rate**: over rows whose predicted route is CLOUD or CLOUD_MASKED, the
fraction of true sensitive spans not covered by any predicted span (a span
is covered when each of its non-space characters lies inside a predicted
span). `--json` writes everything.

The rules baseline writes the same schema:

```
uv run python -m autogate_eval.baselines.rules --rows data/generated/full/rows.jsonl --emit rules_predictions.jsonl
uv run python -m autogate_eval.score --rows data/generated/full/rows.jsonl --preds rules_predictions.jsonl --split test,ood
```

(route, matched intent and the regex spans; `route_probs`,
`intent_probs_top3` and `confidence` are null).

## Calibrate and choose a threshold

```
python -m autogate_router.calibrate --checkpoint checkpoints/qwen3-0.6b/best --rows data/generated/full/rows.jsonl
```

This fits one temperature on the val split's route logits (NLL, LBFGS on
log T checked against a grid), writes `calibration.json` (`temperature`,
`ece_before`, `ece_after`, NLL before/after; ECE with 15 equal-width bins),
and scans confidence thresholds 0.50 to 0.95 in steps of 0.05, then 0.96 to
0.99. At each threshold the rows at or above it are covered: the gate runs
the router's route, and the table shows coverage, SWE, accuracy and critical
miss rate on them. The rest go to CLARIFY (neither execute nor refuse; ask
again), reported separately because CLARIFY has no cost-matrix entry: how
many rows, how many router errors and critical misses it absorbs, and how
many true safety-critical commands it delays. Nothing is chosen
automatically; the output names the smallest threshold whose covered
critical miss rate is below 1%, and the owner picks the operating point.

## Export

```
python -m autogate_router.export --checkpoint checkpoints/qwen3-0.6b/best \
    --out-dir checkpoints/qwen3-0.6b/onnx --rows data/generated/full/rows.jsonl
```

Merges the LoRA adapter into the backbone, traces the three-head model with
eager attention (TorchScript exporter, opset 17; inputs `input_ids`,
`attention_mask`, `utterance_mask`, int64, dynamic batch and sequence;
outputs `route_logits`, `intent_logits`, `span_logits`), and quantises it
with `onnxruntime.quantization.quantize_dynamic` to int8 (`--per-channel`
for per-channel scales). fp32 must match torch within `--atol` (1e-3) on
`--n-examples` val rows in two padded shapes; for int8 the per-head argmax
agreement with torch is reported, not enforced. Models over 2 GB keep their
weights in one external `.data` file beside the graph.

`utterance_mask` is 1 on utterance tokens: build it with
`autogate_router.data.encode_row`, which also gives the offsets for turning
span tags back into character spans.

Sizes: the tiny smoke model is 0.28 MB fp32 and 0.18 MB int8. A 4-step
CPU check of the real Qwen3-0.6B + LoRA exported to 2,276 MB fp32
(`router.onnx` + `router.onnx.data`, max difference to torch 1.6e-5) and
571 MB int8. On that barely trained model the int8 route argmax agreed with
torch on 27 of 32 val rows, because its logits were nearly tied; score the
int8 model on real rows before shipping it.
