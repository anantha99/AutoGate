# AutoGate

An open router and benchmark that decides, per utterance, whether an in-car
LLM assistant should answer on the head unit (`LOCAL`), in the cloud
(`CLOUD`), in the cloud with sensitive spans masked (`CLOUD_MASKED`), wait
for connectivity (`DEFER`), or not run at all (`REFUSE`).

Status: week 1. The rulebook, policy tables, intent taxonomy, seed-only data
pipeline, cost matrix, metrics and rules baseline are in place, as are the
paraphrase format, validator and ASR augmenter; the paraphrases themselves
and the router follow.

## Layout

| Package | Contents |
| --- | --- |
| `autogate_bench` | Schema, policy tables, rulebook labeler, intent taxonomy (`docs/taxonomy.md`), seeds, paraphrase format and validator (`docs/paraphrase-spec.md`), ASR augmenter, span injector, context sampler, splits, dataset writer |
| `autogate_router` | Router labels, data, model (backbone + LoRA + route/intent/span heads), training, prediction, calibration, ONNX export (`docs/training.md`) |
| `autogate_eval` | Cost matrix, safety-weighted error and other metrics, rules baseline, `score` for any predictions file, policy gate (weeks 1-3) |
| `configs/` | `cost_matrix.json`: the PRD cost matrix behind safety-weighted error |
| `docs/pilot.md` | The seed-only pilot: commands, counts, flip profiles, rules baseline results |

## The rulebook

`autogate_bench.rulebook.route(intent, context, has_sensitive_spans)` applies
the precedence tree and returns the route plus the branch that fired:

```
safety-critical command       -> LOCAL
restricted actuation, moving  -> REFUSE
too distracting for demand    -> REFUSE
local model can handle it     -> LOCAL
no connectivity, cloud-preferred -> LOCAL (onboard fallback)
no connectivity               -> DEFER
sensitive spans, strict mode  -> LOCAL   (configurable)
sensitive spans               -> CLOUD_MASKED
otherwise                     -> CLOUD
```

The tree contains no policy of its own. Everything that could be argued about
is data in `autogate_bench.policy.Policy`, which round-trips through JSON so
an OEM can substitute its own policy and re-label the benchmark.

**Table 1: driving demand** from speed and workload. A passenger relaxes the
result one step for the distraction gate only, never below `low`, and never
for actuation.

| speed \ workload | low | medium | high |
| --- | --- | --- | --- |
| parked | parked | parked | parked |
| low | low | medium | high |
| medium | medium | medium | high |
| high | medium | high | high |

**Table 2: distraction gate.** Each intent carries one distraction level:
`low` is a short spoken answer or a vehicle command; `medium` is long spoken
output or a decision; `high` needs the screen or sustained interaction, the
same set Android Automotive's UX restrictions block while moving.

| distraction \ demand | parked | low | medium | high |
| --- | --- | --- | --- | --- |
| low | allow | allow | allow | allow |
| medium | allow | allow | allow | refuse |
| high | allow | refuse | refuse | refuse |

**Table 3: actuation gate.** Each vehicle-control intent carries one class.

| class | parked | moving |
| --- | --- | --- |
| safety_critical (defrost, wipers, hazards) | LOCAL, bypasses everything | LOCAL, bypasses everything |
| comfort (HVAC, seats, volume, drive mode) | allow | allow |
| restricted (ADAS off, door unlock, trunk, park brake) | allow | refuse |

Distraction blocks are `REFUSE`, not `DEFER`: deferring means the car acts
later without being asked again, which is right for a dropped signal and
wrong for reading your messages aloud the moment you park.

## Generate the pilot

```
uv run python -m autogate_bench.generate --out data/generated/pilot --rng-seed 0
uv run python -m autogate_eval.baselines.rules --rows data/generated/pilot/rows.jsonl
```

The first writes `rows.jsonl`, `rows.parquet` and `manifest.json` (`rows.jsonl` is
byte-identical for the same arguments); the second prints the rules baseline's report. Counts
and results are in `docs/pilot.md`.

## Paraphrases

Each seed is expanded into 40 paraphrases (24 plain, 6 Hinglish, 4
Kannada-English, 6 disfluent) in `data/paraphrases/<intent>.yaml`, written by
generation agents that follow `docs/paraphrase-spec.md`. Four ASR-noised
copies per seed are added in code at generation time
(`autogate_bench.asr_noise`). Every paraphrase inherits its seed's split,
style and slots.

```
uv run python -m autogate_bench.paraphrases validate --strict [--intents a,b]
uv run python -m autogate_bench.paraphrases status
uv run python -m autogate_bench.generate --out data/generated/full --paraphrases data/paraphrases
```

`validate` prints counts per intent and variant and every error and warning
as `file:seed:line`, and exits non-zero on an error; `status` shows how far
each intent is from its quota. To regenerate paraphrases with your own
Anthropic key instead (prompt in `prompts/paraphrase.md`):

```
uv sync --extra api
uv run python -m autogate_bench.paraphrase_api --intents call_contact,wipers_on --out data/paraphrases --resume
```

## The paraphrased dataset

`data/paraphrases/` holds 40 hand-written paraphrases per seed (11,240 lines,
all passing strict validation). Generate the full dataset and score the rules
baseline with:

```
uv run python -m autogate_bench.generate --out data/generated/full --rng-seed 0 --paraphrases data/paraphrases
uv run python -m autogate_eval.baselines.rules --rows data/generated/full/rows.jsonl
```

Numbers and what they mean: `docs/dataset-v0.md`. Lines flagged for human
review: `docs/paraphrase-review-notes.md`.

## Train the router

The router is Qwen3-0.6B with LoRA and three heads (route, intent, sensitive
spans), trained on the paraphrased dataset. A CPU smoke run with a tiny random
backbone, nothing downloaded:

```
uv sync --extra dev --extra train --extra export
uv run python -m autogate_router.tiny --out checkpoints/tiny-qwen3 --rows data/generated/pilot/rows.jsonl
uv run python -m autogate_router.train --config configs/train_smoke.yaml
```

The full run (`configs/train_qwen3_0.6b.yaml`), the no-context ablation,
calibration, ONNX export, the Kaggle script and how to score any
`predictions.jsonl` with `python -m autogate_eval.score` are in
`docs/training.md`.

## Development

```
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

## License

Apache 2.0. See `LICENSE`.
