# AutoGate

An open router and benchmark that decides, per utterance, whether an in-car
LLM assistant should answer on the head unit (`LOCAL`), in the cloud
(`CLOUD`), in the cloud with sensitive spans masked (`CLOUD_MASKED`), wait
for connectivity (`DEFER`), or not run at all (`REFUSE`).

Status: week 1. The rulebook, policy tables, intent taxonomy, seed-only data
pipeline, cost matrix, metrics and rules baseline are in place; paraphrasing
and the router follow.

## Layout

| Package | Contents |
| --- | --- |
| `autogate_bench` | Schema, policy tables, rulebook labeler, intent taxonomy (`docs/taxonomy.md`), seeds, span injector, context sampler, splits, dataset writer |
| `autogate_router` | Training, export, inference (week 3) |
| `autogate_eval` | Cost matrix, safety-weighted error and other metrics, rules baseline, policy gate (weeks 1-3) |
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

## Development

```
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

## License

Apache 2.0. See `LICENSE`.
