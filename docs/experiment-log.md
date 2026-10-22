# AutoGate experiment log

The running memory of the project: what we are trying, why, what each run
produced, and what is still open. Newest results go at the top of each
section. Every run gets an entry in "Runs", even a failed one.

Last updated: 2026-10-22. Linked from `CLAUDE.md` so every session reads it.

## How to use this file

- **Before starting work:** read "Current state" and "Open questions".
- **After a run:** add a row to the runs table and a short entry under
  "Run details" with the config, the commit, and what it showed.
- **After a decision:** add a line to the decision log with the date and why.
- Detailed write-ups stay in their own docs (linked below). This file is the
  index and the short version.

## Current state

| item | status |
| --- | --- |
| Benchmark v0 | done: 65 intents, 281 seeds, 11,240 paraphrases, 53,184 rows (`docs/dataset-v0.md`) |
| Rules baseline | done, both all-seeds and fair train-only versions |
| Router v0 (route + intent + span heads) | trained once on Colab T4; beats rules in distribution, fails on unseen intents (`docs/results-v0.md`) |
| Router v1 (tag heads + regularisation) | code done and pushed (commit `723573c`); **waiting for the Colab run** |
| No-context ablation | not run yet; included in the v1 notebook |
| int8 export | broken by dynamic quantization (route agreement 0.915); recipe fix not started |
| LLM zero-shot baselines, paper outline, Table 4 | built in another session on branch `claude/vibrant-hamilton-1psf5i`; **not merged**; labels verified identical |

## Goal and success criteria

From the PRD (`roadgate-prd.md`). The router reads `(utterance, context)` and
picks one of five routes: LOCAL, CLOUD, CLOUD_MASKED, DEFER, REFUSE.

| metric | target |
| --- | --- |
| macro-F1 over 5 routes, test | > 0.85 |
| safety-weighted error (SWE) | < 40% of rules baseline |
| critical miss rate | < 1% |
| model size | < 1B params, < 600 MB int8 |
| on-device latency p95 | < 50 ms |

**Critical miss** = a safety-critical command not run locally, or a REFUSE
case that executes anyway.

## Approaches

### A1. Policy-first labels (done)

Labels come from a deterministic rulebook, not from an LLM's judgment. Each
intent carries four hand-written tags; the rulebook combines tags with the
car's context.

- **Tags:** capability (local_ok, needs_small_local, cloud_preferred,
  needs_cloud), distraction (low, medium, high), actuation (none, comfort,
  safety_critical, restricted), consequence (none, reversible, irreversible).
- **Policy tables:** driving demand from speed x workload; distraction gate;
  actuation gate. Editable as JSON (`src/autogate_bench/policy.py`).
- **Precedence:** safety-critical -> LOCAL; restricted while moving -> REFUSE;
  too distracting -> REFUSE; local-capable -> LOCAL; no signal -> DEFER (or
  LOCAL for cloud_preferred); sensitive spans -> CLOUD_MASKED (or LOCAL under
  strict privacy); else CLOUD.

### A2. Data: seeds, paraphrases, noise (done)

- 4 to 5 hand-written seeds per intent, each a distinct phrasing style
  (command, outcome, question, indirect, fragment).
- 40 paraphrases per seed: 24 plain, 6 Hinglish, 4 Kannada-English,
  6 disfluent. Written by Claude agents against `docs/paraphrase-spec.md`.
- 4 deterministic ASR-noise variants per seed.
- Slots filled from synthetic pools; sensitive spans recorded as offsets.
- 3 contexts per utterance, sampled toward route boundaries.
- Splits by seed (80/10/10); 6 intents held out entirely as OOD.

### A3. Router v0: end-to-end route head (done, run 1)

Qwen3-0.6B + LoRA r=16 on all projections; heads for route (5-way, cost
weighted), intent (65-way), span (BIO). Route comes straight from the route
head.

### A4. Router v1: tag heads + regularisation (built, run 2 pending)

**Why:** run 1 learned intent identity, so unseen intents defaulted to LOCAL.
Tags are what the rulebook uses and every intent has them, so predicting tags
should transfer to unseen intents.

- Four extra heads predict the tags from the sentence.
- Three route sources from one forward pass:
  - `e2e`: the route head, as in v0
  - `tags`: the rulebook applied to the predicted tags plus context
  - `combined`: the more cautious of the two
- Conservative tag decoding: actuation is safety_critical or restricted if
  p >= 0.3; distraction is high if p >= 0.3.
- Tag losses weighted by danger: actuation 1.0, distraction 1.0, capability
  0.5, consequence 0.3. Class weights are 1/sqrt(freq) x a danger factor.
- Regularisation: 2 epochs, lr 1e-4, LoRA r=8 attention-only, LoRA dropout
  0.1, head dropout 0.3, weight decay 0.05, intent weight 0.2, label
  smoothing 0.1, context dropout 0.2, word dropout 0.1.
- Config: `configs/train_qwen3_0.6b_tags.yaml`. Details: `docs/training.md`,
  "Tag heads".

### Ideas not yet tried

- More seeds per intent (6 to 8). Most likely fix for the val-to-test gap.
- On-the-fly slot refill and ASR noise each epoch (currently fixed at
  generation time).
- Early stopping on val SWE.
- int8 fix: per-channel weight-only quantization, or static quantization with
  val calibration; require route agreement > 0.99.
- Cross-validation over seed folds for tighter confidence intervals.
- Raise the sensitive rate from 16% toward the PRD's 35%.
- Real in-car utterances (KVRET) as a second OOD set, to answer the
  single-model-paraphrase concern.

## Runs

All numbers on held-out splits. SWE = safety-weighted error, lower is better.

### Test split (new phrasings of known intents)

| run | date | macro-F1 | SWE | critical miss | span leak | notes |
| --- | --- | --- | --- | --- | --- | --- |
| R0a rules, seeds only (pilot) | 09-25 | 0.875 | 0.315 | 0% | - | sanity check; too easy by design |
| R0b rules, all seeds | 09-25 | 0.627 | 0.919 | 22.5% | 85.7% | matcher saw test seeds; optimistic |
| R0c rules, train seeds only | 10-07 | 0.430 | 1.256 | 37.1% | - | the fair baseline |
| R1 router v0, e2e | 10-07 | **0.780** | **0.426** | **7.1%** | 33.4%* | Colab T4, 1h43m |
| R2 router v1 tags, e2e | pending | | | | | |
| R2 router v1 tags, tags | pending | | | | | |
| R2 router v1 tags, combined | pending | | | | | |
| R3 router v1 tags, no context | pending | | | | | ablation |

### OOD split (6 intents never seen in training)

| run | macro-F1 | SWE | critical miss | notes |
| --- | --- | --- | --- | --- |
| R0b rules, all seeds | 0.583 | 1.044 | 34.1% | not fair: saw OOD seeds |
| R0c rules, train seeds only | 0.231 | 2.231 | 78.6% | |
| R1 router v0, e2e | 0.442 | 1.312 | 25.7% | defaults to LOCAL on unseen intents |
| R2 router v1, tags | pending | | | the key number for A4 |

\* Span leak for R1 is measured on test + OOD combined.

## Run details

### R2 / R3: router v1 with tags (pending)

- Commit `723573c`, notebook `notebooks/kaggle_train.ipynb` (12 cells).
- Smoke test on a tiny random model only (not meaningful): OOD critical
  misses 22/22 for e2e vs 4/22 for tags. Direction is encouraging.
- Look at: OOD table for `tags` vs `e2e`; the per-tag "route-changing errors"
  column to see which tag hurts most.

### R1: router v0 (2026-10-07)

- Qwen3-0.6B, LoRA r=16 all projections, 3 epochs, batch 32, lr 2e-4.
  10.2M of 606M params trained.
- Val by epoch: SWE 0.496 -> 0.388 -> 0.336; macro-F1 0.806 -> 0.828 ->
  0.859. Train loss reached 0.002: memorisation.
- Test intent accuracy 0.503 (no better than keywords); OOD 0.000 by
  construction.
- Errors: LOCAL over-predicted. 276 of 321 critical misses were REFUSE
  predicted as LOCAL.
- Calibration: temperature 3.15, ECE 0.080 -> 0.016. Threshold 0.95 gives
  under 1% critical misses on 54% coverage; 0.80 gives 4.0% on 82%.
- Export: fp32 2,275 MB, exact. int8 571 MB, route agreement 0.915, intent
  0.707: unusable.
- Conclusion: overfit to training phrasings; learned intent identity, not
  transferable semantics. Led to A4.
- Full write-up: `docs/results-v0.md`.

### R0: rules baselines (2026-09-25, 10-07)

- Keyword intent matcher from seeds + regex PII detector + the rulebook.
- 2026-10-22: the v1 notebook's rules cell on Colab reproduced R0b exactly
  (test macro-F1 0.627, critical miss 197/876; OOD 0.583). The dataset
  regenerates identically there, so R2 is directly comparable to R1.
- Seed-only data: intent accuracy 0.99. Paraphrased data: 0.52. Confirms the
  benchmark is not too easy (PRD threshold: rebalance if rules > 0.9 F1).
- Regex finds 19% of sensitive spans; misses lowercase names.
- Write-ups: `docs/pilot.md`, `docs/dataset-v0.md`.

## Decision log

| date | decision | why |
| --- | --- | --- |
| 10-22 | Add tag heads; score three route sources | R1 failed on unseen intents; tags transfer |
| 10-22 | Regularise run 2 (see A4) | R1 train loss 0.002, val-test gap |
| 10-07 | Fair rules baseline uses train seeds only | all-seeds matcher saw test and OOD seeds |
| 10-07 | Notebook runs on Colab and Kaggle; outputs can go to Drive | user trains on Colab; disk wiped on disconnect |
| 10-07 | Repo made public | Colab and Kaggle clone without a token |
| 09-25 | Paraphrases written by Claude agents, not a paid API | free under the subscription; API script kept for reproducibility |
| 09-25 | Router outputs route, intent, spans; orchestrator acts on them | the models never see router output; code does |
| 09-25 | Add `cloud_preferred` capability | offline find-nearby should fall back to onboard data, not wait |
| 09-23 | Add `consequence` tag; confirmation is a gate rule, not a route | confirmation depends on the action, not where it runs |
| 09-23 | Intent count 65 incl. AC, recirculation, seat ventilation | climate is the most common in-car request |
| 09-22 | Three policy tables; distraction blocks are REFUSE not DEFER | deferring would auto-run later without being asked |
| 09-22 | Five routes, no CLARIFY in v1; workload is an input | keep v1 comparable to baselines |
| 09-21 | Repo and packages named AutoGate | owner's choice |

## Open questions

- Keep self-correcting paraphrases that name a sibling intent first? Listed
  in `docs/paraphrase-review-notes.md`.
- Kannada-English spellings need a native speaker's check.
- Strict privacy + sensitive spans on a needs-cloud intent routes LOCAL; should
  it be REFUSE?
- Merge `claude/vibrant-hamilton-1psf5i` (LLM baselines, paper outline,
  Table 4)?
- Create a `main` branch before release.
- Which "ask the driver" threshold to ship (scan in R1 details).

## Where things are

| what | where |
| --- | --- |
| intents and tags | `src/autogate_bench/intents.py`, `docs/taxonomy.md` |
| policy tables and rulebook | `src/autogate_bench/policy.py`, `rulebook.py` |
| seeds and paraphrases | `data/seeds.yaml`, `data/paraphrases/` |
| dataset generation | `python -m autogate_bench.generate` |
| rules baseline | `python -m autogate_eval.baselines.rules` |
| training, predict, score | `src/autogate_router/`, `src/autogate_eval/score.py`, `docs/training.md` |
| training notebook | `notebooks/kaggle_train.ipynb` |
| run outputs | Colab `/content/runs` or Drive `MyDrive/autogate/runs`; not in git |
