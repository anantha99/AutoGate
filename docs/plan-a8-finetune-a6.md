# Plan A8: fine-tune the decision encoder on the A6 data

A phased plan for an implementing agent. It follows A7
(`docs/plan-a7-decision-router.md`), which was dropped by its own stop rule,
and reuses A7's code (`src/autogate_decide/`).

Status: **planned, nothing run.** Written 2026-11-18. **Blocked** on the
owner's audit of `docs/augment-review.md` (A6 data may not be trained on
before it).

## 1. What we are testing

**Where A7 ended.** No A7 configuration passed Gate 2 on test and OOD
together:

| run | test macro-F1 / SWE / critical miss | OOD SWE / critical miss |
| --- | --- | --- |
| R4nc tags (reference) | 0.850 / 0.342 / 6.0% | 1.038 / 22.9% |
| R6a frozen EmbeddingGemma 2 + heads | 0.800 / 0.562 / 9.7% | **1.015 / 21.7%** |
| R6c fully fine-tuned (A7 recipe) | **0.867 / 0.308 / 4.8%** | 1.166 / 30.5% |

Fine-tuning fixed known intents and broke unseen ones. On OOD, R6c misses
88% of high-distraction rows (R6a 80%), and its capability and distraction
probabilities are confidently wrong (ECE 0.34 and 0.32).

**Hypothesis.** The OOD collapse comes from training on too few distinct
goals per tag value (only 7 training intents are high distraction). A6 adds
60 synthetic goals (20 of them high distraction) and 290 minimal pairs.
Fine-tuning on that data, with a guard against drifting away from the
frozen encoder, keeps R6c's test results and brings OOD back to at least
R4's level.

**Why it could fail.**

- 60 synthetic goals written by agents may be too uniform to teach
  "distraction" as a property rather than a list of phrasings.
- The A6 labels are judgement calls that nobody but the owner checks.
- The collapse may come from supervised fine-tuning itself, not from the
  data. That is why the plan has an anti-drift variant (section 5) and a
  no-A6 control.
- Selection still cannot see the real OOD set. The proxy holdout
  (section 4) is a guess at what OOD looks like.

**What it does not prove.** As in A7: synthetic benchmark, not real
drivers. It also does not separate "A6 helps" from "A6 helps this
encoder". R5 (R4's Qwen router on the same data) answers the first
question, and A8 reports beside it when R5 exists.

## 2. Rules for the implementing agent

Everything in section 3 of the A7 plan still applies, in particular:

- **Labels do not change.** The augmented dataset is
  `generate --augment data/augment`. Check that the first 66,414 lines are
  byte-identical to the sha256 `8cc12a4b...` file, and record the augmented
  file's sha256 in every manifest.
- **The audit comes first.** Do not train on `data/augment` until the
  owner has audited `docs/augment-review.md`. Use any corrections the audit
  makes; regenerate and re-record the sha256.
- **Pre-register, then score once.** Fit on train. Select on val and on
  the proxy holdout (section 4). Commit the frozen configuration, then
  score test and OOD **once**.
- **Test reuse.** A7 used the test and OOD sets three times (R6a, R6a2,
  R6c). A8 adds exactly one more use, for the single final configuration.
  State that count next to the result. Nothing is tuned after it.
- fp32 everywhere. CPU or Colab T4; record the torch version, device and
  thread count. Checkpoints and caches are gitignored.
- Use the A7 code paths: `autogate_decide.finetune`, `embed`,
  `run --selection val`, and the per-epoch resume and per-trial selection
  checkpoints (container restarts were hourly in A7).
- Every run gets a runs-table row in `docs/experiment-log.md`, including
  failed ones. Run ids: **R7x** (R5 stays reserved for R4 on the A6 data).

**Stop and ask the owner when:**

- the audit is not done, or its corrections change more than a handful of
  labels (that is a new A6, and the minimal-pair probe in section 4 must be
  re-split);
- the selection step (Phase 2) finds no configuration that beats the
  no-A6 control on the proxy holdout (outcome C below);
- Phase 5 needs a quantised artefact of a fine-tuned encoder (section 7).

## 3. Reference numbers

Use the A7 table above. A8 is compared with R6c (same recipe, no A6) and
R6a (frozen), and with R5 when it exists. Gate 2 and the verdict
thresholds are unchanged from the A7 plan. The owner confirmed the
"proposed" ones on 2026-11-17.

## 4. Held-out data for selection

A7's selection saw only val, which holds training intents, so it could not
see the OOD collapse. A8 adds two held-out sets, both carved out of
training data and never trained on during selection.

**Proxy holdout (8 training intents).** Pick 8 training intents the way
`autogate_bench.splits` picked the real OOD set: balanced by tag, covering
every value of every tag, at least one high-distraction intent and one
safety-critical intent. During selection runs, their seeds, paraphrases and
ASR copies are removed from training. So are A6 minimal pairs that use them,
and synthetic goals within token Jaccard 0.5 of any of their seeds (the
same guard A6 used against the real OOD set). Write the pick as a fixed
list in `configs/decide/a8_proxy_holdout.yaml`, with a test that checks the
balance. Commit it before any run.

**Probe pairs (20% of the minimal pairs).** Hold out 58 of the 290 pairs,
stratified by `differs_in`. Pick whole intent pairs, so that no intent pair
has some pairs trained and some probed. Never train on them, in any phase.
They replace A7's minimal-pair probe, which A6 training contaminates.
Commit the list in `configs/decide/a8_probe_pairs.yaml`.

Selection scores, per epoch and per configuration:

- **val** (training intents, new phrasings): conservative-decoder SWE and
  critical miss, as in A7;
- **proxy holdout** (unseen intents): the same metrics, plus distraction
  dangerous-miss rate and capability accuracy;
- **probe pairs:** share of pairs with every routing tag right on both sides.

## 5. Configurations

Fine-tune recipe from R6c unless stated: EmbeddingGemma 2 text tower
`f9972ac2`, mean pooling, heads for the four tags, sensitive and joint;
loss weights as R6c; batch 32; heads lr 1e-3; warmup 6%; 3 epochs.

| id | encoder update | A6 data | anti-drift | role |
| --- | --- | --- | --- | --- |
| R7-ctl | full, lr 2e-5 | no | no | control: R6c's recipe with the proxy holdout removed. Measures where R6c already stood on unseen intents |
| R7a | full, lr 2e-5 | yes | no | the data effect alone |
| R7b | full, lr 2e-5 | yes | distillation, beta = 1 | data plus a pull towards the frozen geometry |
| R7c | LoRA r16 (q, k, v, o), lr 1e-4 | yes | no | smaller change to the encoder |

**Distillation (R7b).** Add `beta * (1 - cos(z_ft, z_frozen))` to the loss,
where `z_frozen` is the frozen encoder's pooled vector for the same text
(already in the A7 embedding cache). It keeps unseen inputs where the frozen
encoder put them. R6a showed that geometry generalised as well as R4's.
Beta is fixed at 1 and not tuned. Add it to `finetune.py` with a test.

**Synthetic goals in the loss.** They have no intent and train the tag,
sensitive and joint heads only, as in A6. The joint vocabulary is
rebuilt from train plus goals: goals add combinations no intent has. Goal
rows count once per utterance, like every other row.

## 6. Phases

### Phase 0: prerequisites

1. Owner audit of `docs/augment-review.md` done; corrections applied;
   augmented file regenerated; prefix and sha256 checks pass.
2. Proxy holdout and probe-pair lists committed, with their tests.
3. `finetune.py` takes the augmented rows, an intent holdout list, an
   excluded-pairs list and `--distill-weight`. It reports proxy-holdout and
   probe-pair metrics per epoch. Tests on the tiny backbone; no downloads.
4. Embed the augmented texts with the frozen encoder if the A7 cache lacks
   any (it was built from both rows files, so it should not).

### Phase 1: selection runs (R7-ctl, R7a, R7b, R7c)

Each run trains for three epochs with the proxy holdout and probe pairs
removed. It records every epoch's val, proxy and probe scores. On CPU, a
full fine-tune took about 26 minutes an epoch in A7, so expect roughly 6
hours for all four. A Colab T4 is much faster; the owner chooses which.

**Selection rule (pre-registered).** Over every (configuration, epoch)
except R7-ctl:

1. Keep those whose val SWE is at most 0.37 and val critical miss at most
   7.0% (Gate 2's test bars, applied to val).
2. Among them, take the lowest proxy-holdout critical miss. Break ties
   (within 1 point) by the lower proxy-holdout SWE.

**Outcomes:**

| outcome | condition | next |
| --- | --- | --- |
| A | the selected run beats R7-ctl on proxy-holdout critical miss by at least 3 points and on proxy SWE | Phase 2 |
| B | something is selected but it does not beat R7-ctl on the proxy holdout | stop: A6 does not fix the collapse for this encoder. Log it; R5 decides whether A6 helps at all |
| C | nothing passes the val filter | stop and ask the owner |

### Phase 2: final model (R7)

1. Retrain the selected configuration on **all** training data: the proxy
   holdout returns, the probe pairs stay out. Use the selected number of
   epochs and no further selection.
2. Re-embed the dataset with the merged encoder. Check that the five probe
   sentences reproduce their training-time vectors.
3. `run --method heads --selection val` on the augmented rows: linear
   heads, weight decay by val NLL, lambda 0.5, temperatures and abstention
   threshold on val. Commit the frozen configuration.
4. Score test and OOD once. Apply Gate 2 (A7 plan) on the conservative
   decoder without abstention, and report expected cost beside it. Report
   the probe-pair score, the per-variant table, OOD per intent, and the
   intent-bootstrap intervals, as in A7.

**If Gate 2 fails: stop.** Log A8 as dropped with the numbers. A third
fine-tuning plan would need new data, not new knobs.

### Phase 3: registry and safety lane (only after Gate 2 passes)

As A7 Phase 3 (code exists, tested, never run): fair and registered
registry modes, safety lane. Report with and without them.

### Phase 4: device reality (only after Gate 2 passes)

As A7 Phase 5, with one change. Google's shipped int4 LiteRT file holds
the **base** weights, so a fine-tuned encoder needs its own quantised
artefact. Try, in order:

1. a LiteRT-LM bundle built from the merged weights (`litert-lm-builder`
   is in the `litert-lm` 0.19.2 package; check that it supports
   EmbeddingGemma 2);
2. ONNX with weight-only per-channel int8. Do not use dynamic quantisation:
   it broke R1, R2 and R4.

Then re-embed with the artefact, retrain the heads on those vectors,
require route agreement >= 0.99 with fp32, and measure size and latency.

## 7. Verdict

Use the A7 plan's section 7 thresholds unchanged (U1-U7, owner-confirmed).
How to read the result:

- **Gate 2 passes and U1-U7 pass:** adopt; R4 becomes the comparison.
- **Gate 2 passes, U4 or U6 fails:** a better router than R4 that is not
  release-ready. Name the failing rows; the next work is data for them.
- **Gate 2 passes but Phase 4 cannot produce a quantised artefact with
  agreement >= 0.99:** the result stands for fp32 only. Phase 4 becomes
  its own plan.

## 8. Run ids

| id | what |
| --- | --- |
| R7-ctl | R6c recipe, no A6, proxy holdout removed (selection control) |
| R7a, R7b, R7c | selection runs (section 5) |
| R7 | final model, scored once on test and OOD |
| R7q | quantised artefact and latency (Phase 4) |

For each: a runs-table row, a "Run details" entry with config, commit,
encoder revision, augmented-data sha256, device, and a decision-log line at
each outcome.

## 9. Open items for the owner

- **The A6 audit** (blocking).
- CPU (about 6 hours for Phase 1, run here) or Colab T4.
- Whether a fourth use of the test set is acceptable, or a fresh test set
  should be generated first. That would mean new paraphrases of the
  test seeds, which changes no labels but costs a paraphrase round.
